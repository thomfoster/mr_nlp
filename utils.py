import torch
import os
import random
from random import shuffle
import torch.utils.data as D
import numpy as np


def get_filepaths(file_type):
    "Get all the bert_data files"
    rootdir = './data'
    filepaths = []
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if file_type in file:
                filepaths.append(os.path.join(subdir, file))
    return filepaths


class IndividualFileDataset(D.IterableDataset):
    def __init__(self, fp):
        super(IndividualFileDataset).__init__()
        self.fp = fp

    def __iter__(self):
        docs = torch.load(self.fp)
        for s in docs:
            yield s


def gen_loader(args, collate_fn, type='train'):
    '''
    Function for generating a torch.utils.data.DataLoader object for our purposes
    '''
    datasets = [IndividualFileDataset(fp) for fp in get_filepaths(type)]
    shuffle(datasets)
    dataset = D.ChainDataset(datasets)
    loader = D.DataLoader(dataset, batch_size=args.bs, collate_fn=collate_fn)

    return loader


class Batch:
    # Helper class to return all the important things from a summarisation dataset
    def __init__(self, src, segs, clss, labels, mask_attn, mask_clss, src_txt):
        self.src = src
        self.segs = segs
        self.clss = clss
        self.labels = labels
        self.mask_attn = mask_attn
        self.mask_clss = mask_clss
        self.src_txt = src_txt
        

def _binary_smooth(label, alpha=0.1):
    return label*(1-alpha) + alpha*.5


def _yang_pad(data, pad_id):
    # Pads data to its max length along first (sequence length) axis
    width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def collate_fn(batch):
    '''
    Function to preprocess the batch - adds yangpadding to src, segs, clss and labels
    :param batch:
    :return:
    '''

    src = _yang_pad([s['src'] for s in batch], 0)
    segs = _yang_pad([s['segs'] for s in batch], 0)
    clss = _yang_pad([s['clss'] for s in batch], -1)
    labels = _yang_pad([s['labels'] for s in batch], -1)

    # Tag each batch with the original text to allow easier inspection of results
    # Shouldn't be a problem for GPU memory as we don't push it to GPU
    src_txt = _yang_pad([s['src_txt'] for s in batch], -1)

    # Ensure that masks initially specified as 0s and 1s
    # are converted to float32 tensors
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    src = torch.Tensor(src).type(torch.long).to(device)
    segs = torch.Tensor(segs).type(torch.long).to(device)
    clss = torch.Tensor(clss).type(torch.float32).to(device)
    labels = torch.Tensor(labels).type(torch.float32).to(device)

    # Self attention mask to deal with variable sentence lengths inside bert itself
    mask_attn = 1 - (src == 0).type(torch.float32)
    mask_attn = mask_attn.to(device)

    # Self attention mask to deal with variable sentence length in fine tuning layers
    mask_clss = 1 - (clss == -1).type(torch.float32)
    mask_clss = mask_clss.to(device)

    return Batch(src, segs, clss, labels, mask_attn, mask_clss, src_txt)


def _cf(outputs, labels):
    tp = ((outputs == 1) * (labels == 1)).sum()  # True positives
    tn = ((outputs == 0) * (labels == 0)).sum()  # True negatives
    fp = ((outputs == 1) * (labels == 0)).sum()  # False positives
    fn = ((outputs == 0) * (labels == 1)).sum()  # False negatives

    return torch.tensor([tn, fp, fn, tp]).reshape(2, 2).type(torch.int)


def _mcc(tn, fp, fn, tp):  # Log then exp for numerical stability
    corr = tn * tp - fn * fp
    den = .5 * \
        torch.log(torch.tensor([tp + fp, tp + fn, tn + fp, tn + fn])).sum()
    if corr > 0:
        num = torch.log(corr)
        return torch.exp(num - den)
    elif corr < 0:
        num = torch.log(-corr)
        return -torch.exp(num - den)
    else:
        return 0
