import torch
import torch.utils.data as D


class IndividualFileDataset(D.IterableDataset):
    def __init__(self, fp):
        super(IndividualFileDataset).__init__()
        self.fp = fp

    def __iter__(self):
        docs = torch.load(self.fp)
        for s in docs:
            tup = (s['src'], s['segs'], s['clss'], s['labels'])
            yield tup


class Batch:
    def __init__(self, src, segs, clss, labels, mask_attn, mask_clss):
        self.src = src
        self.segs = segs
        self.clss = clss
        self.labels = labels
        self.mask_attn = mask_attn
        self.mask_clss = mask_clss


def _yang_pad(data, pad_id):
    width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def collate_fn(batch):
    src = _yang_pad([s[0] for s in batch], 0)
    segs = _yang_pad([s[1] for s in batch], 0)
    clss = _yang_pad([s[2] for s in batch], -1)
    labels = _yang_pad([s[3] for s in batch], 0)

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

    return Batch(src, segs, clss, labels, mask_attn, mask_clss)
