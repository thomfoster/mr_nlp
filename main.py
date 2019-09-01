import os
import torch
import logging
import sys

import torch.nn as nn
import torch.utils.data as D
import matplotlib.pyplot as plt
import numpy as np
import math

from utils import IndividualFileDataset, Batch, collate_fn
from models import Bert, Classifier, Summarizer
from random import shuffle

def get_filepaths(file_type):
    "Get all the bert_data files" 
    rootdir = './data'
    filepaths = []
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if file_type in file:
                filepaths.append(os.path.join(subdir, file))
    return filepaths



if __name__ == '__main__':

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file and console handlers
    fh = logging.FileHandler('log.txt')
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    train_datasets = [IndividualFileDataset(fp) for fp in get_filepaths('train')]
    train_dataset = D.ChainDataset(train_datasets)

    train_loader = D.DataLoader(train_dataset, batch_size=6, collate_fn=collate_fn)

    language_model = Bert(temp_dir='./temp' , load_pretrained_bert=True, bert_config=None)
    finetune_model = Classifier(hidden_size=768)

    model = Summarizer(language_model, finetune_model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Main training loop
    logger.info("Starting training...")

    for ep in range(5):
        model.train()
        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            loss = torch.Tensor([0.0]).to('cpu')
            output = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0] # select sent scores
            print('outputs shape:',output.shape)
            print(f'labels shape:{batch.labels.shape}, labels type:{batch.labels.type()}')
            loss += criterion(output, batch.labels)
            loss.backward()

            if (idx+1)%6==0:
                print('Backprop on accumulated grads')
                # every 10 iterations, update parameters
                optimizer.step()
                optimizer.zero_grad()

            if (idx+1)%10==0:

                valid_datasets = [IndividualFileDataset(fp) for fp in get_filepaths('valid')]
                shuffle(valid_datasets)
                valid_dataset = D.ChainDataset(valid_datasets)
                valid_loader = D.DataLoader(valid_dataset, batch_size=6, collate_fn=collate_fn)

                tp = tn = fp = fn = 0

                for jdx, batch in enumerate(valid_loader):
                    print('Running through validation batch')
                    print('val labels shape: ', batch.labels.shape)
                    outputs = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0] # select sent scores
                    print(outputs)
                    outputs = (outputs>0.5).type(torch.int)
                    print('val outputs shape:',outputs.shape)
                    tp += ((outputs == 1) * (batch.labels == 1)).sum().item()
                    tn += ((outputs == 0) * (batch.labels == 0)).sum().item()
                    fp += ((outputs == 1) * (batch.labels == 0)).sum().item()
                    fn += ((outputs == 0) * (batch.labels == 1)).sum().item()
                    if jdx == 5:
                        break

                cf = np.array([[tp, fp],[fn, tn]])
                print(cf)


                print(f'Completed {idx} iterations, loss: {loss.item()}')

        logger.info("Epoch %s complete. Loss: %s" % (ep, loss))
        fn = 'chkpt'+str(ep)+'.pth.tar'
        torch.save({
            'epoch': ep + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, fn)
        print(f'saved model after epoch: {ep}')
