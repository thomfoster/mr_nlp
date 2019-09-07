import os
import torch
import logging
import sys

import torch.nn as nn
import torch.utils.data as D
import numpy as np
import math

from utils import IndividualFileDataset, Batch, collate_fn, gen_loader
from models import Bert, Classifier, Summarizer
from random import shuffle

from agent import GeneiAgent

import agent

import argparse

if __name__ == '__main__':


    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
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



    parser = argparse.ArgumentParser(description='Genei V1')

    # Optimizer, batch size and gradient accumulation, and number of steps parameters
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--bs', default=6, type=int, help='batch size')
    parser.add_argument('--grad_accum_steps', default=6, type=int, help='gradient accumulation')
    parser.add_argument('--steps', default=150_000, type=int, help='total number of steps')
    parser.add_argument('--alpha', default=0.1, type=float, help='label-smoothing parameter')
    parser.add_argument('--resume_chkpt_path', default=None)
    parser.add_argument('--save_chkpt_dir', default=None, help='Give a directory to save checkpoints to')
    parser.add_argument('--save_chkpt_freq', default=10_000, type=int, help='Frequency at which chkpts are saved')
    parser.add_argument('--use_S3', default=False)

    args = parser.parse_args()

    logger.info(args)


    # Load in data
    train_loader = gen_loader(args, collate_fn, type='train')
    valid_loader = gen_loader(args, collate_fn, type='valid')

    # Initialize BERT and fine-tune models
    language_model = Bert(temp_dir='./temp' , load_pretrained_bert=True, bert_config=None)
    finetune_model = Classifier(hidden_size=768)

    # Wrap model together
    model = Summarizer(language_model, finetune_model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Agent
    genei = GeneiAgent(model=model, optimizer=optimizer)

    if args.resume_chkpt_path is not None:
        if args.use_S3:
            genei.load_chkpt_from_S3(resume_chkpt_path)
        else:
            genei.load_chkpt(resume_chkpt_path)

    genei.train(train_loader = train_loader,
                valid_loader = valid_loader,
                lr = args.lr,
                tot_training_steps = args.steps,
                grad_accum_steps=args.grad_accum_steps,
                alpha=args.alpha,
                save_chkpt_dir=args.save_chkpt_dir,
                save_chkpt_freq=args.save_chkpt_freq,
                use_S3=args.use_S3
                )


    #
    # # Load checkpoint if checkpoint given
    # if args.resume is not None:
    #     logger.info('==> Resuming from checkpoint..')
    #     assert os.path.isfile(args.chkpt), 'Error: checkpoint file does not exist'
    #     checkpoint = torch.load(args.chkpt)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     step =checkpoint['step']
    #
    # # Use GPU if possible
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # logger.info('Using device: ',device)
    #
    # # Main training loop
    # logger.info("Starting training...")
    # while step < args.steps:
    #     optimizer.zero_grad()
    #     for idx, batch in enumerate(train_loader):
    #         # learning rate schedule
    #         optimizer.param_groups[0]['lr'] = lr_schedule(args.lr,step)
    #
    #         # put model in train mode, so it does dropout
    #         model.train()
    #
    #         # Forward prop, compute output through LM and finetune layer
    #         output = model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0] # select sent scores
    #         logger.debug('outputs shape:',output.shape)
    #         logger.debug(f'labels shape:{batch.labels.shape}, labels type:{batch.labels.type()}')
    #
    #         # Compute loss and backprop
    #         loss = criterion(output, batch.labels).to(device)
    #         loss.backward()
    #
    #         # Gradient accumulation every N batches (default value: every 6 batches)
    #         if (idx+1)%args.grad_accum==0:
    #             logger.debug('Backprop on accumulated grads')
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #         # Update step
    #         step += 1
    #
    #         # Print steps every 50 batches
    #         if (idx+1)%50==0:
    #             logger.info(f'Steps: {step}')
    #
    #         # Validating model every 1000 batches
    #         if (idx+1)%1000==0:
    #             with torch.no_grad():
    #                 # Make and shuffle validation dataset
    #                 valid_datasets = [IndividualFileDataset(fp) for fp in get_filepaths('valid')]
    #                 shuffle(valid_datasets)
    #                 valid_dataset = D.ChainDataset(valid_datasets)
    #                 valid_loader = D.DataLoader(valid_dataset, batch_size=6, collate_fn=collate_fn)
    #
    #                 # Do validation
    #                 validate(valid_loader, model, criterion=criterion, iter_till_break=100)
    #
    #         # Saving model
    #         if (idx+1)%5000==0:
    #             checkpoint_number += 1
    #             checkpoint_number = checkpoint_number % 15
    #             fn = './checkpoints/chkpt'+str(checkpoint_number)+'.pth.tar'
    #             torch.save({
    #                 'step': step,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer' : optimizer.state_dict(),
    #             }, fn)
    #
    #             logger.info('Model saved')
