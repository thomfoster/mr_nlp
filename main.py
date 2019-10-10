import torch
import logging


from utils import collate_fn, gen_loader
from models import Bert, Classifier, Summarizer

from agent import GeneiAgent

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
    parser.add_argument('--val_freq', default=300, type=int, help='frequency at which validation is done')
    parser.add_argument('--resume_chkpt_path', default=None)
    parser.add_argument('--save_chkpt_dir', default=None, help='Give a directory to save checkpoints to')
    parser.add_argument('--save_chkpt_freq', default=10_000, type=int, help='Frequency at which chkpts are saved')
    parser.add_argument('--use_S3', default=False)

    args = parser.parse_args()
    for k,v in vars(args).items():
        logger.info(str(k) + ': ' + str(v))
    
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
            genei.load_chkpt_from_S3(args.resume_chkpt_path)
        else:
            genei.load_chkpt(args.resume_chkpt_path)

    genei.train(train_loader = train_loader,            # torch.utils.dataset.Dataloader object containing train data
                valid_loader = valid_loader,            # torch.utils.dataset.Dataloader object containing valid data
                lr = args.lr,                           # learning rate
                tot_training_steps = args.steps,        # total number of training steps
                grad_accum_steps=args.grad_accum_steps, # number of steps before we backprop the gradient
                alpha=args.alpha,                       # label smoothing
                val_freq=args.val_freq,                 # frequency at which we validate our model performance
                save_chkpt_dir=args.save_chkpt_dir,     # directory to save checkpoint to
                save_chkpt_freq=args.save_chkpt_freq,   # frequency at which we save checkpoints
                use_S3=args.use_S3                      # use S3 or not?
                )
