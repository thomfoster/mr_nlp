import torch, s3fs, shutil, os
import torch.nn as nn
import numpy as np
import logging

from utils import _cf, _binary_smooth, _mcc


def lr_schedule(lr, step):
    return lr * min(step**.5, step * 30_000**-1.5)


class GeneiAgent():
    def __init__(self, model, optimizer=None, criterion=None):
        super(GeneiAgent, self).__init__()

        self.logger = logging.getLogger('__main__')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Using device: {self.device}')

        self.step = 0
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.valid_loss = 0
        self.cf = 0
        self.mcc = 0


    def gen_chkpt_filename(self):
        return 'chkpt_step_'+str(self.step)+'mcc_'+str(round(self.mcc.item(), 3))+'.pth.tar'


    def save_chkpt(self, fn):
        torch.save({'step':self.step+1,
                    'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'cf': self.cf,
                    'mcc': self.mcc}, fn)
        self.logger.info(f'Saved checkpoint at step {self.step} and mcc {self.mcc}')


    def save_chkpt_to_local(self, dir):
        fn = os.path.join(dir, self.gen_chkpt_filename())
        self.save_chkpt(fn)


    def load_chkpt(self, chkpt_file):
        chkpt = torch.load(chkpt_file, map_location=self.device)
        self.step = chkpt['step']
        self.model.load_state_dict(chkpt['state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer'])
        self.cf = chkpt['cf']
        self.mcc = chkpt['mcc']
        self.logger.info(f'Loaded checkpoint with step {self.step}, cf:{self.cf}, mcc:{self.mcc}')


    def save_chkpt_to_AWS(self, aws_save_path):
        self.logger.info('Saving checkpoint to S3...')
        fn = os.path.join(aws_save_path, self.gen_chkpt_filename())
        fs = s3fs.S3FileSystem()
        with fs.open(fn,'wb') as f:
            self.save_chkpt(f)
            self.logger.info("saved checkpoint to AWS")


    @staticmethod
    def download_chkpt_from_S3(chkpt_file):
        fs = s3fs.S3FileSystem()
        with fs.open(chkpt_file) as f:
            logger = logging.getLogger(__name__)
            logger.info('Downloading checkpoint from S3')
            local_file = open(os.path.split(chkpt_file)[1], 'wb')
            shutil.copyfileobj(f, local_file)
            local_file.close()


    def load_chkpt_from_S3(self, chkpt_file):
        GeneiAgent.download_chkpt_from_S3(chkpt_file)
        fn = os.path.split(chkpt_file)[1]
        self.load_chkpt(fn)
        os.remove(fn)


    def train(self,
              train_loader,
              valid_loader,
              lr=0.002,
              tot_training_steps=150_000,
              grad_accum_steps=6,
              alpha=0.1,
              val_freq=300,
              save_chkpt_dir=None,
              save_chkpt_freq=10_000,
              use_S3=False):

        self.logger.info(f'Starting training from step: {self.step}')

        while self.step < tot_training_steps:

            self.optimizer.zero_grad() # Zero the gradient
            self.model.train() # Model in train mode so it does dropout and batch norm appropriately

            for idx, batch in enumerate(train_loader): # Iterate through the batches

                self.optimizer.param_groups[0]['lr'] = lr_schedule(lr, self.step) # Learning rate schedule

                # Forward prop, compute output through LM and finetune layer
                output = self.model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0]

                # Masking of labels that don't exist, and calculation of class weights
                msk = (batch.labels >= 0)
                mskd_lbs = batch.labels[msk]
                mskd_outs = output[msk]
                weights = mskd_lbs * len(mskd_lbs)/mskd_lbs.sum()
                weights[weights==0] = 1

                # Label smoothing for masked labels
                smooth_lbs = _binary_smooth(mskd_lbs, alpha=alpha) # Eg [0,0,0,1,0] ---> [0.05, 0.05, 0.05, 0.95, 0.05]

                # Compute weighted loss and backprop
                loss = nn.BCELoss(weight=weights, reduction='mean')(mskd_outs, smooth_lbs)
                loss.backward()

                # Gradient accumulation every N batches (default value: every 6 batches)
                if (idx+1)%grad_accum_steps==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Increment step
                self.step += 1

                # Print step every 100 iterations
                if (idx+1)%100==0:
                    self.logger.info(f'Steps: {self.step}')

                # Validate model every 1000 batches
                if (idx+1)%val_freq==0:
                    self.validate(valid_loader=valid_loader, alpha=alpha)

                # Save every 10_000 steps
                if (idx+1)%save_chkpt_freq==0:
                    if save_chkpt_dir is not None:
                        if use_S3:
                            self.save_chkpt_to_AWS(save_chkpt_dir)
                        else:
                            self.save_chkpt_to_local(save_chkpt_dir)
                    else:
                        self.logger.info('Cannot save because no saving path specified')


    def validate(self, valid_loader, val_iters=100, alpha=0.1):

        self.model.eval() # Model in eval mode to avoid dropout and use appropriate batch norm

        with torch.no_grad(): # No recording of gradients to save memory for validation

            self.logger.info('Validating...')

            cf = torch.zeros(2,2).type(torch.int)
            valid_loss = 0

            for i, batch in enumerate(valid_loader): # Iterate through validation dataset

                # Forward prop through BERT + finetune and compute loss
                output = self.model(batch.src, batch.segs, batch.clss, batch.mask_attn, batch.mask_clss)[0]  # select sent scores
                msk = (batch.labels >= 0)
                mskd_lbs = batch.labels[msk]
                mskd_outs = output[msk]

                # Tiny func to just view annotated output
                self.export_sents_with_scores(output, batch)

                # Label smoothing for masked labels
                smooth_lbs = _binary_smooth(mskd_lbs, alpha=alpha) # Eg [0,0,0,1,0] ---> [0.05, 0.05, 0.05, 0.95, 0.05]

                # Compute validation loss
                valid_loss += nn.BCELoss(reduction='mean')(mskd_outs, smooth_lbs)

                # Binarization of outputs
                binary_outputs = (mskd_outs > .5).type(torch.int)

                # Confusion matrix update
                cf += _cf(binary_outputs, mskd_lbs)
                self.logger.debug(f'val outputs shape: {binary_outputs.shape}')

                # After n_val_iters, update self.cf and self.mcc, then break
                if (i+1)%val_iters == 0:
                    self.cf = cf
                    self.logger.info(f'confusion_matrix: \n{self.cf}')
                    tn, fp, fn, tp = self.cf.type(torch.float).view(-1)
                    self.mcc = _mcc(tn, fp, fn, tp)
                    self.logger.info(f'MCC:{self.mcc}')
                    self.valid_loss = valid_loss
                    self.logger.info(f'Validation loss:{self.valid_loss}')
                    break

    def export_sents_with_scores(self, outputs, batch):
        # Create array of tuples of sentences annoted with label and score
        num_sents = batch.labels.numpy().shape[-1]
        sents = np.array(batch.src_txt)[:, :num_sents]

        tuples = np.dstack([sents, batch.labels.numpy(), outputs.numpy()])

        # Export each sentence
        for document in tuples:
            for sentence, label, score in document:
                if float(label) < 0:
                    continue
                else:
                    print('Sentence:')
                    print(sentence)
                    print()

                    print('Label: ', label)
                    print('Score: ', score)
                    print()
                    print()
