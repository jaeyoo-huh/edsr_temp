import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, my_dataloader, my_model, ckp):
        self.args = args
        self.dataloader= my_dataloader.loader_train
        self.model = my_model
        self.lr = args.lr
        self.ckp = ckp
        self.cuda = args.cuda        
    
    def train(self, epoch):
        
        self.ckp.write_log('[Epoch {}]\t'.format(epoch))

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr) 
        optimizer.param_groups[0]['capturable'] = True

        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()

        self.dataloader.dataset.set_scale(0)
        for batch, (hr, lr) in enumerate(self.dataloader):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            hr = hr.to(device)
            lr = lr.to(device)

            timer_data.hold()
            timer_model.tic()

            optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.dataloader.dataset),
                    loss,
                    # self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

## timer, loss작성, capturable?, ckp, lr_scheduler, 
## dataset있는경우