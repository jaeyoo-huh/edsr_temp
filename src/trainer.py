import enum
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
        self.loader_train = my_dataloader.loader_train
        self.loader_test = my_dataloader.loader_test
        self.model = my_model
        self.lr = args.lr
        self.ckp = ckp
        self.cuda = args.cuda        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer.param_groups[0]['capturable'] = True
        self.scale = args.scale
        self.psnr = []
    
    def train(self, epoch):
        
        self.ckp.write_log('[Epoch {}]\t'.format(epoch))

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        criterion = nn.L1Loss()
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr) 
        self.optimizer.param_groups[0]['capturable'] = True

        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()

        # self.dataloader.dataset.set_scale(0)
        for batch, (hr, lr) in enumerate(self.loader_train):     
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)       
            hr = hr.to(device)
            lr = lr.to(device)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = criterion(sr, hr)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss.item(),
                    # self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        temp_loss = loss
        temp_sr = sr
        torch.save(
            self.model.state_dict(),
            # temp_loss,
            # temp_sr},
             "./experiment/test2/model_{}.pt".format(epoch))
        torch.save(sr, "./experiment/test2/sr_{}.pt".format(epoch))

## timer, loss작성, capturable?, ckp, lr_scheduler, .item()
## dataset있는경우

    def eval(self, epoch):
        with torch.no_grad():
            
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
            )

            self.model.eval()

            timer_test = utility.timer()
            if self.args.save_results: self.ckp.begin_background()
            for idx_scale, scale in enumerate(self.scale):
                for batch, d in enumerate(tqdm(self.loader_test, ncols=80)):
                    hr = d[0]
                    lr = d[1]
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(device)
                    hr = hr.to(device)
                    lr = lr.to(device)
                    sr = self.model(lr, idx_scale)
                    

                    self.ckp.log[-1, 0, idx_scale] += utility.psnr(sr, hr)
                    
                
                self.ckp.log[-1, 0, idx_scale] /= 10
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[DIV2K x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(                        
                        scale,
                        self.ckp.log[-1, 0, idx_scale],
                        best[0][0, idx_scale],
                        best[1][0, idx_scale] + 1
                    )
                )
            
            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

            if self.args.save_results:
                self.ckp.end_background()

            # if not self.args.test_only:
            #     self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
            torch.save(self.ckp.log, "./experiment/test2/log_{}.pt".format(epoch))

            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )

            self.psnr.append(self.ckp.log[-1, 0, idx_scale])

            if epoch == 299:
                torch.save(self.psnr, "./experiment/test2/PSNR.pt")
