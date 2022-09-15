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
import numpy as np
import imageio

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
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=100, gamma=0.1)
        self.scale = args.scale
        self.psnr = []
        self.learning = []
    
    def train(self, epoch):
        
        self.ckp.write_log('[Epoch {}]\t'.format(epoch))

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        criterion = nn.L1Loss()
        
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda= lambda epoch: 0.95 ** epoch)
        # scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.5)

        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()

        # self.dataloader.dataset.set_scale(0)
        for batch, (lr, hr) in enumerate(self.loader_train):     
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

            if (epoch % 50 == 0) and (batch == 5):
                    lrcpu = lr.cpu()
                    lrnp = lrcpu.detach().numpy()
                    img1 = lrnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/while_training/{}_lr.png'.format(epoch), img1)

                    srcpu = sr.cpu()
                    srnp = srcpu.detach().numpy()
                    img2 = srnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/while_training/{}_sr.png'.format(epoch), img2)

                    hrcpu = hr.cpu()
                    hrnp = hrcpu.detach().numpy()
                    img3 = hrnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/while_training/{}_hr.png'.format(epoch), img3)

        self.scheduler.step()

        # torch.save(self.model.state_dict(),"./experiment/test2/model_{}.pt".format(epoch))
        # torch.save(sr, "./experiment/test2/sr_{}.pt".format(epoch))

        # if epoch % 30 == 0:
        #     # sr = torch.load("./experiment/test/sr_150.pt")    
        #     sr = sr.cpu()
        #     srnp = sr.detach().numpy()
        #     img1 = srnp[0].transpose(1,2,0)
        #     imageio.imwrite('./experiment/0906/{}_sr.png'.format(epoch), img1)

        #     hr = hr.cpu()
        #     hrnp = hr.detach().numpy()
        #     img2 = hrnp[0].transpose(1,2,0)
        #     imageio.imwrite('./experiment/0906/{}_target.png'.format(epoch), img2)
            # print(sr.shape)

## timer, loss작성, capturable?, ckp, lr_scheduler, .item()
## dataset있는경우

    def eval(self, epoch):
        with torch.no_grad():
            
            self.ckp.write_log('\nEvaluation:')
            self.ckp.write_log('\nlr: {}'.format(self.optimizer.param_groups[0]['lr']))
            self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
            )
            # import pdb; pdb.set_trace()
            self.model.eval()

            timer_test = utility.timer()
            
            for batch , d in enumerate(tqdm(self.loader_test, ncols=80)):
                lr = d[0]
                hr = d[1]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                hr = hr.to(device)
                lr = lr.to(device)
                sr = self.model(lr, 0)
                

                self.ckp.log[-1, 0, 0] += utility.psnr(sr, hr)

                # if batch == (len(self.loader_test) - 1):
                if (batch == 5) and (epoch % 30 == 0):
                    lrcpu = lr.cpu()
                    lrnp = lrcpu.detach().numpy()
                    img1 = lrnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/{}_lr.png'.format(epoch), img1)

                    srcpu = sr.cpu()
                    srnp = srcpu.detach().numpy()
                    img2 = srnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/{}_sr.png'.format(epoch), img2)

                    hrcpu = hr.cpu()
                    hrnp = hrcpu.detach().numpy()
                    img3 = hrnp[0].transpose(1,2,0)
                    imageio.imwrite('./experiment/0915_edsr/{}_hr.png'.format(epoch), img3)

                    torch.save(hr, "./experiment/0915_edsr{}_hr.pt".format(epoch))
                    torch.save(lr, "./experiment/0915_edsr/{}_lr.pt".format(epoch))
                    torch.save(sr, "./experiment/0915_edsr/{}_sr.pt".format(epoch))


            self.ckp.log[-1, 0, 0] /= 10
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[DIV2K x 2]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.ckp.log[-1, 0, 0],
                    best[0][0, 0],
                    best[1][0, 0] + 1
                )
            )
            
            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

            # torch.save(self.ckp.log, "./experiment/test4/log_{}.pt".format(epoch))

            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )

            self.psnr.append(self.ckp.log[-1, 0, 0])
            self.learning.append(self.optimizer.param_groups[0]['lr'])

            if epoch == 145 or epoch == 299:             
                f = open("./experiment/0915_edsr/psnr_{}.txt".format(epoch), "w")                                  
                for i in range(epoch):
                    f.write("PSNR_{} : {:.3f}\n".format(i, self.psnr[i]))
                f.close()

                g = open("./experiment/0915_edsr/learning_{}.txt".format(epoch), "w")                                  
                for l in range(epoch):
                    g.write("PSNR_{} : {:.5f}\n".format(l, self.learning[l]))
                g.close()
            
## 체크포인트 없애기, 학습 중간 hr,target 이미지 저장 확인 만들기(이미지 띄우는 코드), 
## hyperparameter 변경하며 학습, image augmentation + 이미지 확인