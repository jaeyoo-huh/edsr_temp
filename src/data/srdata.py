import os
import glob
import random
import pickle
from re import S

from data import common

import numpy as np
import imageio
import cv2
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name, train=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale

        if train:
            self.dir_hr = os.path.join(args.dir_data, 'DIV2K_train_HR')
            self.dir_lr = os.path.join(args.dir_data, 'DIV2K_train_LR_bicubic')
            self.dir_lr_X2 = os.path.join(self.dir_lr, 'X2')
        elif train == False:
            self.dir_hr = os.path.join(args.dir_data, 'DIV2K_test_HR')
            self.dir_lr = os.path.join(args.dir_data, 'DIV2K_test_LR')
            self.dir_lr_X2 = os.path.join(self.dir_lr, 'X2')


        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = [], []

        # i = 0
        for i, h in enumerate(list_hr):
            print('reading HR image: {}'.format(i))
            img_hr= imageio.imread(h)
            self.images_hr.append(img_hr)
            # i = i + 1

            # if i >3:
            #     break
                
        # ll = 0
        for i, l in enumerate(list_lr):
            print('reading LR image: {}'.format(i))
            img_lr = imageio.imread(l)
            self.images_lr.append(img_lr)
            # ll = ll + 1

            # if ll > 3:
            #     break

        # single_data_hr = self.images_hr[2]
        # single_data_lr = self.images_lr[2]
        # hrnp = np.array(single_data_hr)
        # lrnp = np.array(single_data_lr)
        # imageio.imwrite('./temp/hr0.png', hrnp)
        # imageio.imwrite('./temp/lr0.png', lrnp)

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + '.png'))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr_X2, '*' + '.png'))
        )

        return names_hr, names_lr

    def make_pair(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        pair = self.get_patch(f_hr, f_lr)

        return pair

    def check_item(self, pair, idx):
        hrimage = pair[0]        
        lrimage = pair[1]
        
        # nphrimage = np.array(hrimage)
        # nplrimage = np.array(lrimage)

        imageio.imwrite('./temp/{}_hr.png'.format(idx), hrimage)
        imageio.imwrite('./temp/{}_lr.png'.format(idx), lrimage)
        
    def __getitem__(self, idx):
        pair = self.make_pair(idx)
        # if idx % 4000 == 0:
        #     self.check_item(pair, idx)    
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1]

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def get_patch(self, hr, lr):
        if self.train:
            lr, hr = common.get_patch(
                    lr, hr,
                    patch_size=self.args.patch_size,
                    scale=2,
                    multi=(len(self.scale) > 1)
                )
        # if not self.args.no_augment: 
        lr, hr = common.augment(lr, hr)
        # else:
        #     ih, iw = lr.shape[:2]
        #     hr = hr[0:ih * 2, 0:iw * 2]
        return lr, hr



