import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import cv2
import torch
import torch.utils.data as data

class DIV2K(data.Dataset):
    def __init__(self, args, name, train=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale

        self.apath = os.path.join(args.dir_data, self.name)
        bin_path = os.path.join(self.apath, 'bin')

        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')

        os.makedirs(
            self.dir_hr.replace(self.apath, bin_path), 
            exist_ok=False
            )
        os.makedirs(
            os.path.join(
                self.dir_lr.replace(self.apath, bin_path), 'X{}'.format(args.scale)
            ),
            exist_ok=False
            )

        list_hr, list_lr = self.scan()
        self.images_hr, self.images_lr = [], []
        for h in list_hr:
            hr_image_pt = h.replace(self.apath, bin_path)
            hr_image_pt = hr_image_pt.replace('png', '.pt')
            self.images_hr.append(hr_image_pt)
            self._check_and_load(h, hr_image_pt, verbose=True) 
        for l in list_lr:
            lr_image_pt = l.replace(self.apath, bin_path)
            lr_image_pt = lr_image_pt.replace('png', '.pt')
            self.images_lr.append(lr_image_pt)
            self._check_and_load(l, lr_image_pt, verbose=True)

        #for where?
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    ##
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))

        return names_hr, names_lr
    ##
    
    def _check_and_load(self, img, pt, verbose=True):
        if verbose:
            print('Making a binary: {}'.format(pt))
        with open(pt, 'wb') as _pt:
            #pickle.dump(cv2.imread(img), _pt)
            pickle.dump(imageio.imread(img), _pt)
   

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename


    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)



