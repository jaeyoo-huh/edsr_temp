from importlib import import_module
from xml.sax import default_parser_list
import torch
import torch.utils.data as data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import numpy as np
import imageio

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Dataset:
    def __init__(self, args):        
        edsr_dataset = []
        module = import_module('data.srdata')
        edsr_dataset.append(getattr(module, 'SRData')(args, name='DIV2K'))
    
        self.loader_train = dataloader.DataLoader(
            MyConcatDataset(edsr_dataset),
            # edsr_dataset,
            batch_size = args.batch_size,
            # batch_size= 10,
            shuffle = True,
            generator = torch.Generator(device = 'cuda'),
            num_workers = args.num_workers,
        )

        test_dataset = []
        test_module = import_module('data.srdata')
        test_dataset.append(getattr(test_module, 'SRData')(args, train=False, name='DIV2K'))

        self.loader_test = dataloader.DataLoader(
            MyConcatDataset(test_dataset),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers = args.num_workers,
        )
        # self.loader_train.dataset.__getitem__(0)