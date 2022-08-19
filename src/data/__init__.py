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
        

        # single_data = edsr_dataset.__getitem__(0)
        # hr = single_data[0][0]
        # hrnp = np.array(hr)
        # imageio.imwrite('./hr.png', hrnp)
        
        
        self.loader_train = dataloader.DataLoader(
            MyConcatDataset(edsr_dataset),
            batch_size = args.batch_size,
            shuffle = True,
            generator = torch.Generator(device = 'cuda'),
            num_workers = args.num_workers,
        )

        # self.loader_train.dataset.__getitem__(0)
        
        # hrnp = np.array(hr)
        # imageio.imwrite('./hr.png', hrnp)
