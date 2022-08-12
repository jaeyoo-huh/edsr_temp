from importlib import import_module
from xml.sax import default_parser_list
import torch.utils.data as data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

class Dataset:
    def __init__(self, args):        
        edsr_dataset = []
        module = import_module('data.srdata')
        edsr_dataset.append(module.DIV2K(args, name='DIV2K'))
        
        self.loader_train = dataloader.DataLoader(
            dataset = edsr_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
        )