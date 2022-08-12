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
    def __init__(self, args, my_dataloader, my_model):
        self.args = args
        self.my_dataloader= my_dataloader.loader_train
        self.model = my_model
        self.lr = args.lr
    
    def train(self):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(params=s, lr=self.lr) 