import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import model.edsr

def main():
    global model
    _data = data.Dataset(args)
    _model = model.Model(args, checkpoint)



if __name__ == '__main__':
    net = model.edsr.EDSR(args)
    randomt = torch.randn([4,3,16,16])
    print(net(randomt).shape)
