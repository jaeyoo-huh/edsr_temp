import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import model.edsr


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model, data
    my_dataset = data.Dataset(args)
    my_model = model.Model(args, checkpoint)
    epochs = args.epochs
    t = Trainer(args, my_dataset, my_model, checkpoint) 
    # while not t.terminate():
    for epochs in range(args.epochs):
        t.train(epochs)
        

    checkpoint.done()





if __name__ == '__main__':
    main()
    # net = model_temp.edsr.EDSR(args)
    # randomt = torch.randn([4,3,16,16])
    # print(net(randomt).shape)
