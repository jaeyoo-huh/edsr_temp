import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import model.edsr
import numpy as np
import imageio

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
        t.eval(epochs)
        

    checkpoint.done()


if __name__ == '__main__':
    
    # sr = torch.load("./experiment/test/sr_150.pt")    
    # sr = sr.cpu()
    # srnp = sr.detach().numpy()
    # img1 = srnp[5].transpose(1,2,0)
    # imageio.imwrite('./temp/sr2.png', img1)
    # print(sr.shape)

    # ckp = torch.load("./experiment/test/log_150.pt")
    # ckp = ckp.cpu()
    # ckpnp = ckp.detach().numpy()
    # print(ckpnp)

    # a = [1,2,3,4,5]
    # f = open("./experiment/test4/psnr.txt", "w")
    # for i in range(5):
    #     f.write("PSNR_{} : {}\n".format(i+1, a[i]))
    # f.close()

    main()

    # net = model_temp.edsr.EDSR(args)
    # randomt = torch.randn([4,3,16,16])
    # print(net(randomt).shape)
