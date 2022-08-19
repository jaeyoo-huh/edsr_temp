import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent = False):
    return EDSR(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias
        )

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size):    

        super(ResBlock, self).__init__()

        m=[]
        m.append(conv(n_feats, n_feats, kernel_size, bias=True))
        m.append(nn.ReLU())
        m.append(conv(n_feats, n_feats, kernel_size, bias=True))
        
        self.residual = nn.Sequential(*m)

    def forward(self, x):
        res = self.residual(x)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, n_feats, bias=True):

        m=[]
        m.append(conv(n_feats, 4*n_feats, 3, bias))
        m.append(nn.PixelShuffle(2))

        super(Upsampler, self).__init__(*m)
    
    """
    def forward(self, x):
        upsample = self.upsampler(x)

        return upsample
        """


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        self.res_scale = args.res_scale
        self.n_colors = args.n_colors

        conv1 = [conv(self.n_colors, n_feats, kernel_size)]
        resblocks = [ResBlock(conv, n_feats, kernel_size) for i in range(self.res_scale)]
        conv2 = [conv(n_feats, n_feats, kernel_size=3)]

        upsample = [Upsampler(conv, n_feats, bias=True), conv(n_feats, self.n_colors, kernel_size)]

        self.conv1 = nn.Sequential(*conv1)
        self.resblocks = nn.ModuleList(resblocks)
        self.conv2 = nn.Sequential(*conv2)
        self.upsampler = nn.Sequential(*upsample)


    def forward(self, x):
        out = self.conv1(x)

        for i in range(self.res_scale):
            out = self.resblocks[i](out)

        out = self.conv2(out)

        out = self.upsampler(out)

        return out    