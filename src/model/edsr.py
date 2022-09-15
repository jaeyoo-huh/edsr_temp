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

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feats // 2, out_channels=n_feats // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()

        conv1 = [conv(n_feats, n_feats, kernel_size, bias=True)]
        self.conv1 = nn.Sequential(*conv1)
        conv2 = [conv(n_feats // 2, n_feats, kernel_size, bias=True)]
        self.conv2 = nn.Sequential(*conv2)

        m=[]
        m.append(conv(n_feats, n_feats, kernel_size, bias=True))
        # m.append(self.sg)
        # m.append(self.sca)
        m.append(nn.ReLU())
        m.append(conv(n_feats, n_feats, kernel_size, bias=True))
        # m.append(conv(n_feats // 2, n_feats, kernel_size, bias=True))
        
        self.residual = nn.Sequential(*m)

    def forward(self, x):
        # y = x
        # res = self.conv1(y)
        # res = self.sg(res)
        # res = res * self.sca(res)
        # res = self.conv2(res)
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
        self.n_resblocks = args.n_resblocks
        self.n_colors = args.n_colors

        conv1 = [conv(self.n_colors, n_feats, kernel_size)]
        resblocks = ResBlock(conv, n_feats, kernel_size)        
        resblock_list = []
        for _ in range(self.n_resblocks):
            resblock_list.append(ResBlock(conv, n_feats, kernel_size))

        conv2 = [conv(n_feats, n_feats, kernel_size=3)]
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        upsample = [Upsampler(conv, n_feats, bias=True), conv(n_feats, self.n_colors, kernel_size)]

        self.conv1 = nn.Sequential(*conv1)
        self.resblocks = nn.ModuleList(resblock_list)
        # self.resblocks = nn.Sequential(*resblocks)
        self.conv2 = nn.Sequential(*conv2)
        self.upsampler = nn.Sequential(*upsample)

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feats // 2, out_channels=n_feats // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm = LayerNorm2d(channels=3)

    def forward(self, x):
        # x = self.norm(x)
        x = self.sub_mean(x)
        out = self.conv1(x)

        for i in range(self.n_resblocks):
            out = self.resblocks[i](out)

        # res = self.resblocks(x)
        # res += x

        out = self.conv2(out)

        out = self.upsampler(out)

        out = self.add_mean(out)

        return out    


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False





class NAFnet(nn.Module):
    def __init__(self, args, c=3, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., conv = default_conv):
        super().__init__()
        self.n_colors = args.n_colors
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        upsample = [Upsampler(conv, c, bias=True), conv(c, self.n_colors, kernel_size=3)]
        self.upsampler = nn.Sequential(*upsample)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        out = y + x * self.gamma

        out = self.upsampler(out)

        return out