import time
import numpy as np
import functools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from deform_conv import ModulatedDeformConvPack as DCN

from .resnet import resnet50


###############################################################################
#----------------------------------- Init -------------------------------------
###############################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



###############################################################################
#-------------------------------- Define Net ----------------------------------
###############################################################################
def define_RN50_net(shift, step, upsample='ps', add_refine_stage=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = RN50(4, shift=shift, step=step, upsample=upsample, add_refine_stage=add_refine_stage)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
#---------------------------------- ResNet ------------------------------------
###############################################################################

class _ConvLayer(nn.Sequential):
    def __init__(self, inc, ouc, kernel_size=1, stride=1, padding=0, bn=True, bias=False):
        super(_ConvLayer, self).__init__()
        if bn:
            self.add_module('conv', nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)),
            self.add_module('norm', nn.BatchNorm2d(ouc))
            self.add_module('relu', nn.ReLU(inplace=True))
        else:
            self.add_module('conv', nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)),
            self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x):
        out = super(_ConvLayer, self).forward(x)
        return out


class RefineLayer(nn.Sequential):
    def __init__(self, inc, ouc, kernel_size=3, stride=1, padding=1):
        super(RefineLayer, self).__init__()
        self.conv1 = _ConvLayer(inc, ouc, kernel_size, stride, padding)
        self.conv2 = _ConvLayer(ouc, ouc, kernel_size, stride, padding)
        self.conv3 = _ConvLayer(ouc, ouc, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class GCN(nn.Module):
    def __init__(self, inc, ouc, k=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inc, ouc, kernel_size=(k,1), padding=((k-1)//2,0))
        self.conv_l2 = nn.Conv2d(ouc, ouc, kernel_size=(1,k), padding=(0,(k-1)//2))
        self.conv_r1 = nn.Conv2d(inc, ouc, kernel_size=(1,k), padding=((k-1)//2,0))
        self.conv_r2 = nn.Conv2d(ouc, ouc, kernel_size=(k,1), padding=(0,(k-1)//2))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


###############################################################################
#---------------------------------- RN50 ----------------------------------
###############################################################################

class UpScaleLayer(nn.Module):
    def __init__(self, inc, ouc):
        super(UpScaleLayer, self).__init__()
        self.up = nn.Conv2d(inc, ouc, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.up(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class BasicFusionLayer(nn.Module):
    def __init__(self, nf):
        super(BasicFusionLayer, self).__init__()
        self.conv1 = _ConvLayer(nf*2, nf, kernel_size=3, padding=1, bn=True, bias=False)
        self.conv2 = _ConvLayer(nf  , nf, kernel_size=3, padding=1, bn=True, bias=False)

    def forward(self, nbr, ref):
        x = self.conv1(torch.cat([nbr, ref], dim=1))
        x = self.conv2(x)
        return x


class BasicFlowLayer(nn.Module):
    def __init__(self, nf, groups=8):
        super(BasicFlowLayer, self).__init__()
        self.off_conv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.off_conv2 = nn.Conv2d(nf  , nf, 3, 1, 1, bias=True)
        self.dcnpack = DCN(nf, nf, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr, ref):
        off = self.lrelu(self.off_conv1(torch.cat([nbr, ref], dim=1)))
        off = self.lrelu(self.off_conv2(off))
        fea = self.dcnpack([nbr, off])
        return fea


class CAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SAttentionLayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAttentionLayer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y


class CSAttention(nn.Module):
    def __init__(self, nf, reduction):
        super(CSAttention, self).__init__()
        self.ca = CAttentionLayer(nf, reduction)
        self.sa = SAttentionLayer(kernel_size=7)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, chl, shift, step, groups=8):
        super(FusionNet, self).__init__()
        self.shift = shift
        self.step = step
        self.iters = shift + 1
        nf0, nf1, nf2, nf3, nf4 = chl

        self.flow0 = BasicFlowLayer(nf0, groups)
        self.flow1 = BasicFlowLayer(nf1, groups)
        self.flow2 = BasicFlowLayer(nf2, groups)
        self.flow3 = BasicFlowLayer(nf3, groups)

        self.attn0 = CSAttention(nf0*self.iters, self.iters)
        self.attn1 = CSAttention(nf1*self.iters, self.iters)
        self.attn2 = CSAttention(nf2*self.iters, self.iters)
        self.attn3 = CSAttention(nf3*self.iters, self.iters)
        self.attn4 = CSAttention(nf4*self.iters, self.iters)

        self.fuse0 = _ConvLayer(self.iters*nf0, nf0, 1, bn=True, bias=False)
        self.fuse1 = _ConvLayer(self.iters*nf1, nf1, 1, bn=True, bias=False)
        self.fuse2 = _ConvLayer(self.iters*nf2, nf2, 1, bn=True, bias=False)
        self.fuse3 = _ConvLayer(self.iters*nf3, nf3, 1, bn=True, bias=False)
        self.fuse4 = _ConvLayer(self.iters*nf4, nf4, 1, bn=True, bias=False)
            
    def forward(self, fea_lst):
        # f0, f1, f2, f3, f4
        l0, l1, l2, l3, l4 = [], [], [], [], []
        for i in range(self.iters):
            l0.append(self.flow0(fea_lst[0][i:i+self.step], fea_lst[0][-self.step:]))
            l1.append(self.flow1(fea_lst[1][i:i+self.step], fea_lst[1][-self.step:]))
            l2.append(self.flow2(fea_lst[2][i:i+self.step], fea_lst[2][-self.step:]))
            l3.append(self.flow3(fea_lst[3][i:i+self.step], fea_lst[3][-self.step:]))
            l4.append(fea_lst[4][i:i+self.step])

        l0 = torch.cat(l0, dim=1)
        l1 = torch.cat(l1, dim=1)
        l2 = torch.cat(l2, dim=1)
        l3 = torch.cat(l3, dim=1)
        l4 = torch.cat(l4, dim=1)
        
        l0 = self.attn0(l0)
        l1 = self.attn1(l1)
        l2 = self.attn2(l2)
        l3 = self.attn3(l3)
        l4 = self.attn4(l4)

        l0 = self.fuse0(l0)
        l1 = self.fuse1(l1)
        l2 = self.fuse2(l2)
        l3 = self.fuse3(l3)
        l4 = self.fuse4(l4)

        return (l0, l1, l2, l3, l4)


class PsDeconvLayer(nn.Module):
    def __init__(self, inc, ouc, sc=0, kernel_size=3, stride=1, padding=1, upsample='ps'):
        super(PsDeconvLayer, self).__init__()
        self.rs = _ConvLayer(inc, ouc*4, kernel_size=1, bn=True, bias=False)
        self.up = nn.PixelShuffle(2)
        # for rebuttal
        self.upsample = upsample
        if self.upsample == 'deconv':
            self.up_deconv = nn.ConvTranspose2d(inc, ouc, kernel_size=4, stride=2, padding=1, bias=True)
        self.deconv = _ConvLayer(ouc+sc, ouc, kernel_size=3, padding=1, bn=True, bias=False)

    def forward(self, fea, skip=None):
        if self.upsample == 'deconv':
            fea = self.up_deconv(fea)
        else:
            fea = self.up(self.rs(fea))
        if skip is not None: 
            out = self.deconv(torch.cat([fea, skip], dim=1))        
        else:
            out = self.deconv(fea)        
        return out


class Decoder(nn.Module):
    def __init__(self, chl, upsample='ps'):
        super(Decoder, self).__init__()
        nf0, nf1, nf2, nf3, nf4 = chl
        self.skip0 = GCN(nf0, nf0)
        self.skip1 = GCN(nf1, nf1)
        self.skip2 = GCN(nf2, nf2)
        self.skip3 = GCN(nf3, nf3)
        self.skip4 = GCN(nf4, nf4)
        self.deconv0 = PsDeconvLayer(nf0, nf0,   4, upsample)
        self.deconv1 = PsDeconvLayer(nf1, nf0, nf0, upsample)
        self.deconv2 = PsDeconvLayer(nf2, nf1, nf1, upsample)
        self.deconv3 = PsDeconvLayer(nf3, nf2, nf2, upsample)
        self.deconv4 = PsDeconvLayer(nf4, nf3, nf3, upsample)
        
    def forward(self, fea_lst):
        x = fea_lst[0]
        l0 = self.skip0(fea_lst[1])
        l1 = self.skip1(fea_lst[2])
        l2 = self.skip2(fea_lst[3])
        l3 = self.skip3(fea_lst[4])
        l4 = self.skip4(fea_lst[5])
        d4 = self.deconv4(l4, l3)
        d3 = self.deconv3(d4, l2)
        d2 = self.deconv2(d3, l1)
        d1 = self.deconv1(d2, l0)
        d0 = self.deconv0(d1,  x)
        return d0


class RN50(nn.Module):
    def __init__(self, inc, shift=None, step=None, upsample='ps', add_refine_stage=False):
        super(RN50, self).__init__()
        self.add_refine_stage = add_refine_stage
        self.shift = shift 
        self.step = step
        chl = [64, 256, 512, 1024, 2048]

        self.encoder = resnet50(inc, add_layer4=True, add_layer0=True)
        self.fusion = FusionNet(chl, self.shift, self.step)
        self.decoder = Decoder(chl, upsample=upsample)
        self.pred_head = nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        if self.add_refine_stage:
            self.refine = RefineLayer(inc, 64) 
            self.refine_head = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

    def forward(self, x):
        encoded_fea = self.encoder(x) # f0, f1, f2, f3, f4
        encoded_fea = self.fusion(encoded_fea) # f0, f1, f2, f3, f4
        decoded_fea = self.decoder([x[-self.step:]] + list(encoded_fea))
        raw = self.pred_head(decoded_fea)
        pred = torch.sigmoid(raw)
        pred_rf = 0.
        if self.add_refine_stage:
            y = torch.cat([x[-self.step:,:3,:,:], pred], dim=1)
            refine_fea = self.refine(y)
            raw_rf = self.refine_head(refine_fea)
            pred_rf = torch.sigmoid(raw+raw_rf)
        return pred, pred_rf
