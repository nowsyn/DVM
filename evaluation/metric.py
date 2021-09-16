import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import time
import skimage.measure

from PIL import Image 
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from multiprocessing import Pool


def findMaxConnectedRegion(x):
    assert len(x.shape) == 2
    cc, num = skimage.measure.label(x, connectivity=1, return_num=True)
    omega = np.zeros_like(x)
    if num > 0:
        # find the largest connected region
        max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
        omega[cc == max_id] = 1
    return omega

def genGaussKernel(sigma, q=2):
    pi = math.pi
    eps = 1e-2 

    def gauss(x, sigma):
        return np.exp(-np.power(x,2)/(2*np.power(sigma,2))) / (sigma*np.sqrt(2*pi))

    def dgauss(x, sigma):
        return -x * gauss(x,sigma) / np.power(sigma, 2)

    hsize = int(np.ceil(sigma*np.sqrt(-2*np.log(np.sqrt(2*pi)*sigma*eps))))
    size = 2 * hsize + 1
    hx = np.zeros([size, size], dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u, v = i-hsize, j-hsize
            hx[i,j] = gauss(u,sigma) * dgauss(v,sigma)

    hx = hx / np.sqrt(np.sum(np.power(np.abs(hx), 2)))
    hy = hx.transpose(1, 0)
    return hx, hy, size

def calcOpticalFlow(frames):
    prev, curr = frames
    flow = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), curr.astype(np.uint8), None,  
                                        0.5, 5, 10, 2, 7, 1.5, 
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow


class ImageFilter(nn.Module):
    def __init__(self, chn, kernel_size, weight, device):
        super(ImageFilter, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size == weight.size(-1)
        self.filter = nn.Conv2d(chn, chn, kernel_size, padding=0, bias=False) 
        self.filter.weight = nn.Parameter(weight)
        self.device = device

    def pad(self, x): 
        assert len(x.shape) == 3
        x = x.unsqueeze(-1).permute((0,3,1,2))
        b, c, h, w = x.shape
        pad = self.kernel_size // 2
        y = torch.zeros([b, c, h+pad*2, w+pad*2]).to(self.device) 
        y[:,:,0:pad,0:pad] = x[:,:,0:1,0:1].repeat(1,1,pad,pad)
        y[:,:,0:pad,w+pad:] = x[:,:,0:1,-1:].repeat(1,1,pad,pad)
        y[:,:,h+pad:,0:pad] = x[:,:,-1:,0:1].repeat(1,1,pad,pad)
        y[:,:,h+pad:,w+pad:] = x[:,:,-1:,-1:].repeat(1,1,pad,pad)

        y[:,:,0:pad,pad:w+pad] = x[:,:,0:1,:].repeat(1,1,pad,1)
        y[:,:,pad:h+pad,0:pad] = x[:,:,:,0:1].repeat(1,1,1,pad)
        y[:,:,h+pad:,pad:w+pad] = x[:,:,-1:,:].repeat(1,1,pad,1)
        y[:,:,pad:h+pad,w+pad:] = x[:,:,:,-1:].repeat(1,1,1,pad)

        y[:,:,pad:h+pad, pad:w+pad] = x
        return y

    def forward(self, x):
        y = self.filter(self.pad(x))
        return y
        

class BatchMetric(object):
    def __init__(self, device, grad_sigma=1.4, grad_q=2, 
                       conn_step=0.1, conn_thresh=0.5, conn_theta=0.15, conn_p=1):
        # parameters for connectivity
        self.conn_step = conn_step 
        self.conn_thresh = conn_thresh
        self.conn_theta = conn_theta
        self.conn_p = conn_p
        self.device = device

        hx, hy, size = genGaussKernel(grad_sigma, grad_q)
        self.hx = hx
        self.hy = hy
        self.kernel_size = size
        kx = self.hx[::-1, ::-1].copy()
        ky = self.hy[::-1, ::-1].copy()
        kernel_x = torch.from_numpy(kx).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.from_numpy(ky).unsqueeze(0).unsqueeze(0)
        self.fx = ImageFilter(1, self.kernel_size, kernel_x, self.device).cuda(self.device)
        self.fy = ImageFilter(1, self.kernel_size, kernel_y, self.device).cuda(self.device)

    def run(self, input, target, mask=None):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.zeros_like(target_t).to(self.device)
            mask[(target_t>0) * (target_t<255)] = 1
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()
        sad = self.BatchSAD(input_t, target_t, mask)
        mse = self.BatchMSE(input_t, target_t, mask)
        grad = self.BatchGradient(input_t, target_t, mask)
        conn = self.BatchConnectivity(input_t, target_t, mask)
        return sad, mse, grad, conn

    def run_video(self, input, target, mask=None):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.zeros_like(target_t).to(self.device)
            mask[(target_t>0) * (target_t<255)] = 1
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()
        errs, nums = [], []
        err, n = self.SSDA(input_t, target_t, mask)
        errs.append(err)
        nums.append(n)
        err, n = self.dtSSD(input_t, target_t, mask)
        errs.append(err)
        nums.append(n)
        err, n = self.MESSDdt(input_t, target_t, mask)
        errs.append(err)
        nums.append(n)
        return errs, nums

    def run_metric(self, metric, input, target, mask=None):
        torch.cuda.empty_cache()
        input_t = torch.from_numpy(input.astype(np.float32)).to(self.device)
        target_t = torch.from_numpy(target.astype(np.float32)).to(self.device)
        if mask is None:
            mask = torch.zeros_like(target_t).to(self.device)
            mask[(target_t>0) * (target_t<255)] = 1
        else:
            mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            mask = (mask == 128).float()

        if metric == 'sad':
            ret = self.BatchSAD(input_t, target_t, mask)
        elif metric == 'mse':
            ret = self.BatchMSE(input_t, target_t, mask)
        elif metric == 'grad':
            ret = self.BatchGradient(input_t, target_t, mask)
        elif metric == 'conn':
            ret = self.BatchConnectivity(input_t, target_t, mask)
        elif metric == 'ssda':
            ret = self.SSDA(input_t, target_t, mask)
        elif metric == 'dtssd':
            ret = self.dtSSD(input_t, target_t, mask)
        elif metric == 'messddt':
            ret = self.MESSDdt(input_t, target_t, mask)
        else:
            raise NotImplementedError
        return ret

    def BatchSAD(self, pred, target, mask):
        B = target.size(0)
        error_map = (pred - target).abs() / 255.
        batch_loss = (error_map * mask).view(B, -1).sum(dim=-1) 
        batch_loss = batch_loss / 1000. 
        return batch_loss.data.cpu().numpy()

    def BatchMSE(self, pred, target, mask):
        B = target.size(0)
        error_map = (pred-target) / 255.
        batch_loss = (error_map.pow(2) * mask).view(B, -1).sum(dim=-1)
        batch_loss = batch_loss / (mask.view(B, -1).sum(dim=-1) + 1.)
        return batch_loss.data.cpu().numpy()

    def BatchGradient(self, pred, target, mask):
        B = target.size(0)
        pred = pred / 255.
        target = target / 255.
        
        pred_x_t = self.fx(pred).squeeze(1)
        pred_y_t = self.fy(pred).squeeze(1)
        target_x_t = self.fx(target).squeeze(1)
        target_y_t = self.fy(target).squeeze(1)
        pred_amp = (pred_x_t.pow(2) + pred_y_t.pow(2)).sqrt()
        target_amp = (target_x_t.pow(2) + target_y_t.pow(2)).sqrt()
        error_map = (pred_amp - target_amp).pow(2)
        batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
        return batch_loss.data.cpu().numpy()

    def BatchConnectivity(self, pred, target, mask):
        step = self.conn_step
        theta = self.conn_theta

        pred = pred / 255.
        target = target / 255.
        B, dimy, dimx = pred.shape
        thresh_steps = torch.arange(0, 1+step, step).to(self.device)
        l_map = torch.ones_like(pred).to(self.device)*(-1)
        pool = Pool(B)
        for i in range(1, len(thresh_steps)):
            pred_alpha_thresh = pred>=thresh_steps[i]
            target_alpha_thresh = target>=thresh_steps[i]
            mask_i = pred_alpha_thresh * target_alpha_thresh
            omegas = []
            items = [mask_ij.data.cpu().numpy() for mask_ij in mask_i]
            for omega in pool.imap(findMaxConnectedRegion, items):
                omegas.append(omega)
            omegas = torch.from_numpy(np.array(omegas)).to(self.device)
            flag = (l_map==-1) * (omegas==0) 
            l_map[flag==1] = thresh_steps[i-1]
        l_map[l_map==-1] = 1
        pred_d = pred - l_map
        target_d = target - l_map
        pred_phi = 1 - pred_d*(pred_d>=theta).float()
        target_phi = 1 -  target_d*(target_d>=theta).float()
        batch_loss = ((pred_phi-target_phi).abs()*mask).view([B, -1]).sum(-1)
        pool.close()
        return batch_loss.data.cpu().numpy()

    def GaussianGradient(self, mat):
        gx = np.zeros_like(mat)
        gy = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            gx[i, ...] = ndimage.filters.convolve(mat[i], self.hx, mode='nearest')
            gy[i, ...] = ndimage.filters.convolve(mat[i], self.hy, mode='nearest')
        return gx, gy
        
    def SSDA(self, pred, target, mask=None):
        B, h, w = target.shape
        pred = pred / 255.
        target = target / 255.
        error = ((pred-target).pow(2) * mask).view(B, -1).sum(dim=1).sqrt()
        num =  mask.view(B, -1).sum(dim=1) + 1. 
        return error.data.cpu().numpy(), num.data.cpu().numpy()
    
    def dtSSD(self, pred, target, mask=None):
        B, h, w = target.shape
        pred = pred / 255.
        target = target / 255.
        pred_0 = pred[:-1, ...]
        pred_1 = pred[1:, ...]
        target_0 = target[:-1, ...]
        target_1 = target[1:, ...]
        mask_0 = mask[:-1, ...]
        error_map = ((pred_1-pred_0) - (target_1-target_0)).pow(2)
        error = (error_map * mask_0).view(mask_0.shape[0], -1).sum(dim=1).sqrt()
        num = mask_0.view(mask_0.shape[0], -1).sum(dim=1) + 1. 
        return error.data.cpu().numpy(), num.data.cpu().numpy()

    def MESSDdt(self, pred, target, mask=None):
        B, h, w = target.shape

        pool = Pool(B)
        flows = []
        items = [t for t in target.data.cpu().numpy()]
        for flow in pool.imap(calcOpticalFlow, zip(items[:-1], items[1:])):
            flows.append(flow)
        flow = torch.from_numpy(np.rint(np.array(flows)).astype(np.int64)).to(self.device)
        pool.close()

        pred = pred / 255.
        target = target / 255.
        pred_0 = pred[:-1, ...]
        pred_1 = pred[1:, ...]
        target_0 = target[:-1, ...]
        target_1 = target[1:, ...]
        mask_0 = mask[:-1, ...]
        mask_1 = mask[1:, ...]
        
        B, h, w = target_0.shape
        x = torch.arange(0, w).to(self.device)
        y = torch.arange(0, h).to(self.device)
        xx, yy = torch.meshgrid([y, x])
        coords = torch.stack([yy, xx], dim=2).unsqueeze(0).repeat((B, 1, 1, 1))
        coords_n = (coords + flow)
        coords_y = coords_n[..., 0].clamp(0, h-1)
        coords_x = coords_n[..., 1].clamp(0, w-1)
        indices = coords_y * w + coords_x
        pred_1 = torch.take(pred_1, indices)
        target_1 = torch.take(target_1, indices)
        mask_1 = torch.take(mask_1, indices)

        error_map = (pred_0-target_0).pow(2) * mask_0 - (pred_1-target_1).pow(2) * mask_1
        error = error_map.abs().view(mask_0.shape[0], -1).sum(dim=1)
        num = mask_0.view(mask_0.shape[0], -1).sum(dim=1) + 1.
        return error.data.cpu().numpy(), num.data.cpu().numpy()
