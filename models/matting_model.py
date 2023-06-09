import os
import sys
import time
import functools
import itertools
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.sobel import Sobel
from kornia.filters.laplacian import Laplacian

from .base_model import BaseModel
from .networks import define_RN50_net
from .util import get_scheduler


class MattingModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--shift', type=int, default=1, help='how many neighboring frames to use')
        parser.add_argument('--upsample', type=str, default='ps', help='upsampling method')
        parser.add_argument('--add_refine_stage', action='store_true', help='whether to add refine stage')
        # loss hyperparameters
        parser.add_argument('--refine_loss', action='store_true', help='whether to add refine loss')
        parser.add_argument('--refine_weight', type=float, default=1.0, help='weight for refine stage')
        parser.add_argument('--soft_weight', type=float, default=1.0, help='weight for soft indices')
        parser.add_argument('--hard_weight', type=float, default=0.5, help='weight for hard indices')
        parser.add_argument('--comp_loss', action='store_true', help='whether to add composite loss')
        parser.add_argument('--comp_weight', type=float, default=1.0, help='weight for composite loss')
        parser.add_argument('--grad_loss', action='store_true', help='whether to add gradient loss')
        parser.add_argument('--grad_weight', type=float, default=1.0, help='weight for gradient loss')
        parser.add_argument('--grad_filter', action='store_true', help='gradient filter')
        parser.add_argument('--kld_loss', action='store_true', help='whether to add kl-divergence loss')
        parser.add_argument('--kld_weight', type=float, default=1.0, help='weight for kl-divergence loss')
        parser.add_argument('--temp_loss', action='store_true', help='whether to add temp loss')
        parser.add_argument('--temp_weight', type=float, default=1.0, help='weight for temp loss')
        # load
        parser.add_argument('--load_pretrain', action='store_true', help='if specified, load pre-trained models')
        parser.add_argument('--load_dir', type=str, default='checkpoints/pretrain', help='directory of pre-trained models')
        parser.add_argument('--load_epoch', type=str, default='latest', help='epoch of pre-trained models')
        parser.add_argument('--vis_more', action='store_true', help='whether to visualize more results')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # set loss names
        self.loss_names = ['total', 'diff']
        if self.opt.comp_loss: self.loss_names += ['comp']
        if self.opt.grad_loss: self.loss_names += ['grad']
        if self.opt.kld_loss: self.loss_names += ['kld']
        if self.opt.temp_loss: self.loss_names += ['temp']
        if self.opt.add_refine_stage and self.opt.refine_loss: self.loss_names += ['refine']

        # set models
        self.model_names = ['Base']
        self.shift = opt.shift
        self.netBase = define_RN50_net(shift=self.shift, step=opt.n_frames, \
                                        upsample=opt.upsample, add_refine_stage=opt.add_refine_stage, \
                                        init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # set for input
        self.mean = torch.from_numpy(np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)).to(self.device)
        self.std = torch.from_numpy(np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)).to(self.device)

        # used for calculating loss
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.sobel = Sobel().to(self.device)
        self.laplacian = Laplacian(3).to(self.device)

        # used for test and validation
        self.template = torch.zeros((1, 1, 1, 1))
        self.running_res = 0.
        self.running_num = 0
        self.metrics = {'average_mse': 0.0, 'average_sad': 0.0}

        # set output size
        self.batch_size = self.opt.n_frames 

        # set vis names
        self.visual_names = []
        for i in range(self.batch_size):
            if self.opt.vis_more:
                self.visual_names += ['merge_%02d' % i, 'trimap_%02d' % i, 'alpha_%02d' % i, 'predict_%02d' % i]
            else:
                self.visual_names += ['predict_%02d' % i]

        if self.isTrain:
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.netBase.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.netBase.parameters(), lr=opt.lr, momentum=opt.momentum)
            else:
                raise not NotImplementedError
            self.optimizers.append(self.optimizer)

    def normalize(self):
        self.merge_rt = (self.merge_rt / 255. - self.mean) / self.std
        self.trimap_rt = self.trimap_rt / 255.
        if self.alpha_rt is not None:
            self.alpha_rt = self.alpha_rt / 255.

    def set_input(self, input, phase):
        self.phase = phase
        self.ignore_dim = 0
        if self.phase == 'train':
            self.set_input_train(input)
        elif self.phase == 'val':
            self.set_input_val(input)
        else:
            self.set_input_test(input)

    def set_input_train(self, input):
        # place on cpu for vis
        self.input_info = input['info']
        self.image_paths = input['image_paths']

        self.merge = input['merge'].squeeze(self.ignore_dim)
        self.trimap = input['trimap'].squeeze(self.ignore_dim)
        self.alpha = input['alpha'].squeeze(self.ignore_dim)
        self.fg = input['fg'].squeeze(self.ignore_dim)
        self.bg = input['bg'].squeeze(self.ignore_dim)

        for i in range(self.batch_size):
            setattr(self, 'merge_%02d' % i, self.merge[self.shift+i:self.shift+i+1])
            setattr(self, 'trimap_%02d' % i, self.trimap[self.shift+i:self.shift+i+1])
            setattr(self, 'alpha_%02d' % i, self.alpha[self.shift+i:self.shift+i+1])

        # set input for network
        self.merge_rt = self.merge.to(self.device, non_blocking=True)
        self.trimap_rt = self.trimap.to(self.device, non_blocking=True)
        self.alpha_rt = self.alpha.to(self.device, non_blocking=True)
        self.fg_rt = self.fg.to(self.device, non_blocking=True)
        self.bg_rt = self.bg.to(self.device, non_blocking=True)
        self.normalize()

    def set_input_val(self, input):
        # place on cpu for vis
        self.input_info = input['info']
        self.image_paths = input['image_paths']

        self.merge = input['merge'].squeeze(self.ignore_dim)
        self.trimap = input['trimap'].squeeze(self.ignore_dim)
        self.alpha = input['alpha'].squeeze(self.ignore_dim)

        for i in range(self.batch_size):
            setattr(self, 'merge_%02d' % i, self.merge[self.shift+i:self.shift+i+1])
            setattr(self, 'trimap_%02d' % i, self.trimap[self.shift+i:self.shift+i+1])
            setattr(self, 'alpha_%02d' % i, self.alpha[self.shift+i:self.shift+i+1])

        self.merge_rt, self.trimap_rt, self.alpha_rt = None, None, None

    def set_input_test(self, input):
        # place on cpu for vis
        self.input_info = input['info']
        self.image_paths = input['image_paths']

        self.merge = input['merge'].squeeze(self.ignore_dim)
        self.trimap = input['trimap'].squeeze(self.ignore_dim)
        if 'alpha' in input:
            self.alpha = input['alpha'].squeeze(self.ignore_dim)
        else:
            self.alpha = None

        for i in range(self.batch_size):
            setattr(self, 'merge_%02d' % i, self.merge[self.shift+i:self.shift+i+1])
            setattr(self, 'trimap_%02d' % i, self.trimap[self.shift+i:self.shift+i+1])
            if self.alpha is not None:
                setattr(self, 'alpha_%02d' % i, self.alpha[self.shift+i:self.shift+i+1])

        self.merge_rt, self.trimap_rt, self.alpha_rt = None, None, None

    def forward(self):
        if self.opt.use_less_trimaps:
            trimap_rt = self.trimap_rt[self.shift:self.shift+1].repeat(self.trimap_rt.size(0), 1, 1, 1)
            x = torch.cat((self.merge_rt, trimap_rt), 1)
        else:
            x = torch.cat((self.merge_rt, self.trimap_rt), 1)
        self.pred_mattes, self.pred_alpha = self.netBase(x)

        for i in range(self.pred_mattes.size(0)):
            if self.opt.add_refine_stage and self.opt.refine_loss:
                setattr(self, 'predict_%02d' % i, self.pred_alpha[i:i+1] * 255.)
            else:
                setattr(self, 'predict_%02d' % i, self.pred_mattes[i:i+1] * 255.)

    def gen_alpha_loss(self, pred, alpha, smask):
        diff = (pred - alpha) * smask
        loss = torch.sqrt(diff ** 2 + 1e-12)
        loss = loss.sum() / (smask.sum() + 1.)
        return loss

    def gen_comp_loss(self, pred, img, fg, bg, alpha, smask):
        pred = pred.repeat((1,3,1,1))
        comp = pred * fg + (1. - pred) * bg
        comp = (comp / 255. - self.mean) / self.std
        loss = torch.sqrt((comp - img) ** 2 + 1e-12)
        loss = (loss * smask).sum() / (smask.sum() + 1.) / 3.
        return self.opt.comp_weight * loss

    def gen_diff_loss(self, pred, alpha, smask):
        hmask = 1 - smask
        loss = torch.sqrt((pred - alpha) ** 2 + 1e-12)
        soft_loss = self.opt.soft_weight * (loss * smask).sum() / (smask.sum() + 1.)
        hard_loss = self.opt.hard_weight * (loss * hmask).sum() / (hmask.sum() + 1.)
        return soft_loss + hard_loss

    def gen_sobel_loss(self, pred, alpha, smask):
        pred_sobel = self.sobel(pred)
        alpha_sobel = self.sobel(alpha)
        mask = smask * (alpha_sobel - pred_sobel.detach()).abs() * 2.0
        loss = self.l1_loss(pred*mask, alpha*mask) / (mask.sum() + 1.)
        return self.opt.grad_weight * loss

    def gen_laplacian_loss(self, pred, alpha, smask):
        pred_lap = self.laplacian(pred)
        alpha_lap = self.laplacian(alpha)
        mask = smask * (alpha_lap - pred_lap.detach()).abs() * 3.0 
        loss = self.l1_loss(pred*mask, alpha*mask) / (mask.sum() + 1.)
        return self.opt.grad_weight * loss

    def gen_kld_loss(self, pred, alpha, smask):
        pred_logit = (pred+1e-10) / (pred.sum() + 1e-5)
        alpha_logit = (alpha+1e-10) / (alpha.sum() + 1e-5)
        loss = self.kld_loss(pred_logit.log(), alpha_logit)
        return self.opt.kld_weight * loss

    def gen_temp_loss(self, pred, alpha, mask):
        diff_pred = pred[1:] - pred[:-1]
        diff_alpha = alpha[1:] - alpha[:-1]
        loss = self.l2_loss(diff_pred, diff_alpha)
        return self.opt.temp_weight * loss
    
    def backward(self):
        self.loss_total = 0.0
        mask = (self.trimap_rt[self.shift:]>0.).float() * (self.trimap_rt[self.shift:]<1.).float() 

        # basic loss
        self.loss_diff = self.gen_diff_loss(self.pred_mattes, self.alpha_rt[self.shift:], mask)
        self.loss_total += self.loss_diff

        if self.opt.comp_loss:
            self.loss_comp = self.gen_comp_loss(self.pred_mattes, self.merge_rt[self.shift:], 
                self.fg_rt[self.shift:], self.bg_rt[self.shift:], self.alpha_rt[self.shift:], mask)
            self.loss_total += self.loss_comp
        if self.opt.grad_loss:
            self.loss_grad = self.gen_sobel_loss(self.pred_mattes, self.alpha_rt[self.shift:], mask)
            self.loss_total += self.loss_grad
        if self.opt.kld_loss:
            self.loss_kld = self.gen_kld_loss(self.pred_mattes, self.alpha_rt[self.shift:], mask)
            self.loss_total += self.loss_kld
        if self.opt.temp_loss:
            self.loss_temp = self.gen_temp_loss(self.pred_mattes, self.alpha_rt[self.shift:], mask)
            self.loss_total += self.loss_temp
        if self.opt.add_refine_stage and self.opt.refine_loss:
            self.loss_refine = self.gen_diff_loss(self.pred_alpha, self.alpha_rt[self.shift:], mask)
            self.loss_total += self.loss_refine * self.opt.refine_weight

        if self.isTrain:
            self.loss_total.backward()

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if opt.load_pretrain:
            print("loading pretrain '{}'".format(opt.load_dir))
            ckpt = torch.load(opt.load_dir)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            if isinstance(self.netBase, torch.nn.DataParallel):
                self.netBase.module.load_state_dict(ckpt,strict=False)
            else:
                self.netBase.load_state_dict(ckpt,strict=False)
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix, strict=False)
        
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.netBase.parameters(), 1.0)
        self.optimizer.step()

    def test_with_crop(self):
        n, c, h, w = self.merge.size()
        predict = self.template.repeat((self.batch_size, 1, h, w))
        cnt = torch.zeros_like(predict)

        step_h = self.opt.load_size
        step_w = self.opt.load_size

        haxis = list(range(0, h-step_h, step_h-40)) + [h-step_h]
        waxis = list(range(0, w-step_w, step_w-40)) + [w-step_w]

        for start_h in haxis:
            end_h = start_h + step_h
            for start_w in waxis:
                end_w = start_w + step_w

                merge_i = self.merge[:, :, start_h:end_h, start_w:end_w]
                trimap_i = self.trimap[:, :, start_h:end_h, start_w:end_w]
                if (trimap_i == 128).sum() <= 0:
                    continue

                self.merge_rt = merge_i.contiguous().to(self.device, non_blocking=True)
                self.trimap_rt = trimap_i.contiguous().to(self.device, non_blocking=True)
                self.normalize()
                self.forward()

                if self.opt.add_refine_stage: 
                    predict[:,:,start_h:end_h,start_w:end_w] += self.pred_alpha.data.cpu()
                else:
                    predict[:,:,start_h:end_h,start_w:end_w] += self.pred_mattes.data.cpu()
                cnt[:,:,start_h:end_h,start_w:end_w] += 1

        cnt[cnt<1] = 1
        predict = predict / cnt
        predict[self.trimap[self.shift:] == 0] = 0
        predict[self.trimap[self.shift:] == 255] = 1
        for i in range(predict.size(0)):
            setattr(self, 'predict_%02d' % i, predict[i:i+1] * 255.)
        self.compute_visuals()

    def test_with_whole(self):
        n, c, h, w = self.merge[self.shift:].size()
        predict = self.template.repeat((n, 1, h, w))
        nh, nw = h-h%32, w-w%32
        self.merge_rt = self.merge.to(self.device, non_blocking=True)
        self.trimap_rt = self.trimap.to(self.device, non_blocking=True)
        self.merge_rt = F.interpolate(self.merge_rt, size=(nh, nw))
        self.trimap_rt = F.interpolate(self.trimap_rt, size=(nh, nw))
        self.normalize()
        self.forward()
        if self.opt.add_refine_stage: 
            predict = self.pred_alpha.data.cpu()
        else:
            predict = self.pred_mattes.data.cpu()
        predict = F.interpolate(predict, size=(h, w))
        predict[self.trimap[self.shift:] == 0] = 0
        predict[self.trimap[self.shift:] == 255] = 1
        for i in range(predict.size(0)):
            setattr(self, 'predict_%02d' % i, predict[i:i+1] * 255.)
        self.compute_visuals()

    def test_with_resize(self):
        n, c, oh, ow = self.merge[self.shift:].size()
        ratio = self.opt.max_size / max(oh, ow)
        rh, rw = int(ratio * oh), int(ratio * ow)
        rh, rw = rh-rh%32, rw-rw%32
        predict = self.template.repeat((n, 1, rh, rw))
        self.merge_rt = self.merge.to(self.device, non_blocking=True)
        self.trimap_rt = self.trimap.to(self.device, non_blocking=True)
        self.merge_rt = F.interpolate(self.merge_rt, size=(rh, rw))
        self.trimap_rt = F.interpolate(self.trimap_rt, size=(rh, rw))
        self.normalize()
        self.forward()
        if self.opt.add_refine_stage: 
            predict = self.pred_alpha.data.cpu()
        else:
            predict = self.pred_mattes.data.cpu()
        predict = F.interpolate(predict, size=(oh, ow))
        predict[self.trimap[self.shift:] == 0] = 0
        predict[self.trimap[self.shift:] == 255] = 1
        for i in range(predict.size(0)):
            setattr(self, 'predict_%02d' % i, predict[i:i+1] * 255.)
        self.compute_visuals()

    def clear(self):
        self.running_res = 0.
        self.running_num = 0
        for k, v in self.metrics.items():
            self.metrics[k] = 0.

    def test(self):
        with torch.no_grad():
            if self.opt.test_mode == 'crop':
                self.test_with_crop()
            elif self.opt.test_mode == 'whole':
                self.test_with_whole()
            elif self.opt.test_mode == 'resize':
                self.test_with_resize()
            else:
                raise NotImplementedError

    def calculate_metric(self):
        for i in range(self.batch_size):
            alpha = getattr(self, 'alpha_%02d' % i).to(self.device) / 255.
            pred = getattr(self, 'predict_%02d' % i).to(self.device) / 255.
            trimap = getattr(self, 'trimap_%02d' % i).to(self.device)
            mask = (trimap == 128).float()
            diff = (pred - alpha) * mask
            mse = diff.pow(2).sum() / (mask.sum()+1.)
            sad = diff.abs().sum() 
            print('Testing: mse = %.3f, sad = %.3f' % (mse, sad))
            self.running_num += 1
            self.running_res += sad
            self.metrics['average_mse'] += mse
            self.metrics['average_sad'] += sad

    def get_metrics(self):
        res = self.running_res / (self.running_num + 1e-8)
        for k, v in self.metrics.items():
            self.metrics[k] = v / (self.running_num + 1e-8)
        return res, self.metrics
