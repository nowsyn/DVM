import os
import cv2
import glob
import time
import math
import ntpath
import random
import numpy as np
from PIL import Image
from multiprocessing import Pool

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

from data.base_dataset import BaseDataset, get_transform
from data.tfrecord_reader import tfrecord_loader, tfrecord_index_loader


class VideoDB(object):
    def __init__(self, dataroot, max_n_video=-1, data_phase='train', shuffle=False):
        self.dataroot = dataroot
        self.max_n_video = max_n_video
        self.shuffle = shuffle
        self.data_phase = data_phase
        if self.data_phase == 'train' or 'composition' in dataroot:
            self.description = {"v": "byte", "f": "byte", "b": "byte", "a": "byte", "index": "int"}
        else:
            self.description = {"v": "byte",  "a": "byte", "t": "byte", "index": "int"}
        self.scan_videos()
        self.load_index()

    def filter(self, video_pairs, filter_dict, include=True):
        video_pairs_kept = []
        for name, record_path, index_path in video_pairs:
            prefix = "_".join(name.split("_")[:-1])
            if include and (prefix in filter_dict):
                video_pairs_kept.append((name, record_path, index_path))
            elif (not include) and (prefix not in filter_dict):
                video_pairs_kept.append((name, record_path, index_path))
        return video_pairs_kept

    def scan_videos(self):
        record_paths = sorted(glob.glob(self.dataroot + '/*.tfrecord'))
        index_paths = sorted(glob.glob(self.dataroot + '/*.index'))
        names = [os.path.splitext(os.path.basename(p))[0] for p in record_paths]

        video_pairs = list(zip(names, record_paths, index_paths))
        self.video_pairs = []
        for name, record_path, index_path in video_pairs:
            self.video_pairs.append((name, record_path, index_path))
        if self.shuffle:
            random.shuffle(self.video_pairs)
        if self.max_n_video > 0:
            self.video_pairs = self.video_pairs[:self.max_n_video]
        print("In total: %d videos" % len(self.video_pairs))

    def load_index(self):
        self.sizes = []
        for _, _, index_path in self.video_pairs:
            index = tfrecord_index_loader(index_path)
            self.sizes.append(len(list(index)))
        self.n_videos = len(self.sizes)

    def pad_chunk(self, vlist, flist, blist, alist, tlist, num, first=True):
        if first:
            vlist = [vlist[0]] * num + vlist
            if len(flist)>0:
                flist = [flist[0]] * num + flist
            if len(blist)>0:
                blist = [blist[0]] * num + blist
            if len(alist)>0:
                alist = [alist[0]] * num + alist
            if len(tlist)>0:
                tlist = [tlist[0]] * num + tlist
        else:
            vlist = vlist + [vlist[-1]] * num
            if len(flist)>0:
                flist = flist + [flist[-1]] * num
            if len(blist)>0:
                blist = blist + [blist[-1]] * num
            if len(alist)>0:
                alist = alist + [alist[-1]] * num
            if len(tlist)>0:
                tlist = tlist + [tlist[-1]] * num
        return vlist, flist, blist, alist, tlist

    def load_frames(self, video_no, start, num):
        n_frames, end = self.sizes[video_no], start+num
        astart, aend = max(0, start), min(n_frames, end)
        name, record_path, index_path = self.video_pairs[video_no] 
        vlist, flist, blist, alist, tlist = [], [], [], [], []
        loader = tfrecord_loader(record_path, index_path, self.description, chunk=(astart, aend-astart))
        for i, record in enumerate(loader):
            vlist.append(cv2.imdecode(record['v'], cv2.IMREAD_COLOR))
            if 'f' in record:
                flist.append(cv2.imdecode(record['f'], cv2.IMREAD_COLOR))
            if 'b' in record:
                blist.append(cv2.imdecode(record['b'], cv2.IMREAD_COLOR))
            if 'a' in record:
                alist.append(cv2.imdecode(record['a'], cv2.IMREAD_COLOR))
            if 't' in record:
                tlist.append(cv2.imdecode(record['t'], cv2.IMREAD_COLOR))
        if start < 0:
            vlist, flist, blist, alist, tlist = self.pad_chunk(
                vlist, flist, blist, alist, tlist, num=(-start), first=True)
        if end > n_frames:
            vlist, flist, blist, alist, tlist = self.pad_chunk(
                vlist, flist, blist, alist, tlist, num=(end-n_frames), first=False)
        assert len(vlist) == num, 'Not enough frames loaded. Expected = %d, Actual = %d' % (num, len(vlist)) 
        return vlist, flist, blist, alist, tlist

    def iterate_chunks(self, size, stride=1, max_n_chunk=-1):
        chunk_list = []
        for i in range(self.n_videos):
            step = size * stride
            n_frames = self.sizes[i]
            chunks = list(range(0, n_frames-step, step)) + [n_frames-step]
            chunk_list_i = []
            for j in chunks:
                for k in range(stride):
                    chunk_list_i.append((i, j + k))
            if max_n_chunk > 0:
                chunk_list_i = chunk_list_i[:max_n_chunk] 
            chunk_list.extend(chunk_list_i)
        return chunk_list

    def iterate_frames(self, start=0, stride=1, max_n_frame=-1):
        frame_list = []
        for i in range(self.n_videos):
            n_frames = self.sizes[i]
            frame_list_i = []
            if max_n_frame > 0:
                n_frames = min(n_frames, max_n_frame)
            for j in range(start, n_frames, stride):
                frame_list_i.append((i, j))
            frame_list.extend(frame_list_i)
        return frame_list

    def index2name(self, index):
        return self.video_pairs[index][0]


class TfrecordDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--data_phase", type=str, choices=['train', 'val', 'test'], default="train", 
            help="load data in different ways when phase is train or val")
        parser.add_argument("--data_type", type=str, choices=['video', 'image'], default="image", 
            help="data to be loaded is image or video")
        parser.add_argument("--n_frames", type=int, default=1, 
            help="number of frames extracted for a single datapoint")
        parser.add_argument("--frame_loading", type=str, choices=["frame", "chunk"], default="frame",
            help="load frame or chunks as input") 
        parser.add_argument("--stride", type=int, default=1, 
            help="stride of when frame loading is chunk")
        parser.add_argument("--start_frame", type=int, default=0, 
            help="starting frame number")
        parser.add_argument("--max_n_video", type=int, default=-1, 
            help="The max number of videos used in the training (default means no limit on it)")
        parser.add_argument("--max_n_chunk", type=int, default=-1,  
            help="The max number of chunks used in the testing (default means no limit on it)")
        parser.add_argument("--max_n_frame", type=int, default=-1,  
            help="The max number of frames used in the testing (default means no limit on it)")
        parser.add_argument("--tensor_scaling", action="store_true", 
            help="whether to scale image tensor when save or display images")
        # augmentations
        parser.add_argument('--random_rotation_degree', type=int, default=15, 
            help='set range for random rotation degree')
        parser.add_argument("--trimap_dilation", type=int, default=1, 
            help="Whether to enlarge the transition region of the trimap. This set the filter size, must by an odd integer")
        parser.add_argument("--cal_of", action="store_true", 
            help="whether to calculate the optical flow and include it in the dataset")
        parser.add_argument("--use_less_trimaps", action="store_true", 
            help="whether to use less trimaps")
        # others
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.data_phase = self.opt.data_phase
        if self.data_phase != 'train':
            self.dataroot = self.opt.dataroot_val
        else:
            self.dataroot = self.opt.dataroot

        # load dataset
        self.video_db = VideoDB(self.dataroot, self.opt.max_n_video, self.data_phase)
        self.n_videos = self.video_db.n_videos
        self.sizes = self.video_db.sizes
        self.chunk_list = self.video_db.iterate_chunks(self.opt.n_frames, max_n_chunk=self.opt.max_n_chunk)
        self.frame_list = self.video_db.iterate_frames(max_n_frame=self.opt.max_n_frame)
        self.n_chunks = len(self.chunk_list)
        self.n_frames = len(self.frame_list)
        self.curr = None

        self.crop_sizes = [int(size) for size in self.opt.crop_sizes.split(',')]

        if self.opt.frame_loading == 'frame':
            if self.data_phase == 'train':
                self.frame_loading = 'jump'
                self.shift = 0 
            else:
                self.frame_loading = 'cont'
                self.shift = 0 
        if self.opt.frame_loading == 'chunk':
            if self.data_phase == 'train':
                self.frame_loading = 'chunk'
                self.shift = self.opt.shift
            else:
                self.frame_loading = 'chunk_cont'
                self.shift = self.opt.shift

        if self.frame_loading in ['jump', 'cont']:
            assert self.opt.n_frames == 1, 'Unsupported number of images train/test.'

        if self.frame_loading == "jump": # for image train
            self.dataset_size = self.n_videos
        elif self.frame_loading == "cont": # form image test
            self.dataset_size = self.n_frames
        elif self.frame_loading == "chunk": # for video train
            self.dataset_size = self.n_videos
        elif self.frame_loading == "chunk_cont": # for video test
            self.dataset_size = self.n_chunks
        return

    def __getitem__(self, index):
        index = index % self.dataset_size
        if self.frame_loading == "jump":
            self.video_no = index
            self.curr = random.randint(0, self.sizes[self.video_no]-(self.opt.n_frames*self.opt.stride)-1)
            start = self.curr
            size = self.opt.n_frames
        elif self.frame_loading == 'cont':
            self.video_no, self.curr = self.frame_list[index]
            start = self.curr
            size = self.opt.n_frames
        elif self.frame_loading == "chunk":
            self.video_no = index
            self.curr = random.randint(self.shift, self.sizes[self.video_no]-(self.opt.n_frames*self.opt.stride)-1)
            start = self.curr - self.shift
            size = self.opt.n_frames + self.shift
        elif self.frame_loading == "chunk_cont":
            self.video_no, self.curr = self.chunk_list[index]
            start = self.curr - self.shift
            size = self.opt.n_frames + self.shift
        else:
            raise NotImplementedError
        
        save_name = "%s/%05d.png" % (self.video_db.index2name(self.video_no), self.curr)
        vlist, flist, blist, alist, tlist = self.video_db.load_frames(self.video_no, start, size)
        vcat, fcat, bcat, acat, tcat, info = self.processNumpyFrames(vlist, flist, blist, alist, tlist)
        if self.opt.use_less_trimaps:
            if self.frame_loading == 'cont' and start%2==1:
                vlist1, flist1, blist1, alist1, tlist1 = self.video_db.load_frames(self.video_no, start-1, size)
                vcat1, fcat1, bcat1, acat1, tcat1, info1 = self.processNumpyFrames(vlist1, flist1, blist1, alist1, tlist1)
                tcat = tcat1
            else:
                tcat[:] = tcat[self.shift]
        assert tcat is not None, 'Trimap must be provided or generated from alpha.'

        result = {"merge": vcat, "trimap": tcat, "image_paths": save_name, "info": info}
        if acat is not None:
            result['alpha'] = acat
        if fcat is not None:
            result['fg'] = fcat
        if bcat is not None:
            result['bg'] = bcat
        if self.opt.cal_of:
            flow = calculateDenseOpticalFlow(vcat)
            result['flow'] = flow
        return result

    def processNumpyFrames(self, vlist, flist, blist, alist, tlist):
        if self.data_phase == 'train':
            vlist, flist, blist, alist, tlist, info = self.processNumpyFramesForTrain(vlist, flist, blist, alist, tlist)
        else:
            vlist, flist, blist, alist, tlist, info = self.processNumpyFramesForTest(vlist, flist, blist, alist, tlist)
        vcat = torch.stack(vlist)
        fcat = torch.stack(flist) if len(flist)>0 else None
        bcat = torch.stack(blist) if len(blist)>0 else None
        acat = torch.stack(alist) if len(alist)>0 else None
        tcat = torch.stack(tlist) if len(tlist)>0 else None
        return vcat, fcat, bcat, acat, tcat, info

    def processNumpyFramesForTrain(self, vlist, flist, blist, alist, tlist):
        ref_alpha = alist[self.shift][:, :, 0]
        h, w = ref_alpha.shape
        info = {}
        info['n_frames'] = len(vlist) 
        info["origin_h"] = h 
        info["origin_w"] = w

        crop_size = self.crop_sizes[np.random.randint(len(self.crop_sizes))]
        ratio = float(crop_size)/min(h, w)
        nh = int(ratio*h) + 1 if ratio>1 else h
        nw = int(ratio*w) + 1 if ratio>1 else w
        delta_h = crop_size / 2
        delta_w = crop_size / 2
        target = np.where((ref_alpha > 0) & (ref_alpha < 255))
        if len(target[0]) > 0:
            rand_ind = np.random.randint(len(target[0]))
            center_h = min(max(target[0][rand_ind], delta_h), nh - delta_h)
            center_w = min(max(target[1][rand_ind], delta_w), nw - delta_w)
            start_h = int(center_h - delta_h)
            start_w = int(center_w - delta_w)
            crop_w, crop_h = crop_size, crop_size
            rotate = False
            angle = 0
            flip = (np.random.random() < 0.5)
        else:
            center_h = np.random.randint(delta_h, nh-delta_h)
            center_w = np.random.randint(delta_w, nw-delta_w)
            start_h = int(center_h - delta_h)
            start_w = int(center_w - delta_w)
            crop_w, crop_h = crop_size, crop_size
            rotate = False
            angle = 0
            flip = (np.random.random() < 0.5)

        setting = (crop_size, start_h, start_w, crop_h, crop_w, rotate, angle, flip, (h, w))
        vlist, _ = self.operate_for_train_np(vlist, 'i', setting)
        if len(flist)>0:
            flist, _ = self.operate_for_train_np(flist, 'i', setting)
        if len(blist)>0:
            blist, _ = self.operate_for_train_np(blist, 'i', setting)
        if len(alist)>0:
            alist, tlist = self.operate_for_train_np(alist, 'a', setting)
        if len(alist) == 0 and len(tlist)>0:
            tlist, _ = self.operate_for_train_np(tlist, 't', setting)
        return vlist, flist, blist, alist, tlist, info

    def processNumpyFramesForTest(self, vlist, flist, blist, alist, tlist):
        info = {'n_frames': len(vlist)}
        vlist, _ = self.operate_for_test_np(vlist, 'i')
        if len(flist)>0:
            flist, _ = self.operate_for_test_np(flist, 'i')
        if len(blist)>0:
            blist, _ = self.operate_for_test_np(blist, 'i')
        if len(alist)>0:
            alist, tlist = self.operate_for_test_np(alist, 'a')
        if len(alist)==0 and len(tlist)>0:
            tlist, _ = self.operate_for_test_np(tlist, 't')
        return vlist, flist, blist, alist, tlist, info

    def gen_trimap(self, alpha, k_size=3, iterations=5, fixed=False):
        if not fixed:
            k_size = random.choice(range(2, 5))
            iterations = np.random.randint(5, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        eroded = cv2.erode(alpha, kernel, iterations=iterations)
        trimap = np.zeros(alpha.shape, dtype=np.float32) + 128
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        return trimap

    def operate_for_train_np(self, batch, mode, setting):
        crop_size, start_h, start_w, crop_h, crop_w, rotate, angle, flip, size = setting
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        h, w = size
        ratio = float(crop_size) / min(w, h)
        M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
        xlist, ylist = [], []
        for  x in batch:
            if ratio > 1.0:
                x = cv2.resize(x, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            x = x[start_h:end_h, start_w:end_w, :]
            if rotate:
                x = cv2.warpAffine(x, M, (crop_w, crop_h), borderMode=cv2.BORDER_CONSTANT)
            if flip:
                x = x[:, ::-1, :]
            if x.shape[0]!=self.opt.load_size or x.shape[1]!=self.opt.load_size:
                x = cv2.resize(x, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_CUBIC)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype(np.float32).transpose((2,0,1))
            if mode == 'a' or mode == 't':
                x = x[0:1, :, :]
            xlist.append(torch.from_numpy(x))
            if mode == 'a':
                y = self.gen_trimap(x[0], fixed=False)
                ylist.append(torch.from_numpy(y[np.newaxis, :, :]))
        return xlist, ylist

    def operate_for_test_np(self, batch, mode):
        xlist, ylist = [], []
        for x in batch:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype(np.float32).transpose((2,0,1))
            if mode == 'a' or mode == 't':
                x = x[0:1, :, :]
            xlist.append(torch.from_numpy(x))
            if mode == 'a':
                y = self.gen_trimap(x[0], fixed=True)
                ylist.append(torch.from_numpy(y[np.newaxis, :, :]))
        return xlist, ylist

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def calculateDenseOpticalFlow(input):
    frames = input.data.numpy()
    frames = frames.transpose([0,2,3,1])
    frames = ((frames + 1) / 2 * 255).round().astype(np.uint8)

    flow = []
    for i in range(frames.shape[0]-1):
        prev = frames[i]
        next = frames[i+1]
        f = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow.append(f)
    flow = np.stack(flow)
    return flow
