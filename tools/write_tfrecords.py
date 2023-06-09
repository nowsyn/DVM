import os
import re
import glob
import time
import cv2
import torch
import tfrecord
import argparse
from PIL import Image
from tfrecord import example_pb2
from tfrecord.torch.dataset import TFRecordDataset
from multiprocessing import Pool


def scan_videos(dataroot:
    video_pairs = []
    full_paths = glob.glob(os.path.join(dataroot, "a", "*/"))
    names = sorted([n.split("/")[-2] for n in full_paths])

    print("In total: %d videos" % len(names))

    for i, n in enumerate(names):
        v_folder = os.path.join(dataroot, "v", n)
        f_folder = os.path.join(dataroot, "f", n)
        b_folder = os.path.join(dataroot, "b", n)
        a_folder = os.path.join(dataroot, "a", n)
        t_folder = os.path.join(dataroot, "t", n)
        video_pairs.append((n, v_folder, f_folder, b_folder, a_folder, t_folder))
    return video_pairs


def load_frames_vfba(name, vpath, fpath, bpath, apath):
    vfiles = sorted(glob.glob(vpath+'/*.jpg'))
    ffiles = sorted(glob.glob(fpath+'/*.jpg'))
    bfiles = sorted(glob.glob(bpath+'/*.jpg'))
    afiles = sorted(glob.glob(apath+'/*.jpg'))
    for (vp, fp, bp, ap) in zip(vfiles, ffiles, bfiles, afiles):
        v = open(vp, 'rb').read()
        f = open(fp, 'rb').read()
        b = open(bp, 'rb').read()
        a = open(ap, 'rb').read()
        datum = { 
            "v": (v, "byte"),
            "f": (f, "byte"),
            "b": (b, "byte"),
            "a": (a, "byte"),
        }
        yield datum 


def load_frames_va(name, vpath, apath):
    vfiles = sorted(glob.glob(vpath+'/*.jpg'))
    afiles = sorted(glob.glob(apath+'/*.jpg'))
    for (vp, ap) in zip(vfiles, afiles):
        v = open(vp, 'rb').read()
        a = open(ap, 'rb').read()
        datum = { 
            "v": (v, "byte"),
            "a": (a, "byte"),
        }
        yield datum 


def load_frames_vt(name, vpath, tpath):
    vfiles = sorted(glob.glob(vpath+'/*.jpg'))
    tfiles = sorted(glob.glob(tpath+'/*.jpg'))
    for (vp, tp) in zip(vfiles, tfiles):
        v = open(vp, 'rb').read()
        t = open(tp, 'rb').read()
        datum = { 
            "v": (v, "byte"),
            "t": (t, "byte"),
        }
        yield datum


def load_frames_vat(name, vpath, apath, tpath):
    vfiles = sorted(glob.glob(vpath+'/*.jpg'))
    afiles = sorted(glob.glob(apath+'/*.jpg'))
    tfiles = sorted(glob.glob(tpath+'/*.jpg'))
    for (vp, ap, tp) in zip(vfiles, afiles, tfiles):
        v = open(vp, 'rb').read()
        a = open(ap, 'rb').read()
        t = open(tp, 'rb').read()
        datum = { 
            "v": (v, "byte"),
            "a": (a, "byte"),
            "t": (t, "byte"),
        }
        yield datum


def generate_index(srcpath, dstpath):
    cmd = "python -m tfrecord.tools.tfrecord2idx {} {}".format(srcpath, dstpath)
    cmd = cmd.replace('(', '\(')
    cmd = cmd.replace(')', '\)')
    ret = os.system(cmd)
    return ret


def write_tfrecord_one(item):
    svpath, dstpath, n, vpath, fpath, bpath, apath, tpath, mode = item
    writer = tfrecord.TFRecordWriter(dstpath)
    if mode == 'vfba':
        iterator = load_frames_vfba(n, vpath, fpath, bpath, apath)
    elif mode == 'va':
        iterator = load_frames_va(n, vpath, apath)
    elif mode == 'vt':
        iterator = load_frames_vt(n, vpath, tpath)
    elif mode == 'vat':
        iterator = load_frames_vat(n, vpath, apath, tpath)
    else:
        raise NotImplementedError
    for idx, datum in enumerate(iterator):
        datum['index'] = ([idx], "int")
        writer.write(datum)
    writer.close()
    idxpath = os.path.join(svpath, n+'.index')
    generate_index(dstpath, idxpath)
    return dstpath


def write_tfrecord(dataroot, svpath, human_only, mode='vfa', n_proc=24):
    video_pairs = scan_videos(dataroot, human_only=human_only)
    pool = Pool(n_proc)
    items = []
    for n, vpath, fpath, bpath, apath, tpath in video_pairs:
        dstpath = os.path.join(svpath, n+'.tfrecord')
        if os.path.exists(dstpath):
            print(dstpath)
            continue
        items.append((svpath, dstpath, n, vpath, fpath, bpath, apath, tpath, mode))
    for ret in pool.imap(write_tfrecord_one, items):
        print(ret)


def verify(rdpath, svpath, name, mode):
    vpath = os.path.join(rdpath, 'v', name)
    fpath = os.path.join(rdpath, 'f', name)
    bpath = os.path.join(rdpath, 'b', name)
    apath = os.path.join(rdpath, 'a', name)
    tpath = os.path.join(rdpath, 't', name)

    dstpath = os.path.join(svpath, name+'.tfrecord')
    writer = tfrecord.TFRecordWriter(dstpath)

    if mode == 'vfba':
        iterator = load_frames_vfba(name, vpath, fpath, bpath, apath)
    elif mode == 'va':
        iterator = load_frames_va(name, vpath, apath)
    elif mode == 'vt':
        iterator = load_frames_vt(name, vpath, tpath)
    else:
        raise NotImplementedError
    for idx, datum in enumerate(iterator):
        datum['index'] = ([idx], "int")
        writer.write(datum)
    writer.close()

    idxpath = os.path.join(svpath, name+'.index')
    ret = generate_index(dstpath, idxpath)
    print('ret', ret)


def main():
    parser = argparse.ArgumentParser(description='arguments passed the program')
    parser.add_argument('-i', "--input", type=str, help="Input folder for videos")
    parser.add_argument('-o', "--output", type=str, help="Onput folder for videos")
    parser.add_argument('-s', "--stage", type=str, help="running stage, [write | verify | check]")
    parser.add_argument('-m', "--mode", type=str, default='vfba', choices=['vfba', 'va', 'vt', 'vat'], help="Input mode")
    parser.add_argument('--human_only', action='store_true', help="Process human videos only")
    args = parser.parse_args()

    if args.stage == 'write':
        os.makedirs(args.output, exist_ok=True)
        write_tfrecord(args.input, args.output, args.human_only, args.mode)

    if args.stage == 'verify':
        t1 = time.time()
        record_paths = sorted(glob.glob(args.output+"/*.tfrecord"))
        record_paths = sorted(open('check.log').read().splitlines())
        for idx, record_path in enumerate(record_paths):
            name = os.path.splitext(os.path.basename(record_path))[0]
            index_path = os.path.join(args.output, name+'.index')
            description = {"v": "byte", "f": "byte", "b": "byte", "a": "byte", "index": "int"}
            loader = tfrecord.tfrecord_loader(record_path, index_path, description)
            try:
                for i, record in enumerate(loader):
                    print(name, i)
            except:
                print('Error: {}'.format(record_path))
                verify(args.input, args.output, name, args.mode)
        t2 = time.time()
        print('time', (t2-t1)/(idx+1))

    if args.stage == 'check':
        record_paths = sorted(glob.glob(args.output+"/*.tfrecord"))
        record_paths = sorted(open('check.log').read().splitlines())
        writer = open('check.log', 'w')
        for record_path in record_paths:
            name = os.path.splitext(os.path.basename(record_path))[0]
            index_path = os.path.join(args.output, name+'.index')
            description = {"v": "byte", "f": "byte", "b": "byte", "a": "byte", "index": "int"}
            loader = tfrecord.tfrecord_loader(record_path, index_path, description)
            try:
                for sample in loader:
                    v = sample['v']
                    f = sample['f']
                    b = sample['b']
                    a = sample['a']
                    v  = cv2.imdecode(v, cv2.IMREAD_COLOR)
                    fg = cv2.imdecode(f, cv2.IMREAD_COLOR)
                    bg = cv2.imdecode(b, cv2.IMREAD_COLOR)
                    a  = cv2.imdecode(a, cv2.IMREAD_COLOR)
                    print(v.shape, fg.shape, bg.shape, a.shape)
            except:
                print(record_path)
                writer.write(record_path+'\n')

        
if __name__ == '__main__':
    main()
