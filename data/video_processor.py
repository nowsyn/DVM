import cv2, imutils
import glob, os
import random, math
import numpy as np
import ntpath
from multiprocessing import Pool

VIDEO = "video"
IMAGE = "image"
CODEC = "MJPG"
SAVE_EXT = "avi"

random.seed(2019)

def loadVideoNumpy(file_path, n_channels = 3):
    cap = cv2.VideoCapture(file_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(height, width)
    result = numpy.zeros((height, width, n_channels, n_frames))

    for i in range(n_frames):
        ret, frame = cap.read()
        # print(frame.shape) # (height, width, channels)
        result[:, :, :, i] = frame

    return result

# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/48097478
def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

class RandomWalk():
    def __init__(self,
            min_x, max_x, min_y, max_y,
            speed_range=(0.5,2),
            rotation_range=(-60, 60),
            angular_speed_range=(-1,1),
            zoom_range=(0.9,1.1),
            zoom_speed_range=(-0.015, 0.015)
            ):
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        self.min_speed, self.max_speed = speed_range[0], speed_range[1]
        self.min_angular_speed, self.max_angular_speed = angular_speed_range[0], angular_speed_range[1]
        self.min_rot, self.max_rot = rotation_range[0], rotation_range[1]
        self.min_zoom, self.max_zoom = zoom_range[0], zoom_range[1]
        self.min_zoom_speed, self.max_zoom_speed = zoom_speed_range[0], zoom_speed_range[1]
        self.x = (min_x + max_x)//2
        self.y = (min_y + max_y)//2
        self.A = 0 # rotation angle
        self.zoom = 0.9 # zoom factor
        self.count = 1

        self.randomV()
        self.randomB()
        self.randomZoomV()

    def randomV(self):
        self.a = random.random() * 2 * math.pi
        self.speed = random.uniform(self.min_speed, self.max_speed)
        self.v = [math.cos(self.a) * self.speed, math.sin(self.a) * self.speed]

    def randomB(self):
        self.B = random.uniform(self.min_angular_speed, self.max_angular_speed) # angular volecity

    def randomZoomV(self):
        self.zoom_v = random.uniform(self.min_zoom_speed, self. max_zoom_speed) # speed in which the zoom factor changes

    def detectCollision(self):
        if self.x < self.min_x or self.x > self.max_x:
            self.v[0] = -self.v[0]
            self.x += self.v[0]
        if self.y < self.min_y or self.y > self.max_y:
            self.v[1] = -self.v[1]
            self.y += self.v[1]
        if self.A < self.min_rot or self.A > self.max_rot:
            self.B = - self.B
            self.A += self.B
        # Instead of reverse the zooming speed, keep it at a constant zoom
        if self.zoom < self.min_zoom:
            self.zoom = self.min_zoom
        if self.zoom > self.max_zoom:
            self.zoom = self.max_zoom

        # if self.zoom < self.min_zoom or self.zoom > self.max_zoom:
        #     self.zoom_v = - self.zoom_v
        #     self.zoom += self.zoom_v


    def step(self):
        self.x += self.v[0]
        self.y += self.v[1]
        self.detectCollision()

        self.A += self.B

        self.zoom += self.zoom_v

        self.count += 1
        if self.count % 20 == 0:
            self.randomV()
            self.randomB()
            self.randomZoomV()

        return int(self.x), int(self.y), self.A, self.zoom

# https://blog.csdn.net/liuqinshouss/article/details/78696032
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img


class VideoProcessor():
    def __init__(self, data_root, save_root, file_list_path, phase, classes_used,
            translation = True,
            zooming = True,
            rotation = True,
            save_mode = VIDEO,
            bg_blur=True,
            max_fg_size = 1920,
            max_bg_size = 1920,
            max_frames = 150
        ):
        self.data_root = data_root
        self.save_root = save_root
        self.translation = translation
        self.zooming = zooming
        self.rotation = rotation
        self.save_mode = save_mode
        self.bg_blur = bg_blur
        # If the the longer dimension foreground is larger than max_fg_size, it will be resize to max_fg_size
        self.max_fg_size = float(max_fg_size)
        self.max_bg_size = float(max_bg_size)
        self.max_frames = max_frames

        # Scan the foreground video files
        self.all_samples = []
        for line in open(file_list_path).read().splitlines():
            img_or_video, fg, bg = line.split('+')
            if img_or_video == 'image':
                alpha = os.path.join(data_root, 'fg', img_or_video, phase, 'alpha', fg)
                fg = os.path.join(data_root, 'fg', img_or_video, phase, 'fg', fg)
                bg = os.path.join(data_root, 'bg', phase, bg)
                if os.path.exists(alpha) and os.path.exists(fg) and os.path.exists(bg):
                    self.all_samples.append((img_or_video, (alpha, fg, bg)))
            else:
                fg = os.path.join(data_root, 'fg', img_or_video, phase, fg)
                bg = os.path.join(data_root, 'bg', phase, bg)
                if os.path.exists(fg) and os.path.exists(bg):
                    self.all_samples.append((img_or_video, (fg, bg)))

    def batch_image(self, n_proc=1, max_n = float("inf")):
        random.seed(2019)
        batch = []
        for sample in self.all_samples:
            img_or_video, data = sample
            if img_or_video == 'image':
                a, f, b = data
                batch.append((b, (f, a)))
        for i in range(len(batch)):
            batch[i] = (i, batch[i])
        print("total number of datapoints from images:", len(batch))
        p = Pool(n_proc)
        p.map(self.processImage, batch)

    def batch_video(self, n_proc=1, max_n = float("inf")):
        random.seed(2019)
        batch = []
        for sample in self.all_samples:
            img_or_video, data = sample
            if img_or_video == 'video':
                f, b = data
                batch.append((f, b))
        for i in range(len(batch)):
            batch[i] = (i, batch[i])
        print("total number of datapoints from videos:", len(batch))
        p = Pool(n_proc)
        p.map(self.processVideo, batch)

    def processImage(self, b):
        self.max_frames = 60

        jobid, (b_path, image_pair) = b
        f_path, a_path = image_pair[0], image_pair[1]

        f_name = f_path.split('/')[-1].split(".")[-2]
        b_name = b_path.split('/')[-1].split(".")[-2]

        save_path = os.path.join(self.save_root, "v", "image_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_a = os.path.join(self.save_root, "a", "image_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_f = os.path.join(self.save_root, "f", "image_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_b = os.path.join(self.save_root, "b", "image_%s_%s.%s" % (f_name, b_name, SAVE_EXT))

        check_path = os.path.join(".".join(save_path.split(".")[:-1]), "00000.jpg")
        # if os.path.exists(check_path):
        #     return

        print(f_path)
        print(a_path)
        print(b_path)
        print(jobid)
        self.compositeVideoOnAnother(f_path, a_path, b_path, save_path, save_path_a, save_path_f, save_path_b, type=IMAGE)

    def processVideo(self, b):
        self.max_frames = 150

        jobid, (f, b) = b
        f_path = os.path.join(f, "f.mp4")
        a_path = os.path.join(f, "a.mp4")

        b_path = b
        f_name = f.split('/')[-1]
        b_name = b.split('/')[-1].split(".")[-2]
        save_path = os.path.join(self.save_root, "v", "video_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_a = os.path.join(self.save_root, "a", "video_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_f = os.path.join(self.save_root, "f", "video_%s_%s.%s" % (f_name, b_name, SAVE_EXT))
        save_path_b = os.path.join(self.save_root, "b", "video_%s_%s.%s" % (f_name, b_name, SAVE_EXT))

        check_path = os.path.join(".".join(save_path.split(".")[:-1]), "00000.jpg")
        # if os.path.exists(check_path):
        #     return

        print(f_path)
        print(a_path)
        print(b_path)
        print(jobid)
        self.compositeVideoOnAnother(f_path, a_path, b_path, save_path, save_path_a, save_path_f, save_path_b, type=VIDEO)

    def compositeVideoOnAnother(self, f_path, a_path, b_path, save_path, save_path_a, save_path_f, save_path_b, type=VIDEO):
        if type == VIDEO:
            # read the foreground video and alpha
            f_cap = cv2.VideoCapture(f_path)
            a_cap = cv2.VideoCapture(a_path)
            f_frames = int(f_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            a_frames = int(a_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            f_width_original = int(f_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            f_height_original = int(f_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if f_frames != a_frames: return
        elif type == IMAGE:
            # f_image = cv_imread(f_path)
            # a_image = cv_imread(a_path)
            # a_image = np.expand_dims(a_image, 2)
            # a_image = np.repeat(a_image, 3, axis=2)
            f_image = cv2.imread(f_path)
            a_image = cv2.imread(a_path)
            f_frames = 1e7

            f_height_original, f_width_original, _ = f_image.shape
        else:
            assert False

        # Read the background video
        b_cap = cv2.VideoCapture(b_path)
        b_frames = int(b_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        b_width_original = int(b_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        b_height_original = int(b_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # First shrink the foreground to max_fg_size if the foreground is too large
        fwr = self.max_fg_size / f_width_original
        fhr = self.max_fg_size / f_height_original
        fr = min(fwr, fhr)
        if fr < 1:
            fw_resize = int(f_width_original*fr)
            fh_resize = int(f_height_original*fr)
            # If the foreground is an image, resize it before compositing
            if type == IMAGE:
                f_image = cv2.resize(src=f_image, dsize=(fw_resize, fh_resize), interpolation=cv2.INTER_CUBIC)
                a_image = cv2.resize(src=a_image, dsize=(fw_resize, fh_resize), interpolation=cv2.INTER_CUBIC)
        else:
            fw_resize = f_width_original
            fh_resize = f_height_original

        # Then enlarge the background to cover the foreground if background is small
        bwr = fw_resize / float(b_width_original)
        bhr = fh_resize / float(b_height_original)
        br = max(bwr, bhr)
        if br > 1:
            b_width = int(b_width_original*br)
            b_height = int(b_height_original*br)
        else:
            b_width = b_width_original
            b_height = b_height_original

        # Then crop the useless background region
        bw_keep = int(min(min(min(int(fw_resize*1.2), b_width), self.max_bg_size), fw_resize*2))
        # bw_start = (b_width-bw_keep)//2
        bw_start = random.randint(0, b_width-bw_keep)
        bh_keep = int(min(min(min(int(fh_resize*1.2), b_height), self.max_bg_size), fh_resize*2))
        # bh_start = (b_height-bh_keep)//2
        bh_start = random.randint(0, b_height-bh_keep)
        # print(b_width, bw_start, bw_keep, '||', b_height, bh_start, bh_keep)
        bw_resize = b_width 
        bh_resize = b_height
        b_width = bw_keep
        b_height = bh_keep

        # print("f:", fw_resize, fh_resize)
        # print("b:", b_width, b_height)

        if self.save_mode == VIDEO:
            # init the writting handler of generated video and alpha ground truth
            fourcc = cv2.VideoWriter_fourcc(*CODEC)
            out = cv2.VideoWriter(save_path, fourcc, 25.0, (b_width, b_height))
            out_a = cv2.VideoWriter(save_path_a, fourcc, 25.0, (b_width, b_height))
            out_f = cv2.VideoWriter(save_path_f, fourcc, 25.0, (b_width, b_height))
            out_b = cv2.VideoWriter(save_path_b, fourcc, 25.0, (b_width, b_height))

        n_frames = min(f_frames, b_frames, self.max_frames)

        rw = RandomWalk(min_x=int(b_height*0.05), max_x=int(b_height*0.95), min_y=int(b_width*0.05), max_y=int(b_width*0.95))

        if not self.translation or type==VIDEO:
            paste_center_x = b_height//2
            paste_center_y = b_width//2

        for i in range(n_frames):
            if type == VIDEO:
                f_ret, f = f_cap.read()
                a_ret, a = a_cap.read()
                if f_ret == False or a_ret == False or f is None or a is None:
                    break
            else:
                f = np.copy(f_image)
                a = np.copy(a_image)
            b_ret, b = b_cap.read()

            # To handle a bug that happens sometimes during video loading
            if b_ret == False or b is None:
                break

            # shrink the foreground for each frame loading if the foreground is a video
            if fr < 1 and type == VIDEO:
                f = cv2.resize(src=f, dsize=(fw_resize, fh_resize), interpolation=cv2.INTER_CUBIC)
                a = cv2.resize(src=a, dsize=(fw_resize, fh_resize), interpolation=cv2.INTER_CUBIC)

            # enlarge the background if needed
            if br > 1:
                # b = cv2.resize(src=b, dsize=(bw_resize, bh_resize), interpolation=cv2.INTER_CUBIC)
                b = cv2.resize(src=b, dsize=(bw_resize, bh_resize), interpolation=cv2.INTER_LINEAR)

            # remove the useless background
            b = b[bh_start:bh_start+bh_keep, bw_start:bw_start+bw_keep, :]
            assert b.shape[0] == b_height
            assert b.shape[1] == b_width

            if self.bg_blur:
                b = cv2.GaussianBlur(b,(9,9),0)

            if type != VIDEO:
                trans_x, trans_y, rot, zoom = rw.step()
                if self.translation:
                    paste_center_x, paste_center_y = trans_x, trans_y

                if self.rotation:
                    f = imutils.rotate_bound(f, rot)
                    a = imutils.rotate_bound(a, rot)

                if self.zooming:
                    f = clipped_zoom(f, zoom)
                    a = clipped_zoom(a, zoom)

            f_height = f.shape[0]
            f_width = f.shape[1]
            a_height = a.shape[0]
            a_width = a.shape[1]

            if f_width != a_width or f_height != a_height:
                print("f_width != a_width or f_height != a_height")
                print("f_path:", f_path)
                print("a_path:", a_path)
                print("f.shape:", f.shape)
                print("a.shape:", a.shape)

            assert f_width == a_width
            assert f_height == a_height

            paste_min_x = max(paste_center_x - math.floor(f_height/2), 0)
            paste_max_x = min(paste_center_x + math.floor(f_height/2), b_height)
            paste_min_y = max(paste_center_y - math.floor(f_width/2), 0)
            paste_max_y = min(paste_center_y + math.floor(f_width/2), b_width)

            a_min_x = max(math.floor(f_height/2) - paste_center_x, 0)
            # a_max_x = min(b_height - paste_center_x + math.floor(f_height/2), f_height)
            a_max_x = min(a_min_x + (paste_max_x-paste_min_x), f_height)
            a_min_y = max(math.floor(f_width/2) - paste_center_y, 0)
            # a_max_y = min(b_width - paste_center_y + math.floor(f_width/2), f_width)
            a_max_y = min(a_min_y + (paste_max_y-paste_min_y), f_width)

            # Get ground truth
            gt_a = np.zeros_like(b)
            gt_a[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = a[a_min_x: a_max_x, a_min_y: a_max_y, :]
            # try:
            #     gt_a[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = a[a_min_x: a_max_x, a_min_y: a_max_y, :]
            # except:
            #     print("crop and paste error")
            #     print(f.shape, a.shape, b.shape)
            #     break

            gt_b = b.copy()
            # alpha blending
            a = a / 255.0
            b[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = \
                b[paste_min_x:paste_max_x, paste_min_y:paste_max_y, :] * (1 - a[a_min_x:a_max_x, a_min_y:a_max_y, :]) + \
                f[a_min_x:a_max_x, a_min_y:a_max_y, :] * a[a_min_x:a_max_x, a_min_y:a_max_y, :]
            # try:
            #     b[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = \
            #         b[paste_min_x:paste_max_x, paste_min_y:paste_max_y, :] * (1 - a[a_min_x:a_max_x, a_min_y:a_max_y, :]) + \
            #         f[a_min_x:a_max_x, a_min_y:a_max_y, :] * a[a_min_x:a_max_x, a_min_y:a_max_y, :]
            # except:
            #     print("crop and paste error")
            #     print(f.shape, a.shape, b.shape)
            #     break

            # Get the foreground-only video frame
            gt_f = np.zeros_like(b)
            gt_f[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = f[a_min_x: a_max_x, a_min_y: a_max_y, :] * \
                                                                          a[a_min_x: a_max_x, a_min_y: a_max_y, :]
            # try:
            #     gt_f[paste_min_x: paste_max_x, paste_min_y: paste_max_y, :] = f[a_min_x: a_max_x, a_min_y: a_max_y, :] * a[a_min_x: a_max_x, a_min_y: a_max_y, :]
            # except:
            #     print("crop and paste error")
            #     print(f.shape, a.shape, b.shape)
            #     break

            if self.save_mode == IMAGE:
                img_v_path = os.path.join(".".join(save_path.split(".")[:-1]), "%05d.jpg" % i)
                img_a_path = os.path.join(".".join(save_path_a.split(".")[:-1]), "%05d.jpg" % i)
                img_f_path = os.path.join(".".join(save_path_f.split(".")[:-1]), "%05d.jpg" % i)
                img_b_path = os.path.join(".".join(save_path_b.split(".")[:-1]), "%05d.jpg" % i)
                for p in [img_v_path, img_a_path, img_f_path, img_b_path]:
                    d = ntpath.dirname(p)
                    if not os.path.isdir(d):
                        os.makedirs(d)
                # print(b.shape)
                # print(gt_a.shape)
                # print(gt_f.shape)
                cv2.imwrite(img_v_path, b)
                cv2.imwrite(img_a_path, gt_a)
                cv2.imwrite(img_f_path, gt_f)
                cv2.imwrite(img_b_path, gt_b)
            else:
                out.write(b)
                out_a.write(gt_a)
                out_f.write(gt_f)
                out_b.write(gt_b)

        if type == VIDEO:
            f_cap.release()
            a_cap.release()
        if self.save_mode == VIDEO:
            b_cap.release()
            out.release()
            out_a.release()
            out_f.release()
            out_b.release()
