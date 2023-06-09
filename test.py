import os
import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_images_by_frame_id
from util import html

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch), str(opt.max_size))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()

    num_test = min(len(dataset), opt.ntest) if opt.ntest!=-1 else len(dataset)
    print('total test samples: ', num_test)

    for i, data in enumerate(dataset):
        if i >= num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data, 'test')  # unpack data from data loader
        model.test()           # run inference
        if opt.calc_metric:
            model.calculate_metric()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print('processing sample(%d/%d) %s' % (i, num_test, img_path))
        save_images_by_frame_id(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, \
                    width=opt.display_winsize, scaling=opt.tensor_scaling)

    if opt.calc_metric:
        res, metrics = model.get_metrics()
        for k, v in metrics.items():
            print('%s: %.3f ' % (k, v))

    webpage.save()  # save the HTML
