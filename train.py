import sys
import time
import random
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def train_one_epoch(opt, model, dataset, visualizer, epoch, total_iters):
    epoch_start_time = time.time()  
    iter_data_time = time.time()    
    epoch_iter = 0 

    for i, data in enumerate(dataset):  
        iter_start_time = time.time()         
        t_data = iter_start_time - iter_data_time

        visualizer.reset()
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data, 'train') 
        model.optimize_parameters()

        if opt.lr_policy == 'cyclic':
            lr = model.update_learning_rate(total_iters)  

        # display images on visdom and save images to a HTML file
        if total_iters % opt.display_freq == 0:   
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # print training losses and save logging information to the disk
        if total_iters % opt.print_freq == 0:    
            lr = model.optimizers[0].param_groups[0]['lr']
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, lr, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)

        # cache our latest model every <save_latest_freq> iterations
        if total_iters % opt.save_latest_freq == 0:   
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()

    # cache our model every <save_epoch_freq> epochs
    if epoch % opt.save_epoch_freq == 0:              
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    # update learning rates at the end of every epoch.
    if opt.lr_policy != 'cyclic':
        if opt.continue_train:
            lr = opt.lr * (1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1))
            model.update_learning_rate(lr=lr)
        else:
            model.update_learning_rate()                  

    print('End of epoch %d/%d \t Time Taken: %d sec' % (
        epoch, opt.niter+opt.niter_decay, time.time()-epoch_start_time))

    return total_iters


def eval_one_epoch(opt, model, dataset, visualizer, epoch):
    model.eval()
    model.clear()
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):  
        model.set_input(data, 'val') 
        model.test()
        model.calculate_metric()
    res, metrics = model.get_metrics()
    visualizer.print_current_metrics(epoch, metrics)
    print('End of validation on epoch %d\t Time Taken: %d sec' % (epoch, time.time()-epoch_start_time))
    return res



if __name__ == '__main__':
    best = 1e8

    opt = TrainOptions().parse()

    # configure dataset
    opt.data_phase = 'train'
    train_dataset = create_dataset(opt)
    print('The number of training images = %d' % len(train_dataset))
    opt.data_phase = 'val'
    val_dataset = create_dataset(opt)
    print('The number of testing images = %d' % len(val_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    start_epoch = model.get_latest_epoch() if opt.continue_train else opt.epoch_count
    print('Start epoch:', start_epoch)

    if opt.continue_train:
        lr = opt.lr * (1.0 - max(0, start_epoch - 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1))
        model.update_learning_rate(lr=lr)

    # we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):    
        # train one epoch
        total_iters = train_one_epoch(opt, model, train_dataset, visualizer, epoch, total_iters)

        # validate model
        if (epoch>=opt.validate_start and epoch % opt.validate_freq == 0):
            res = eval_one_epoch(opt, model, val_dataset, visualizer, epoch)
            if res < best:
                model.save_networks('best')
                best = res

    message = "Best: res = %.3f" % best
    visualizer.log_message(message)
