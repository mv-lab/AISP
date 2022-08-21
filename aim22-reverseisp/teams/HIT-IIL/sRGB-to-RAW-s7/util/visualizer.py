import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from functools import partial
from functools import wraps
import time

def write_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(30):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                print('%s OSError' % str(args))
                time.sleep(1)
        return ret
    return wrapper

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        if opt.isTrain:
            self.name = opt.name
            self.save_dir = join(opt.checkpoints_dir, opt.name, 'log')
            self.writer = SummaryWriter(logdir=join(self.save_dir))
        else:
            self.name = '%s_%s_%d' % (
                opt.name, opt.dataset_name, opt.load_iter)
            self.save_dir = join(opt.checkpoints_dir, opt.name)
            if opt.save_imgs:
                self.writer = SummaryWriter(logdir=join(
                    self.save_dir, 'ckpts', self.name))

    @write_until_success
    def display_current_results(self, phase, visuals, iters):
        for k, v in visuals.items():
            v = v.cpu()
            self.writer.add_image('%s/%s'%(phase, k), v[0]/255, iters)
        self.writer.flush()

    @write_until_success
    def print_current_losses(self, epoch, iters, losses,
                             t_comp, t_data, total_iters):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' \
                  % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4e ' % (k, v)
            self.writer.add_scalar('loss/%s'%k, v, total_iters)
        print(message)
    
    @write_until_success
    def print_psnr(self, epoch, total_epoch, time_val, mean_psnr):
        self.writer.add_scalar('val/psnr', mean_psnr, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t PSNR: %f'
                % (epoch, total_epoch, time_val, mean_psnr))


