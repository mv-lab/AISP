import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import math
import sys
import torch.multiprocessing as mp

from util.util import calc_psnr as calc_psnr
from util.AISP_utils import demosaic, postprocess_raw, plot_pair
from util.util import pack_rggb_channels
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import cv2


# from skimage.exposure import match_histograms

def save_rgb(img, filename):
    if np.max(img) <= 1:
        img = img * 255

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, img)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset_train = create_dataset(opt.dataset_name, 'train', opt)
    dataset_size_train = len(dataset_train)
    print('The number of training images = %d' % dataset_size_train)
    dataset_val = create_dataset(opt.dataset_name, 'val', opt)
    dataset_size_val = len(dataset_val)
    print('The number of val images = %d' % dataset_size_val)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = ((model.start_epoch * (dataset_size_train // opt.batch_size)) \
                   // opt.print_freq) * opt.print_freq

    for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
        # training
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()

        iter_data_time = iter_start_time = time.time()
        for i, data in enumerate(dataset_train):
            if total_iters % opt.print_freq == 0:
                t_data = time.time() - iter_data_time
            total_iters += 1  
            epoch_iter += 1  
            model.set_input(data)
            model.optimize_parameters()
            res = model.get_current_visuals()

            if opt.save_imgs:

                psnr_train = calc_psnr(data['raw'], res['data_out'].detach().cpu())
                print(data['fname'][0], psnr_train)
                res = model.get_current_visuals()
                folder_dir = './ckpt/%s/output_train' % (opt.name);
                os.makedirs(folder_dir, exist_ok=True)

                save_dir = '%s/%s.jpg' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0] + '_dslr')
                dslr = res['data_dslr'][0].cpu().permute(1, 2, 0).numpy();
                save_rgb(dslr, save_dir)

                save_dir = '%s/%s.jpg' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0] + '_GCMModel_out_warp_all')
                dslr = res['GCMModel_out_warp'][0].cpu().permute(1, 2, 0).numpy();
                save_rgb(dslr, save_dir)

                save_dir = '%s/%s.jpg' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0] + '_GCMModel_out_warp_test')
                dslr = res['mask_test'][0].cpu().permute(1, 2, 0).numpy();
                save_rgb(dslr, save_dir)


            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data, total_iters)
                iter_start_time = time.time()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d'
                  % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %.3f sec'
              % (epoch, opt.niter + opt.niter_decay,
                 time.time() - epoch_start_time))
        model.update_learning_rate()

        # val
        if opt.calc_metrics:
            model.eval()
            val_iter_time = time.time()
            tqdm_val = tqdm(dataset_val)
            psnr = [0.0] * dataset_size_val
            time_val = 0
            for i, data in enumerate(tqdm_val):
                model.set_input(data)
                time_val_start = time.time()
                with torch.no_grad():
                    model.test()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()
                psnr[i] = calc_psnr(res['data_raw'].detach().cpu(), res['data_out'].detach().cpu())
            visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr))

        sys.stdout.flush()





