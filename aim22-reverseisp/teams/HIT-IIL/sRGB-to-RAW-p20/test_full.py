import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
from util.util import pack_rggb_channels
from os.path import join
from tensorboardX import SummaryWriter
import cv2
from util.util import pack_rggb_channels
import glob

def save_rgb (img, filename):
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(filename, img)

def log(log_file, str, also_print=True):
    with open(log_file, 'a+') as F:
        F.write(str)
    if also_print:
        print(str, end='')
        
def _open_img(img_p,ratio):
    return np.load(img_p, allow_pickle=True).astype(float) / ratio


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        if opt.visual_full_imgs:
            dataset = create_dataset(dataset_name, 'visual', opt)
        else:
            dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)
        


    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()


            for i, data in enumerate(tqdm_val):
                         
                model.set_input(data)
                model.test()
                res = model.get_current_visuals()
                recon_raw = res['data_out'][0].detach().permute(1, 2, 0).numpy()
                
                ratio = 1020
                folder_dir = opt.save_path
                PS = opt.test_patch
                os.makedirs(folder_dir, exist_ok=True)
                
                H,W,C= recon_raw.shape
                pic_i=0      
                avg_ps=0
                for rr in np.arange(0, H - PS + 1, PS):
                    for cc in np.arange(0, W - PS + 1, PS):
                        
                        raw_patch = recon_raw[rr:rr + PS, cc:cc + PS,:]
                        pic_index=data['fname'][0].split('/')[-1].split('_')[-1].split('.')[0]+'_'+str(pic_i)+'.npy'
                        raw_patch=pack_rggb_channels(raw_patch)
                        raw_patch = (raw_patch * ratio).astype(np.uint16)
                        save_dir = '%s/%s' % (folder_dir, pic_index)
                        os.makedirs(folder_dir, exist_ok=True)
                        np.save(save_dir, raw_patch)          
                        pic_i+=1

   
    for dataset in datasets:
        datasets[dataset].close()


