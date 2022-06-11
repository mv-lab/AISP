import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
from glob import glob
from utils import load_rawpy, load_img, load_raw, demosaic, mosaic, postprocess_raw, save_rgb, plot_pair

inputs = sorted(glob('data-s7/val_rgb/*'))

for fimg in inputs:
    img = load_img(fimg, norm=True)
    rgb2raw = mosaic(img)
    assert (252, 252, 4) == rgb2raw.shape
    
    fimg = fimg.split('/')[-1].replace('.jpg','')
    rgb2raw = (rgb2raw * 1024).astype(np.uint16)
    np.save(f'data-s7/sample_submission/{fimg}.npy', rgb2raw)
