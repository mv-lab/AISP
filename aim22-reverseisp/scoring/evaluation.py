#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_ssim as ssim
from myssim import compare_ssim as ssim


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    if mse == 0:
        return 999
    psnr = 10 * np.log10(1.0 / mse)
    return psnr
    
def _open_img(img_p):
    return np.load(img_p, allow_pickle=True).astype(float) / 1024.

def compute_psnr(ref_im, res_im):
    return output_psnr_mse(ref_im, res_im)

def compute_mssim(ref_im, res_im):
    ref_img = ref_im
    res_img = res_im
    channels = []
    for i in range(4):
        channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
        gaussian_weights=True, use_sample_covariance=False))
    return np.mean(channels)


# as per the metadata file, input and output directories are the arguments
#[_, input_dir, output_dir] = sys.argv

input_dir = sys.argv[1]
output_dir = sys.argv[2]

try:
    res_dir = glob(input_dir + '/res/*/')[0]
except:
    res_dir = os.path.join(input_dir, 'res/submission/')

ref_dir = os.path.join(input_dir, 'ref/val_gt/')

runtime = -1
cpu = -1
data = -1
other = ""
readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
try:
    readme_fname = readme_fnames[0]
    print("Parsing extra information from %s"%readme_fname)
    with open(os.path.join(res_dir, readme_fname)) as readme_file:
        readme = readme_file.readlines()
        lines = [l.strip() for l in readme if l.find(":")>=0]
        runtime = float(":".join(lines[0].split(":")[1:]))
        cpu = int(":".join(lines[1].split(":")[1:]))
        data = int(":".join(lines[2].split(":")[1:]))
        other = ":".join(lines[3].split(":")[1:])
except:
    print("Error occured while parsing readme.txt")
    print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")

print("Parsed information:")
print("Runtime: %f"%runtime)
print("CPU/GPU: %d"%cpu)
print("Data: %d"%data)
print("Other: %s"%other)


ref_pngs = sorted([p for p in os.listdir(ref_dir + '/') if 'npy' in p])
res_pngs = sorted([p for p in os.listdir(res_dir + '/') if 'npy' in p])
if not (len(ref_pngs)==len(res_pngs)):
    raise Exception('Expected %d .png images'%len(ref_pngs))

print ('Files to process: ', len(ref_pngs))

scores = []
scores_ssim = []
for (fref_im, fres_im) in zip(ref_pngs, res_pngs):
    
    ref_im = _open_img(os.path.join(ref_dir,fref_im))
    res_im = _open_img(os.path.join(res_dir,fres_im))

    assert ref_im.shape[-1] == 4
    assert res_im.shape[-1] == 4
    assert (np.min(ref_im) >= 0) and (np.max(ref_im) <= 1)
    assert (np.min(res_im) >= 0) and (np.max(res_im) <= 1)

    _psnr = compute_psnr(ref_im,res_im)
    scores.append(_psnr)
    _ssim = compute_mssim(ref_im, res_im)
    scores_ssim.append(_ssim)
    #print ('Comparing: ', os.path.join(ref_dir,fref_im), os.path.join(res_dir,fres_im))
    #print (ref_im.shape, res_im.shape, ref_im.min(), ref_im.max(), res_im.min(), res_im.max(), _psnr, _ssim)

psnr = np.mean(scores)
mssim = np.mean(scores_ssim)
if (len(scores) != len(ref_pngs)) or (len(scores_ssim) != len(ref_pngs)):
    psnr=0
    mssim=0


# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("PSNR:%f\n"%psnr)
    output_file.write("SSIM:%f\n"%mssim)
    output_file.write("ExtraRuntime:%f\n"%runtime)
    output_file.write("ExtraPlatform:%d\n"%cpu)
    output_file.write("ExtraData:%d\n"%data)

output_file.close()
print ('Done!')
