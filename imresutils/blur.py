import numpy as np
from matplotlib import pyplot as plt
import random
import torch
from scipy import signal
import kornia


def augment_kernel(kernel, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    Rotate kernels (or images)
    '''
    if mode == 0:
        return kernel
    elif mode == 1:
        return np.flipud(np.rot90(kernel))
    elif mode == 2:
        return np.flipud(kernel)
    elif mode == 3:
        return np.rot90(kernel, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(kernel, k=2))
    elif mode == 5:
        return np.rot90(kernel)
    elif mode == 6:
        return np.rot90(kernel, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(kernel, k=3))
    
def apply_custom_filter(inp_img, kernel_):
    return kornia.filters.filter2d(inp_img, kernel_, normalized=True)

def generate_gkernel(ker_sz=None, sigma=None):
    gkern1 = signal.gaussian(ker_sz, std=sigma[0]).reshape(ker_sz, 1)
    gkern2 = signal.gaussian(ker_sz, std=sigma[1]).reshape(ker_sz, 1)
    gkern  = np.outer(gkern1, gkern2)
    return gkern
    
def apply_gkernel(inp_img, ker_sz=5, ksigma_vals=[.05 + i for i in range(5)]):
    """
    Apply uniform gaussian kernel of sizes between 5 and 11.
    """
    # sample for variance
    sigma_val1 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma_val2 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma = (sigma_val1, sigma_val2)
    
    kernel = generate_gkernel(ker_sz, sigma)
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel
    
def apply_psf(inp_img, kernels):
    """
    Apply PSF from a pool. See kernels.npy
    """
    idx = np.random.choice(np.arange(11), p=[0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.091, 0.09])
    kernel = kernels[idx].astype(np.float64)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = 25
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img.unsqueeze(0), tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.)


if __name__ == "__main__":
    pass