

import torch
import torch.nn as nn

from .JPEG_utils import diff_round, quality_to_factor, Quantization
from .compression import compress_jpeg
from .decompression import decompress_jpeg


class DiffJPEG(nn.Module):    
    def __init__(self, differentiable=True, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image height
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
            # rounding = Quantization()
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        # self.decompress = decompress_jpeg(height, width, rounding=rounding,
        #                                   factor=factor)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor)

    def forward(self, x):
        '''
        '''
        org_height = x.shape[2]
        org_width = x.shape[3]
        y, cb, cr = self.compress(x)

        recovered = self.decompress(y, cb, cr, org_height, org_width)
        return recovered


