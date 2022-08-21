import torch
from .base_model import BaseModel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
from util.util import get_coord
import numpy as np
from models.arch_util import LayerNorm2d
from models.local_arch import Local_Base
from . import net as N


def demosaic (raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns: 
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """
    
    assert raw.shape[1] == 4
    shape = raw.shape
    
    blue       = raw[:,0:1,:,:]
    green_red  = raw[:,1:2,:,:]
    red        = raw[:,2:3,:,:]
    green_blue = raw[:,3:,:,:]
    avg_green  = (green_red + green_blue) / 2
    image      = torch.cat((red, avg_green, blue), dim=1)
    image      = F.interpolate(input=image, size=(shape[2]*2, shape[3]*2), 
                               mode='bilinear', align_corners=True)
    return image

def gamma_compression(image):
    """Converts from linear to gamma space."""
    return torch.clamp(image, 1e-8, 1.0) ** (1.0 / 2.2)

def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3*(image**2)) - (2*(image**3))

def ISP(raw):
    raw = demosaic(raw)
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = torch.clamp(raw, 0.0, 1.0)
    return raw


def pixel_unshuffle(input, downscale_factor):
	'''
	input: batchSize * c * k*w * k*h
	kdownscale_factor: k
	batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
	'''
	c = input.shape[1]

	kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
				1, downscale_factor, downscale_factor],
				device=input.device)
	for y in range(downscale_factor):
		for x in range(downscale_factor):
			kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
	return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
	def __init__(self, downscale_factor):
		super(PixelUnshuffle, self).__init__()
		self.downscale_factor = downscale_factor
	def forward(self, input):
		'''
		input: batchSize * c * k*w * k*h
		kdownscale_factor: k
		batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
		'''

		return pixel_unshuffle(input, self.downscale_factor)

def inverse_gamma(image):
    """Converts from linear to gamma space."""
    return torch.clamp(image, 1e-8, 1.0) ** (2.2)

def inverse_tonemap(image):
	image =torch.clamp(image,0.,1.)
	image = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
	return image

class nafModel(BaseModel):
	staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(nafModel, self).__init__(opt)

		self.opt = opt
			
		self.loss_names = ['GCMModel_L1', 'NAFISPNet_L1', 'NAFISPNet_SSIM','NAFISPNet_VGG', 'Total']
        
		if self.isTrain:
			self.visual_names = [ 'data_out','data_raw_demosaic','data_dslr','GCMModel_out','GCMModel_out_warp','rgb_mask','raw_warp', 'raw_mask','data_raw','data_dslr_mask','data_out_mask'] 
		else:
			self.visual_names = [ 'data_out','data_dslr']
            
		self.model_names = ['NAFISPNet', 'GCMModel']
		self.optimizer_names = ['NAFISPNet_optimizer_%s' % opt.optimizer,
								'GCMModel_optimizer_%s' % opt.optimizer]

		isp = NAFISPNet(opt)
		self.netNAFISPNet= N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

		self.pool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

		gcm = GCMModel(opt)
		self.netGCMModel = N.init_net(gcm, opt.init_type, opt.init_gain, opt.gpu_ids)

		if self.isTrain:
			
			from pwc import pwc_net
			pwcnet = pwc_net.PWCNET()
			self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
			self.set_requires_grad(self.netPWCNET, requires_grad=False)
		

		if self.isTrain:		
			self.optimizer_NAFISPNet = optim.AdamW(self.netNAFISPNet.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)
			self.optimizer_GCMModel = optim.AdamW(self.netGCMModel.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.weight_decay)
			self.optimizers = [self.optimizer_NAFISPNet, self.optimizer_GCMModel]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)


	def set_input(self, input):
		if self.isTrain:
			self.data_raw = input['raw'].to(self.device)
			self.data_raw_demosaic = input['raw_demosaic'].to(self.device)
		self.data_dslr = input['dslr'].to(self.device)
		self.image_paths = input['fname']

	def forward(self):
		if self.isTrain:
			self.GCMModel_out = self.netGCMModel(self.data_raw_demosaic, self.data_dslr)
			self.GCMModel_out_warp, self.rgb_mask,self.raw_warp, self.raw_mask = \
				self.get_backwarp( self.data_dslr,self.GCMModel_out,self.data_raw, self.netPWCNET)
		
		self.data_dslr_pool=self.pool(self.data_dslr)
		self.data_out = self.netNAFISPNet(self.data_dslr_pool)
		
		
		if self.isTrain:
			self.data_dslr_mask = self.data_dslr * self.rgb_mask;
			self.data_out_mask = self.data_out * self.raw_mask
			self.raw_warp_rgb = ISP(self.raw_warp).to(self.device)
			self.data_out_mask_rgb = ISP(self.data_out_mask).to(self.device) 

	def backward(self):
		self.loss_GCMModel_L1 = self.criterionL1(self.GCMModel_out_warp, self.data_dslr_mask).mean()
		self.loss_NAFISPNet_L1 = self.criterionL1(self.data_out_mask, self.raw_warp).mean()
		self.loss_NAFISPNet_SSIM = 1 - self.criterionSSIM(self.data_out_mask, self.raw_warp).mean()
		self.loss_NAFISPNet_VGG = self.criterionVGG(self.data_out_mask_rgb, self.raw_warp_rgb).mean()
		self.loss_Total = self.loss_GCMModel_L1 + self.loss_NAFISPNet_L1+ \
			              self.loss_NAFISPNet_VGG * 0.4 +  self.loss_NAFISPNet_SSIM * 0.3
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_NAFISPNet.zero_grad()
		self.optimizer_GCMModel.zero_grad()
		self.backward()
		self.optimizer_NAFISPNet.step()
		self.optimizer_GCMModel.step()

class GCMModel(nn.Module):
	def __init__(self, opt):
		super(GCMModel, self).__init__()
		self.opt = opt
		self.ch_1 = 32
		self.ch_2 = 64

		guide_input_channels = 6
		align_input_channels = 3
		
		self.guide_net = N.seq(
			N.conv(guide_input_channels, self.ch_1, 7, stride=2, padding=0, mode='CR'),
			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRC'),
			nn.AdaptiveAvgPool2d(1),
			N.conv(self.ch_1, self.ch_2, 1, stride=1, padding=0, mode='C')
		)

		self.align_head = N.conv(align_input_channels, self.ch_2, 1, padding=0, mode='CR')

		self.align_base = N.seq(
			N.conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0, mode='CRCRCR')
		)
		self.align_tail = N.seq(
			N.conv(self.ch_2, 3, 1, padding=0, mode='C')
		)

	def forward(self, demosaic_raw, dslr):
		demosaic_raw = torch.pow(demosaic_raw, 1/2.2)

		guide_input = torch.cat((demosaic_raw, dslr), 1)
		base_input = demosaic_raw

		guide = self.guide_net(guide_input)
	
		out = self.align_head(base_input)
		out = guide * out + out
		out = self.align_base(out)
		out = self.align_tail(out) + demosaic_raw
		
		return out



class NAFISPNet(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro_1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Sequential(
			#nn.Conv2d(in_channels=width, out_channels=width//4, kernel_size=3, padding=1, stride=1, groups=1,
                        #      bias=True),
			#PixelUnshuffle(downscale_factor=2),
			nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[N.NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[N.NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[N.NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.skip_conv = nn.Conv2d(width, width, kernel_size=1, bias=True)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp = inverse_gamma(inverse_tonemap(inp))
        x_input = self.intro(inp)
        x = self.intro_1(x_input)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # print(x.shape,self.skip_conv(x_input).shape)
        x = x + self.skip_conv(x_input)
        x = self.ending(x)
        # x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


