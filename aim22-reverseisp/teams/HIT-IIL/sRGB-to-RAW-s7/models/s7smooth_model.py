import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
from pwc import pwc_net
from util.util import get_coord
import numpy as np



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

class S7smoothModel(BaseModel):
	staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(S7smoothModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = ['GCMModel_L1', 'LiteISPNet_L1', 'LiteISPNet_SSIM','LiteISPNet_VGG', 'Total']
        
		if self.isTrain:
			self.visual_names = [ 'data_out','data_raw_demosaic','data_dslr','GCMModel_out','GCMModel_out_warp','rgb_mask','raw_warp', 'raw_mask','data_raw','data_dslr_mask','data_out_mask'] 
		else:
			self.visual_names = [ 'data_out','data_dslr']
            
		self.model_names = ['LiteISPNet', 'GCMModel']
		self.optimizer_names = ['LiteISPNet_optimizer_%s' % opt.optimizer,
								'GCMModel_optimizer_%s' % opt.optimizer]

		isp = LiteISPNet(opt)
		self.netLiteISPNet= N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

		gcm = GCMModel(opt)
		self.netGCMModel = N.init_net(gcm, opt.init_type, opt.init_gain, opt.gpu_ids)

		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)

		if self.isTrain:		
			self.optimizer_LiteISPNet = optim.AdamW(self.netLiteISPNet.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
			self.optimizer_GCMModel = optim.AdamW(self.netGCMModel.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
			self.optimizers = [self.optimizer_LiteISPNet, self.optimizer_GCMModel]

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

		self.data_out = self.netLiteISPNet(self.data_dslr)
		
		if self.isTrain:
			self.data_dslr_mask = self.data_dslr * self.rgb_mask
			self.data_out_mask = self.data_out * self.raw_mask
			self.raw_warp_rgb = ISP(self.raw_warp).to(self.device)
			self.data_out_mask_rgb = ISP(self.data_out_mask).to(self.device) 

	def backward(self):
		self.loss_GCMModel_L1 = self.criterionL1(self.GCMModel_out_warp, self.data_dslr_mask).mean()
		self.loss_LiteISPNet_L1 = self.criterionL1(self.data_out_mask, self.raw_warp).mean()
		self.loss_LiteISPNet_SSIM = 1 - self.criterionSSIM(self.data_out_mask, self.raw_warp).mean()
		self.loss_LiteISPNet_VGG = self.criterionVGG(self.data_out_mask_rgb, self.raw_warp_rgb).mean()
		self.loss_Total = self.loss_GCMModel_L1 + self.loss_LiteISPNet_L1+ \
			              self.loss_LiteISPNet_VGG * 0.4 +  self.loss_LiteISPNet_SSIM * 0.1
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_LiteISPNet.zero_grad()
		self.optimizer_GCMModel.zero_grad()
		self.backward()
		self.optimizer_LiteISPNet.step()
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

class LiteISPNet(nn.Module):
	def __init__(self, opt):
		super(LiteISPNet, self).__init__()
		self.opt = opt
		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4

		self.head = N.seq(
			N.conv(3, ch_1, mode='C')
		)  # shape: (N, ch_1, H/2, W/2)



		self.down1 = N.seq(
			N.conv(ch_1, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C'),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward(ch_1)
		)  # shape: (N, ch_1*4, H/8, W/8)

		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward(ch_2)
		)  # shape: (N, ch_2*4, H/16, W/16)

		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # shape: (N, ch_2*4, H/16, W/16)

		self.up3 = N.seq(
			N.DWTInverse(ch_2*4),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/8, W/8)

		self.up2 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # shape: (N, ch_1*4, H/4, W/4)

		self.up1 = N.seq(
			N.DWTInverse(ch_1*4),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)  # shape: (N, ch_1, H/2, W/2)

		self.tail = N.seq(
			N.conv(ch_1, ch_1//4, mode='C'),
			PixelUnshuffle(downscale_factor=2),
			N.conv(ch_1, 4, mode='C')
		)  # shape: (N, 3, H, W)   

	def forward(self, raw, ):
		# input = raw
		raw =torch.clamp(raw,0.,1.)
		raw = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * raw) / 3.0)
		raw = torch.clamp(raw,1e-8 ,1.)
		input = torch.pow(raw, 2.2)
		h = self.head(input)

		
		d1 = self.down1(h)
		d2 = self.down2(d1)
		d3 = self.down3(d2)
		m = self.middle(d3) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)

		return out



