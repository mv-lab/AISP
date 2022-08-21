import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import torch
from util.util import torch_save
import math 
import torch.nn.functional as F

def calc_psnr(sr, hr, range=1.):
    # shave = 2
    with torch.no_grad():
        diff = (sr - hr) / range
        mse = torch.pow(diff, 2)
        mse= torch.mean(mse,dim=1,keepdim=True)
        return (-10 * torch.log10(mse))

class BaseModel(ABC):
	def __init__(self, opt):
		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.isTrain = opt.isTrain
		self.scale = opt.scale

		if len(self.gpu_ids) > 0:
			self.device = torch.device('cuda', self.gpu_ids[0])
		else:
			self.device = torch.device('cpu')
		self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
		self.loss_names = []
		self.model_names = []
		self.visual_names = []
		self.optimizers = []
		self.optimizer_names = []
		self.image_paths = []
		self.metric = 0  # used for learning rate policy 'plateau'
		self.start_epoch = 0
				
		self.backwarp_tenGrid = {}
		self.backwarp_tenPartial = {}

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	@abstractmethod
	def set_input(self, input):
		pass

	@abstractmethod
	def forward(self):
		pass

	@abstractmethod
	def optimize_parameters(self):
		pass

	def setup(self, opt=None):
		opt = opt if opt is not None else self.opt
		if self.isTrain:
			self.schedulers = [networks.get_scheduler(optimizer, opt) \
							   for optimizer in self.optimizers]
			for scheduler in self.schedulers:
				scheduler.last_epoch = opt.load_iter
		if opt.load_iter > 0 or opt.load_path != '':
			load_suffix = opt.load_iter
			self.load_networks(load_suffix)
			if opt.load_optimizers:
				self.load_optimizers(opt.load_iter)

		self.print_networks(opt.verbose)

	def eval(self):
		for name in self.model_names:
			net = getattr(self, 'net' + name)
			net.eval()

	def train(self):
		for name in self.model_names:
			net = getattr(self, 'net' + name)
			net.train()

	def test(self):
		with torch.no_grad():
			self.forward()

	def get_image_paths(self):
		return self.image_paths

	def update_learning_rate(self):
		for i, scheduler in enumerate(self.schedulers):
			if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
				scheduler.step(self.metric)
			else:
				scheduler.step()
			print('lr of %s = %.7f' % (
					self.optimizer_names[i], scheduler.get_last_lr()[0]))

	def get_current_visuals(self):
		visual_ret = OrderedDict()
		for name in self.visual_names:
			if 'xy' in name or 'coord' in name:
				visual_ret[name] = getattr(self, name).detach()
			else:
				visual_ret[name] = torch.clamp(
							getattr(self, name).detach(), 0., 1.)
		return visual_ret

	def get_current_losses(self):
		errors_ret = OrderedDict()
		for name in self.loss_names:
			errors_ret[name] = float(getattr(self, 'loss_' + name))
		return errors_ret

	def save_networks(self, epoch):
		for name in self.model_names:
			save_filename = '%s_model_%d.pth' % (name, epoch)
			save_path = os.path.join(self.save_dir, save_filename)
			net = getattr(self, 'net' + name)
			if self.device.type == 'cuda':
				state = {'state_dict': net.module.cpu().state_dict()}
				torch_save(state, save_path)
				net.to(self.device)
			else:
				state = {'state_dict': net.state_dict()}
				torch_save(state, save_path)
		self.save_optimizers(epoch)

	def load_networks(self, epoch):
# 		self.model_names.append('GCMModel')
		for name in self.model_names: #[0:1]:
			# if name is 'Discriminator':
			# 	continue
			load_filename = '%s_model_%d.pth' % (name, epoch)
# 			if name=='GCMModel':
# 				load_filename = '%s_model_%d.pth' % (name, 1)
			if self.opt.load_path != '':
				load_path = self.opt.load_path
			else:
				load_path = os.path.join(self.save_dir, load_filename)
			print(name,load_path)
			net = getattr(self, 'net' + name)
			if isinstance(net, torch.nn.DataParallel):
				net = net.module
			state_dict = torch.load(load_path, map_location=self.device)
			print('loading the model from %s' % (load_path))
			if hasattr(state_dict, '_metadata'):
				del state_dict._metadata

			net_state = net.state_dict()
			is_loaded = {n:False for n in net_state.keys()}
			for name, param in state_dict['state_dict'].items():
				if name in net_state:
					try:
						net_state[name].copy_(param)
						is_loaded[name] = True
					except Exception:
						print('While copying the parameter named [%s], '
							  'whose dimensions in the model are %s and '
							  'whose dimensions in the checkpoint are %s.'
							  % (name, list(net_state[name].shape),
								 list(param.shape)))
						raise RuntimeError
				else:
					print('Saved parameter named [%s] is skipped' % name)
			mark = True
			for name in is_loaded:
				if not is_loaded[name]:
					print('Parameter named [%s] is randomly initialized' % name)
					mark = False
			if mark:
				print('All parameters are initialized using [%s]' % load_path)

			self.start_epoch = epoch

	def save_optimizers(self, epoch):
		assert len(self.optimizers) == len(self.optimizer_names)
		for id, optimizer in enumerate(self.optimizers):
			save_filename = self.optimizer_names[id]
			state = {'name': save_filename,
					 'epoch': epoch,
					 'state_dict': optimizer.state_dict()}
			save_path = os.path.join(self.save_dir, save_filename+'.pth')
			torch_save(state, save_path)

	def load_optimizers(self, epoch):
		assert len(self.optimizers) == len(self.optimizer_names)
		for id, optimizer in enumerate(self.optimizer_names):
			load_filename = self.optimizer_names[id]
			load_path = os.path.join(self.save_dir, load_filename+'.pth')
			print('loading the optimizer from %s' % load_path)
			state_dict = torch.load(load_path)
			print(state_dict['epoch'])
			assert optimizer == state_dict['name']
			assert epoch == state_dict['epoch']
			self.optimizers[id].load_state_dict(state_dict['state_dict'])

	def print_networks(self, verbose):
		print('---------- Networks initialized -------------')
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				num_params = 0
				for param in net.parameters():
					num_params += param.numel()
				if verbose:
					print(net)
				print('[Network %s] Total number of parameters : %.3f M'
					  % (name, num_params / 1e6))
		print('-----------------------------------------------')

	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def estimate(self, tenFirst, tenSecond, net):
		assert(tenFirst.shape[3] == tenSecond.shape[3])
		assert(tenFirst.shape[2] == tenSecond.shape[2])
		intWidth = tenFirst.shape[3]
		intHeight = tenFirst.shape[2]
		# tenPreprocessedFirst = tenFirst.view(1, 3, intHeight, intWidth)
		# tenPreprocessedSecond = tenSecond.view(1, 3, intHeight, intWidth)

		intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
		intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

		tenPreprocessedFirst = F.interpolate(input=tenFirst, 
								size=(intPreprocessedHeight, intPreprocessedWidth), 
								mode='bilinear', align_corners=False)
		tenPreprocessedSecond = F.interpolate(input=tenSecond, 
								size=(intPreprocessedHeight, intPreprocessedWidth), 
								mode='bilinear', align_corners=False)

		tenFlow = 20.0 * F.interpolate(
			             input=net(tenPreprocessedFirst, tenPreprocessedSecond), 
						 size=(intHeight, intWidth), mode='bilinear', align_corners=False)

		tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
		tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

		return tenFlow[:, :, :, :]
	
	def backwarp(self, tenInput, tenFlow):
		index = str(tenFlow.shape) + str(tenInput.device)
		if index not in self.backwarp_tenGrid:
			tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
					 tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
			tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
					 tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
			self.backwarp_tenGrid[index] = torch.cat([tenHor, tenVer], 1).to(tenInput.device)

		if index not in self.backwarp_tenPartial:
			self.backwarp_tenPartial[index] = tenFlow.new_ones([
				 tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

		tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
							 tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
		tenInput = torch.cat([tenInput, self.backwarp_tenPartial[index]], 1)

		tenOutput = F.grid_sample(input=tenInput, 
					grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
					mode='bilinear', padding_mode='zeros', align_corners=False)

		return tenOutput

	def get_backwarp(self, tenFirst, tenSecond,raw, net, flow=None):
		if flow is None:
			flow = self.get_flow(tenFirst, tenSecond, net)
            
		flow_raw = F.interpolate(flow, scale_factor=0.5)/2.
		tenoutput = self.backwarp(tenSecond, flow) 	
		rgb_tenMask = tenoutput[:, -1:, :, :]
		rgb_tenMask[rgb_tenMask > 0.999] = 1.0
		rgb_tenMask[rgb_tenMask < 1.0] = 0.0
        
		rawoutput = self.backwarp(raw, flow_raw) 	
		raw_tenMask = rawoutput[:, -1:, :, :]
		raw_tenMask[raw_tenMask > 0.999] = 1.0
		raw_tenMask[raw_tenMask < 1.0] = 0.0  
		d=tenoutput[:, :-1, :, :] * rgb_tenMask
		return tenoutput[:, :-1, :, :] * rgb_tenMask, rgb_tenMask,rawoutput[:, :-1, :, :] * raw_tenMask, raw_tenMask
	
	def get_backwarp_down(self, tenFirst, tenSecond,raw, net, flow=None):
		if flow is None:
			flow = self.get_flow(tenFirst, tenSecond, net)
            
		tenoutput = self.backwarp(tenSecond, flow) 	
		rgb_tenMask = tenoutput[:, -1:, :, :]
		rgb_tenMask[rgb_tenMask > 0.999] = 1.0
		rgb_tenMask[rgb_tenMask < 1.0] = 0.0
        
		rawoutput = self.backwarp(raw, flow) 	
		raw_tenMask = rawoutput[:, -1:, :, :]
		raw_tenMask[raw_tenMask > 0.999] = 1.0
		raw_tenMask[raw_tenMask < 1.0] = 0.0  
		d=tenoutput[:, :-1, :, :] * rgb_tenMask
		return tenoutput[:, :-1, :, :] * rgb_tenMask, rgb_tenMask,rawoutput[:, :-1, :, :] * raw_tenMask, raw_tenMask
	
	def get_backwarp_nogcm(self, tenFirst, tenSecond,raw, net, flow=None):
		if flow is None:
			flow = self.get_flow(tenFirst, tenSecond, net)
            
		tenoutput = self.backwarp(raw, flow) 	
		rgb_tenMask = tenoutput[:, -1:, :, :]
		rgb_tenMask[rgb_tenMask > 0.999] = 1.0
		rgb_tenMask[rgb_tenMask < 1.0] = 0.0
        
		return tenoutput[:, :-1, :, :] * rgb_tenMask, rgb_tenMask



	def get_backwarp_isp(self, tenFirst, tenSecond, net, flow=None):
		if flow is None:
			flow = self.get_flow(tenFirst, tenSecond, net)
		
		tenoutput = self.backwarp(tenSecond, flow) 	
		tenMask = tenoutput[:, -1:, :, :]
		tenMask[tenMask > 0.999] = 1.0
		tenMask[tenMask < 1.0] = 0.0
		return tenoutput[:, :-1, :, :] * tenMask, tenMask

	def get_backwarp_fb(self, tenFirst, tenSecond,raw, net, flow=None):
        #获取前向flow
		flow_fw = self.get_flow(tenFirst, tenSecond, net)
		
		flow_raw = F.interpolate(flow_fw, scale_factor=0.5)/2.
		tenoutput = self.backwarp(tenSecond, flow_fw) 	
		rgb_tenMask = tenoutput[:, -1:, :, :]
		rgb_tenMask[rgb_tenMask > 0.999] = 1.0
		rgb_tenMask[rgb_tenMask < 1.0] = 0.0
        
		rawoutput = self.backwarp(raw, flow_raw) 
		raw_tenMask = rawoutput[:, -1:, :, :]
		raw_tenMask[raw_tenMask > 0.999] = 1.0
		raw_tenMask[raw_tenMask < 1.0] = 0.0  
        
        #获取后向flow
		flow_bw = self.get_flow( tenFirst,tenoutput, net)
		flow_fw_warped = self.backwarp(flow_fw, flow_fw)
		flow_diff_fw = flow_bw + flow_fw_warped[:,:-1,:,:]
		mag_sq_bw = self.length_sq(flow_bw) + self.length_sq(flow_fw_warped[:,:-1,:,:]) 
		occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
		rgb_flowMask = flow_bw_warped[:, -1:, :, :]
		rgb_flowMask[self.length_sq(flow_diff_bw) <= occ_thresh_bw] = 1.0
		rgb_flowMask[self.length_sq(flow_diff_bw) > occ_thresh_bw] = 0.0
        
		return tenoutput[:, :-1, :, :] * rgb_tenMask * rgb_flowMask, rgb_tenMask,rawoutput[:, :-1, :, :] * raw_tenMask, raw_tenMask,flow_raw,flow_bw

	def get_backwarp_all(self, tenFirst, tenSecond,raw, net, flow=None):
		if flow is None:
			flow_fw = self.get_flow(tenFirst, tenSecond, net)
            
		flow_raw = F.interpolate(flow_fw, scale_factor=0.5)/2.
		tenoutput = self.backwarp(tenSecond, flow_fw) 	
		rgb_tenMask = tenoutput[:, -1:, :, :]
		rgb_tenMask[rgb_tenMask > 0.999] = 1.0
		rgb_tenMask[rgb_tenMask < 1.0] = 0.0
        
		rawoutput = self.backwarp(raw, flow_raw) 	
		raw_tenMask = rawoutput[:, -1:, :, :]
		raw_tenMask[raw_tenMask > 0.999] = 1.0
		raw_tenMask[raw_tenMask < 1.0] = 0.0  
        
		flow_bw = self.get_flow( tenSecond,tenFirst, net)
		tenSecond_wrap = self.backwarp(tenoutput[:, :-1, :, :], flow_bw)  
		rgb_mask_consis = tenSecond_wrap[:, -1:, :, :]

		wrap_psnr=calc_psnr(tenSecond_wrap[:, :-1, :, :],tenSecond)
		rgb_mask_consis[wrap_psnr<=30]=0.0
		rgb_mask_consis[wrap_psnr>30]=1.0
		raw_mask_consis=rgb_mask_consis[:,:,0::2,0::2]
        
		return tenoutput[:, :-1, :, :] * rgb_tenMask * rgb_mask_consis, rgb_tenMask* rgb_mask_consis,rawoutput[:, :-1, :, :] * raw_tenMask* raw_mask_consis, raw_tenMask* raw_mask_consis
	
	def get_flow(self, tenFirst, tenSecond, net):
		with torch.no_grad():
			net.eval()
			flow = self.estimate(tenFirst, tenSecond, net) 
		return flow