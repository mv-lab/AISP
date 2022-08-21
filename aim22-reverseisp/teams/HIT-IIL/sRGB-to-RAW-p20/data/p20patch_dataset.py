import numpy as np
import os
from data.base_dataset import BaseDataset
from util.util import augment, remove_black_level
from util.util import extract_bayer_channels, get_raw_demosaic, load_img
import glob
import random


# Zurich RAW to RGB (ZRR) dataset
class P20patchDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='ZRR'):
		super(P20patchDataset, self).__init__(opt, split, dataset_name)


		self.batch_size = opt.batch_size

		if split == 'train':
			self.root_dir = os.path.join(self.root,'train_full');
			self.train_raws = sorted(glob.glob(os.path.join( self.root_dir, '*.npy')))
			self.train_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.train_raws_file=[]
			self.train_rgbs_file=[]
			self.patch = opt.patch_size
			for seq_path in self.train_raws:
				seq =  np.load(seq_path, encoding='bytes', allow_pickle=True);            
				self.train_raws_file.append(seq)
                
			for seq_path in self.train_rgbs:
				seq = load_img(seq_path)                
				self.train_rgbs_file.append(seq)
                
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_train
			self.len_data = len(self.names)*48

		elif split == 'val':
			self.root_dir = os.path.join(self.root, 'test_full')
			self.patch = opt.patch_size
			self.test_raws = sorted(glob.glob(os.path.join( self.root_dir, '*.npy')))
			self.test_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_val
			self.len_data = len(self.names)

		elif split == 'test':
			self.root_dir = os.path.join(self.root)
			self.test_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_test
			self.len_data = len(self.names)

		else:
			raise ValueError

		


	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		idx = idx % (self.len_data//48)
		H,W,C = self.train_raws_file[idx].shape      
		crop_h = random.randrange(0,H - self.patch ,2)
		crop_w = random.randrange(0,W - self.patch ,2)
		raw = self.train_raws_file[idx][crop_h:crop_h+self.patch , crop_w:crop_w+self.patch,:]
		dslr_image = self.train_rgbs_file[idx][2*crop_h:2*crop_h+2*self.patch, 2*crop_w:2*crop_w+2*self.patch,:]
		raw_combined, raw_demosaic = self._process_raw(raw)
		dslr_image  = dslr_image.transpose((2, 0, 1))
		raw_combined, raw_demosaic, dslr_image = augment(
			raw_combined, raw_demosaic, dslr_image)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'fname': self.names[idx]}

	def _getitem_val(self, idx):  
		raw_init = np.load(self.test_raws[idx], encoding='bytes', allow_pickle=True);
		raw = raw_init[0:0+self.patch , 0:0+self.patch,:]
		raw_combined, raw_demosaic = self._process_raw(raw)
		dslr_image_init = load_img(self.test_rgbs[idx])
		dslr_image = dslr_image_init[0:0+2*self.patch , 0:0 + 2*self.patch,:]
		dslr_image  = dslr_image.transpose((2, 0, 1))

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		dslr_image = load_img(self.test_rgbs[idx])       
		dslr_image  = dslr_image.transpose((2, 0, 1))

		return {
				'dslr': dslr_image,
				'fname': self.names[idx]}


	def _process_raw(self, raw):
		raw = remove_black_level(raw,white_lv=4*255)
		raw_combined = extract_bayer_channels(raw)
		raw_demosaic = get_raw_demosaic(raw)
		return raw_combined, raw_demosaic



if __name__ == '__main__':
	pass


