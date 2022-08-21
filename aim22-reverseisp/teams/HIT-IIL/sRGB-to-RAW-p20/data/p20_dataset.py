import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.util import augment, remove_black_level, get_coord
from util.util import extract_bayer_channels, get_raw_demosaic, load_img
import glob
# from skimage.exposure import match_histograms


# Zurich RAW to RGB (ZRR) dataset
class P20Dataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='ZRR'):
		super(P20Dataset, self).__init__(opt, split, dataset_name)


		self.batch_size = opt.batch_size

		if split == 'train':
			self.root_dir = os.path.join(self.root,'train');
			self.train_raws = sorted(glob.glob(os.path.join( self.root_dir, '*.npy')))
			self.train_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_train

		elif split == 'val':
			self.root_dir = os.path.join(self.root, 'test')
			self.test_raws = sorted(glob.glob(os.path.join( self.root_dir, '*.npy')))
			self.test_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_val

		elif split == 'test':
			self.root_dir = os.path.join(self.root, 'test')
			self.test_rgbs = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')))
			self.names = sorted(glob.glob(os.path.join( self.root_dir, '*.jpg')));
			self._getitem = self._getitem_test

		else:
			raise ValueError

		self.len_data = len(self.names)


	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
        
		raw = np.load(self.train_raws[idx], encoding='bytes', allow_pickle=True);
		raw_combined, raw_demosaic = self._process_raw(raw)
		dslr_image = load_img(self.train_rgbs[idx])
		dslr_image  = np.ascontiguousarray(dslr_image.transpose((2, 0, 1)))
		
		raw_combined, raw_demosaic, dslr_image = augment(
			raw_combined, raw_demosaic, dslr_image)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'fname': self.names[idx]}

	def _getitem_val(self, idx):  
		raw = np.load(self.test_raws[idx], encoding='bytes', allow_pickle=True);
		raw_combined, raw_demosaic = self._process_raw(raw)
		dslr_image = load_img(self.test_rgbs[idx])
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

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	# Due to the memory (32 GB) limitation, here we only preload the raw images. 
	# If you have enough memory, you can also modify the code to preload the sRGB images to speed up the training process.
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = obj.raw_imio.read(os.path.join(obj.raw_dir, obj.names[i] + '.png'))
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])


if __name__ == '__main__':
	pass
