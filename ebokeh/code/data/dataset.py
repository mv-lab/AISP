import glob
import math
import os

import numpy as np
import os.path as osp

from typing import Callable, Optional
from PIL import Image
from skimage import img_as_float
from torch import tensor, float32, linspace, meshgrid, FloatTensor, cat
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, ToTensor, CenterCrop
from torchvision.transforms.functional import crop, resize, hflip, vflip, rotate
from random import random, choice
from numpy.random import randint

from code.utils.dataloading_utils import get_divisible_padding, pad_to_divisible_image, center_pad


class BokehTransformDataset(Dataset):
    def __init__(self, data_dir: str,
                 max_resolution=None, include_alpha=True, include_pos_map=False, include_lens_factor=False,
                 train=False, validation=False, test=False, predict=False,
                 samples_train=None, samples_val=None, samples_test=None, samples_pred=None,
                 augs=False, crop=False, crop_size=512, check_alpha_crop=False, check_alpha_chance=1.,
                 min_alpha_threshold=0.025, max_alpha_threshold=0.4, edge_bias=False, resolution_fix=False,
                 divisible_pad=False, input_crop=None,
                 metadata=None, debug=False, debug_dir="./debug", transform: Optional[Callable] = ToTensor()):

        self._data_dir = data_dir
        self._augs = augs
        self._max_resolution = max_resolution
        self._include_alpha = include_alpha
        self._include_pos_map = include_pos_map
        self._include_lens_factor = include_lens_factor

        self._train = train
        self._validation = validation
        self._test = test
        self._predict = predict

        self._samples_train = samples_train
        self._samples_val = samples_val
        self._samples_test = samples_test
        self._samples_pred = samples_pred

        self._crop = crop
        self._crop_size = (crop_size, crop_size)
        self._check_alpha_crop = check_alpha_crop
        self._check_alpha_chance = check_alpha_chance
        self._min_alpha_threshold = min_alpha_threshold
        self._max_alpha_threshold = max_alpha_threshold
        self._edge_bias = edge_bias

        self._train = train
        self._validation = validation
        self._test = test

        self._debug = debug
        self._debug_dir = debug_dir
        self._transform = transform

        self._metadata = metadata
        self._resolution_fix = resolution_fix
        self._divisible_pad = divisible_pad
        self._input_crop = input_crop

        # sanity checks

        if self._train + self._validation + self._test + self._predict != 1:
            raise ValueError(f"Exactly one of train, validation, test or predict must be True!")

        if self._samples_train and self._samples_train > 20000:
            raise ValueError(f"max samples_train is 20000, current value is {self._samples_train}")

        if self._samples_val and self._samples_val > 500:
            raise ValueError(f"max samples_val is 500, current value is {self._samples_val}")

        if self._samples_test and self._samples_test > 500:
            raise ValueError(f"max samples_test is 500, current value is {self._samples_test}")

        # load data

        if self._train:
            self._source_paths = sorted(glob.glob(osp.join(self._data_dir, "*.src.jpg")))[:self._samples_train]
            self._target_paths = sorted(glob.glob(osp.join(self._data_dir, "*.tgt.jpg")))[:self._samples_train]
            self._alpha_paths = sorted(glob.glob(osp.join(self._data_dir, "*.alpha.png")))[:self._samples_train]
        elif self._validation:
            self._source_paths = sorted(glob.glob(osp.join(self._data_dir, "*.src.jpg")))[:self._samples_val]
            self._target_paths = sorted(glob.glob(osp.join(self._data_dir, "*.tgt.jpg")))[:self._samples_val]
            self._alpha_paths = sorted(glob.glob(osp.join(self._data_dir, "*.alpha.png")))[:self._samples_val]
        elif self._test:
            self._source_paths = sorted(glob.glob(osp.join(self._data_dir, "*.src.jpg")))[:self._samples_test]
            self._target_paths = sorted(glob.glob(osp.join(self._data_dir, "*.tgt.jpg")))[:self._samples_test]
            self._alpha_paths = sorted(glob.glob(osp.join(self._data_dir, "*.alpha.png")))[:self._samples_test]
        elif self._predict:
            source_paths = sorted(glob.glob(osp.join(self._data_dir, "*.src.jpg")))[:self._samples_test]
            self._source_paths = source_paths[:self._samples_pred] if self._samples_pred else source_paths
        else:
            raise ValueError("Must specify train, validation or test mode.")

        if self._metadata is None:
            if self._train:
                self._meta_data = read_meta_data(osp.join(self._data_dir, "meta.txt"),
                                                 max_id=self._samples_train if self._samples_train
                                                 else len(self._source_paths))
            elif self._validation:
                self._meta_data = read_meta_data(osp.join(self._data_dir, "meta.txt"),
                                                 max_id=self._samples_val if self._samples_val else
                                                 len(self._source_paths))
            elif self._test:
                self._meta_data = read_meta_data(osp.join(self._data_dir, "meta.txt"),
                                                 max_id=
                                                 self._samples_test if self._samples_test else len(self._source_paths)
                                                 )
            elif self._predict:
                self._meta_data = read_meta_data(osp.join(self._data_dir, "meta.txt"),
                                                 max_id=self._samples_pred if self._samples_pred
                                                 else len(self._source_paths))

        # validate data
        if not self._predict:
            file_counts = [
                len(self._source_paths),
                len(self._target_paths),
                # len(self._alpha_paths) if self._train else len(self._target_paths)
            ]
            if not file_counts[0] or len(set(file_counts)) != 1:
                idx_source = set(path.split('/')[-1].split('.')[0] for path in self._source_paths)
                idx_targer = set(path.split('/')[-1].split('.')[0] for path in self._target_paths)
                raise ValueError(
                    f"Empty or non-matching number of files in root dir: {file_counts}. "
                    "Expected an equal number of source, target and target-alpha files. "
                    "Also expecting matching meta file entries."
                    f"Mismatched ids: {idx_source - idx_targer}"
                )

    def __len__(self):
        return len(self._source_paths)

    def __getitem__(self, index):
        source = Image.open(self._source_paths[index])
        target = Image.open(self._target_paths[index]) if not self._predict else None
        alpha = Image.open(self._alpha_paths[index]) if self._train and self._metadata is None else None

        input_resolution = source.size

        if self._input_crop is not None:
            center_crop = CenterCrop(self._input_crop)
            source = center_crop(center_pad(source, self._input_crop))
            target = center_crop(center_pad(target, self._input_crop)) if target is not None else None
            alpha = center_crop(center_pad(alpha, self._input_crop)) if alpha is not None else None

        if self._divisible_pad:
            source = pad_to_divisible_image(source, 16)
            target = pad_to_divisible_image(target, 16) if target is not None else None
            alpha = pad_to_divisible_image(alpha, 16) if alpha is not None else None

        filename = osp.basename(self._source_paths[index])
        image_id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = self._meta_data[image_id] if self._metadata is None else (None, None, None)

        if self._resolution_fix:
            padding = get_divisible_padding(*source.size, 16) if self._resolution_fix else (0, 0, 0, 0)
            padding = (padding[2] + padding[3], padding[0] + padding[1])
            pos_map = get_pos_map(*(source.size[0]+padding[0], source.size[1]+padding[1])) if self._include_pos_map \
                else None
        else:
            pos_map = get_pos_map(*source.size) if self._include_pos_map else None

        images = {'source': source, 'target': target, 'alpha': alpha,
                  'pos_x': pos_map[0].unsqueeze(0) if self._include_pos_map else None,
                  'pos_y': pos_map[1].unsqueeze(0) if self._include_pos_map else None}

        if self._crop and self._train:
            self.random_pos_crop(images)

        if self._max_resolution:
            self.resize(images)

        if self._augs and self._train:
            augmentations(images)

        bokeh_strength = calculate_bokeh_strength(src_lens, tgt_lens, disparity) if self._metadata is None \
            else tensor(self._metadata['bokeh_strength'], dtype=float32)

        lens_factor = get_lens_factor(tgt_lens) if self._metadata is None else \
            tensor(self._metadata["lens_factor"], dtype=float32)

        if self._debug:
            os.makedirs(self._debug_dir, exist_ok=True)
            for key in images:
                images[key].save(osp.join(self._debug_dir, f"{image_id}-{key}.jpg")) if images[key] else None

        if self._predict:
            images['target'] = Image.new('RGB', size=(1, 1), color=(0, 0, 0))

        return {
            "source": self._transform(images['source']) if self._transform else images['source'],
            "target": self._transform(images['target']) if self._transform else images['target'],
            "bokeh_strength": bokeh_strength,
            "lens_factor": FloatTensor(lens_factor) if self._include_lens_factor else tensor(0.),
            "pos_map": cat(
                (images['pos_x'], images['pos_y']), dim=0) if self._include_pos_map else tensor(0.),
            "image_id": [image_id],
            "resolution": [input_resolution]
        }

    def random_pos_crop(self, images: dict):  # source, target, alpha):
        w, h = images['source'].size
        top = randint(0, h - self._crop_size[0]) if 0 != h - self._crop_size[0] else 0
        left = randint(0, w - self._crop_size[1]) if 0 != h - self._crop_size[0] else 0

        if self._edge_bias:
            if random() < 0.05:
                top = choice([0, h - self._crop_size[0]])

            if random() < 0.05:
                left = choice([0, w - self._crop_size[1]])

        if self._check_alpha_crop:
            alpha_crop = crop(images['alpha'], top, left, self._crop_size[0], self._crop_size[1])
            min_alpha_threshold_tmp = self._min_alpha_threshold if random() < self._check_alpha_chance else 0.
            while not min_alpha_threshold_tmp < np.mean(img_as_float(alpha_crop)) < self._max_alpha_threshold:
                top = randint(0, h - self._crop_size[0])
                left = randint(0, w - self._crop_size[1])
                alpha_crop = crop(images['alpha'], top, left, self._crop_size[0], self._crop_size[1])

            images['alpha'] = alpha_crop
            for key in images.keys():
                images[key] = crop(images[key], top, left, self._crop_size[0], self._crop_size[1]) \
                    if images[key] is not None else None

        else:
            for key in images.keys():
                images[key] = crop(images[key], top, left, self._crop_size[0], self._crop_size[1]) \
                    if images[key] is not None else None

    def pad_source(self, source):
        # determine next reolution divisible by 16
        w, h = source.size
        w_pad = w // 16 * 16 + 16 if w % 16 != 0 else w
        h_pad = h // 16 * 16 + 16 if h % 16 != 0 else h

        # pad images
        source = np.asarray(source)
        source = np.pad(source, (((h_pad - h)//2, (h_pad - h)//2), ((w_pad - w)//2, (w_pad - w)//2), (0, 0)),
                        mode='reflect')
        return Image.fromarray(source)

    def resize(self, images):
        if max(self._max_resolution) > max(images['source'].size):
            for key in images.keys():
                if images[key] is not None:
                    images[key] = resize(images[key], self._max_resolution,
                                         InterpolationMode.BICUBIC if key == 'alpha' else InterpolationMode.NEAREST)


def read_meta_data(meta_file_path: str, min_id=0, max_id=20000):
    """Read the meta file containing source / target lens and disparity for each image.

    Args:
        meta_file_path (str): File path

    Raises:
        ValueError: File not found.

    Returns:
        dict: Meta dict of tuples like {id: (id, src_lens, tgt_lens, disparity)}.
    """
    if not osp.isfile(meta_file_path):
        raise ValueError(f"Meta file missing under {meta_file_path}.")

    meta = {}
    with open(meta_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
        meta[id] = (src_lens, tgt_lens, disparity)
    return meta


def calculate_bokeh_strength(src_lens, tgt_lens, disparity):
    src_fstop: float = float(src_lens.split("f")[1].split("BS")[0])
    tgt_fstop: float = float(tgt_lens.split("f")[1].split("BS")[0])

    src_strength: float = 1 / math.pow(2, src_fstop)
    tgt_strength: float = 1 / math.pow(2, tgt_fstop)

    lens_difference = tgt_strength - src_strength

    bokeh_strength = (min(1., max(-1., (lens_difference * (int(disparity) / 100)*2))) + 1) * 0.5
    return tensor(bokeh_strength, dtype=float32)


def get_lens_factor(tgt_lens):
    if tgt_lens.split("f")[0] == 'Canon50mm':
        return tensor(1, dtype=float32)
    elif tgt_lens.split("f")[0] == 'Sony50mm':
        return tensor(-1, dtype=float32)
    else:
        raise ValueError(f"Unknown lens {tgt_lens.split('f')[0]}")


def get_pos_map(w, h):
    if w > h:
        crop_dist = (1 - (h / w)) / 2
        x_lin = linspace(0, 1, w)
        y_lin = linspace(1 - crop_dist, crop_dist, h)
        return meshgrid(x_lin, y_lin, indexing='xy')
    elif w < h:
        crop_dist = (1 - (w / h)) / 2
        x_lin = linspace(crop_dist, 1 - crop_dist, w)
        y_lin = linspace(1, 0, h)
        return meshgrid(x_lin, y_lin, indexing='xy')
    else:  # w == h
        x_lin = linspace(0, 1, w)
        y_lin = linspace(1, 0, h)
        return meshgrid(x_lin, y_lin, indexing='xy')


def augmentations(images: dict):
    if random() < 0.5:
        for key in images.keys():
            images[key] = hflip(images[key]) if images[key] is not None else None
            # fix coordinates after flip
            if key == 'pos_x' and images[key] is not None:
                images[key] = 1 - images[key]
    if random() < 0.5:
        for key in images.keys():
            images[key] = vflip(images[key]) if images[key] is not None else None
            # fix coordinates after flip
            if key == 'pos_y' and images[key] is not None:
                images[key] = 1 - images[key]
    if random() < 0.5:
        angle = choice([90, 180, 270])
        for key in images.keys():
            images[key] = rotate(images[key], angle) if images[key] is not None else None
            # fix coordinates after rotation
            if key == 'pos_x' and images[key] is not None:
                if angle == 90:
                    images[key] = 1 - images[key]
                elif angle == 180:
                    images[key] = 1 - images[key]
                elif angle == 270:
                    images[key] = images[key]
                    # no need to invert coordinates, as they are already in the correct order
            elif key == 'pos_y' and images[key] is not None:
                if angle == 90:
                    images[key] = images[key]
                    # no need to invert coordinates, as they are already in the correct order
                elif angle == 180:
                    images[key] = 1 - images[key]
                elif angle == 270:
                    images[key] = 1 - images[key]
