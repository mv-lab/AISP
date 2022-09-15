import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import io
from torchvision.transforms import functional as tf

VAL = .1
INPUT = ".jpg"
TARGET = ".npy"

def sort_key(filepath):
    # sorts by name
    return  int(filepath.name.split("_")[0])


def read_image(path):
    return tf.convert_image_dtype(io.read_image(path), torch.float32)


def save_raw(raw, path, extension="npy"):
    raw = (raw * 1024).astype(np.uint16)
    final_path = f"{path}.{extension}"
    np.save(final_path, raw) 


class Evaluation_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_dir = opt.dataset_dir

        source = Path(root_dir)
        input_list = sorted(source.glob(f"*{INPUT}"),  key=sort_key)

        self.patch_list = [
            {"input": path.as_posix()}  for path in input_list
        ]


    def __getitem__(self, index):
        input = read_image(self.patch_list[index]["input"])
        name = self.patch_list[index]["input"].split('/')[-1].split(".")[0]
        sample = {'input':input, 'name':name}
        return sample

    def __len__(self):
        return len(self.patch_list)