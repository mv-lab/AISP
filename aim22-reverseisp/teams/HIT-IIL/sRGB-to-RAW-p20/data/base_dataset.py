import torch.utils.data as data
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt, split, dataset_name):
        self.opt = opt
        self.split = split
        self.root = opt.dataroot
        self.dataset_name = dataset_name.lower()

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass
 
