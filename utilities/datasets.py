from __future__ import annotations

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Literal, Union

from utilities import *
from parameters import DataSetsPath

LiteralDataSet = Literal['MNIST', 'Flowers', 'STL10', 'CIFAR10']
class Dataset:
    _delayed : DelayedFunctions

    _train : DataLoader
    _test  : DataLoader
    def _reload(self):
        if self._dataset == 'MNIST':
            transformation = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(self._dtype)
                ])
            dataset = datasets.MNIST(root=DataSetsPath, train=True, transform=transformation, download=True)
        elif self._dataset == 'Flowers':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.Flowers102(root=DataSetsPath, split='train', transform=transformation, download=True)
        elif self._dataset == 'STL10':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.STL10(root=DataSetsPath, split='train', transform=transformation, download=True)
        elif self._dataset == 'CIFAR10':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.CIFAR10(root=DataSetsPath, train=True,   transform=transformation, download=True)
        else: raise ValueError(f'Dataset is {self._dataset}')
        sampler = self._sampler_type(dataset)
        self._train = DataLoader(dataset, batch_size=self._batch, sampler=sampler)
        self._test = DataLoader(dataset, batch_size=self._batch, sampler=sampler)
    @property
    def train(self):
        self._delayed.launch()
        return self._train
    @property
    def test(self):
        self._delayed.launch()
        return self._test

    # Properties
    _dataset : LiteralDataSet
    @property
    def dataset(self):
        class DatasetSelector:
            _self : Dataset
            def __init__(self, _self:Dataset): self._self = _self
            def get(self): return self._self._dataset

            def set(self, dataset:LiteralDataSet):
                if not hasattr(self._self, '_dataset') or self._self._dataset != dataset:
                    self._self._delayed.add(self._self._reload)
                self._self._dataset = dataset
            def __eq__(self, dataset:Union[LiteralDataSet,DatasetSelector]):
                if isinstance(dataset, DatasetSelector):
                    super().__eq__(dataset)
                else: self.set(dataset)

            def mnist(self):    self.set('MNIST')
            def flowers(self):  self.set('Flowers')
        return DatasetSelector(self)

    _batch : int
    @property
    def batch(self):
        return self._batch
    @batch.setter
    def batch(self, size:int):
        if not hasattr(self, '_batch') or size != self._batch:
            self._delayed.add(self._reload)
            self._batch = size

    _width : int
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, pixels:int):
        if not hasattr(self, '_width') or pixels != self._width:
            self._delayed.add(self._reload)
            self._width = pixels

    _height : int
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, pixels:int):
        if not hasattr(self, '_height') or pixels != self._height:
            self._delayed.add(self._reload)
            self._height = pixels

    _dtype : torch.dtype
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, type:torch.dtype):
        if not hasattr(self, '_dtype') or type != self._dtype:
            self._delayed.add(self._reload)
            self._dtype = type

    _sampler_type : type(torch.utils.data.Sampler)
    @property
    def sampler(self):
        class SamplerSelector:
            _self:Dataset
            def __init__(self, _self:Dataset):
                self._self = _self
            def get(self):
                return self._self._sampler_type
            def set(self, type:type(torch.utils.data.Sampler)):
                if not hasattr(self._self, '_sampler_type') or self._self._sampler_type != type:
                    self._self._delayed.add(self._self._reload)
                self._self._sampler_type = type

            def default(self):
                self.set(torch.utils.data.RandomSampler)
            def parallel(self):
                self.set(torch.utils.data.distributed.DistributedSampler)
        return SamplerSelector(self)

    def __init__(self, dataset:LiteralDataSet=None, batch:int=None, width:int=None, height:int=None, dtype:torch.dtype=torch.float32, sampler:type(torch.utils.data.Sampler)=torch.utils.data.RandomSampler):
        self._delayed = DelayedFunctions()

        if dataset is not None: self.dataset.set(dataset)
        if batch  is not None: self.batch  = batch
        if width  is not None: self.width  = width
        if height is not None: self.height = height
        if dtype  is not None: self.dtype  = dtype
        if sampler is not None: self.sampler.set(sampler)


    @staticmethod
    def single(dataset:LiteralDataSet, width:int, height:int, dtype:torch.dtype=torch.float32):
        data = Dataset(dataset, 1, width, height, dtype)
        return next(iter(data.test))