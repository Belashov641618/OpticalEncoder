from __future__ import annotations

import torch
import sys
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Literal, Union, Optional, Iterable
from tqdm import tqdm
from math import sqrt

from utilities import *
from parameters import DataSetsPath

class LoadSelector:
    _dataset : Dataset
    def __init__(self, dataset:Dataset):
        self._dataset = dataset
    def train(self):
        _ = next(iter(self._dataset.train))
    def test(self):
        _ = next(iter(self._dataset.test))

class ReferenceSelector:
    _dataset:Dataset
    def __init__(self, dataset:Dataset):
        self._dataset = dataset
    def set(self, state:int):
        if self._dataset._reference_type != state:
            self._dataset._reference_type = state
            self._dataset._delayed.add(self._dataset._reload)
    def default(self):
        self.set(0)
    def same(self):
        self.set(1)
    def mirrored(self):
        self.set(2)


class SameReferenceIterator:
    def __init__(self, loader:DataLoader):
        self.loader = loader
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self
    def __next__(self):
        image, label = next(self.iterator)
        return image, image.clone()
    def __len__(self):
        return len(self.loader)

class MirrorReferenceIterator:
    def __init__(self, loader:DataLoader):
        self.loader = loader
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self
    def __next__(self):
        image, label = next(self.iterator)
        return image, torch.flip(image.clone(), (2,3))
    def __len__(self):
        return len(self.loader)

LiteralDataSet  = Literal['MNIST', 'Flowers', 'STL10', 'CIFAR10']
DataSetToWidth  = dict(MNIST=28, Flowers=100, STL10=96, CIFAR10=32)
DataSetToHeight = dict(MNIST=28, Flowers=100, STL10=96, CIFAR10=32)
class Dataset:
    _delayed : DelayedFunctions

    _train : DataLoader
    _test  : DataLoader
    def _reload(self):
        if any(attr is None for attr in (self._dataset, self._batch)): raise AttributeError
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        if self._dataset == 'MNIST':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width-self._paddings[2]-self._paddings[3], self._height-self._paddings[0]-self._paddings[1]), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                *self._transformations,
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.MNIST(root=DataSetsPath, train=True, transform=transformation, download=True)
        elif self._dataset == 'Flowers':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width-self._paddings[2]-self._paddings[3], self._height-self._paddings[0]-self._paddings[1]), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                *self._transformations,
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.Flowers102(root=DataSetsPath, split='train', transform=transformation, download=True)
        elif self._dataset == 'STL10':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width-self._paddings[2]-self._paddings[3], self._height-self._paddings[0]-self._paddings[1]), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                *self._transformations,
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.STL10(root=DataSetsPath, split='train', transform=transformation, download=True)
        elif self._dataset == 'CIFAR10':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width-self._paddings[2]-self._paddings[3], self._height-self._paddings[0]-self._paddings[1]), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                *self._transformations,
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            dataset = datasets.CIFAR10(root=DataSetsPath, train=True,   transform=transformation, download=True)
        else:
            sys.stdout.close()
            sys.stdout = original_stdout
            raise ValueError(f'Dataset is {self._dataset}')
        sys.stdout.close()
        sys.stdout = original_stdout

        sampler = self._sampler_type(dataset, **self._sampler_kwargs)
        self._train = DataLoader(dataset, batch_size=self._batch, sampler=sampler, pin_memory=True, num_workers=self._threads, prefetch_factor=self._preload)
        self._test = DataLoader(dataset, batch_size=self._batch, sampler=sampler, pin_memory=True, num_workers=self._threads, prefetch_factor=self._preload)
    @property
    def train(self):
        self._delayed.launch()
        if self._reference_type == 1:
            return SameReferenceIterator(self._train)
        elif self._reference_type == 2:
            return MirrorReferenceIterator(self._train)
        else:
            return self._train
    @property
    def test(self):
        self._delayed.launch()
        if self._reference_type == 1:
            return SameReferenceIterator(self._test)
        elif self._reference_type == 2:
            return MirrorReferenceIterator(self._test)
        else:
            return self._test

    # Transformations
    _transformations:list[torch.nn.Module]
    _padding_values:tuple[int,...]
    def _set_padding_values(self, values:tuple[int,...]):
        if values != self._padding_values:
            self._padding_values = values
            index = -1
            for i, element in enumerate(self._transformations):
                if isinstance(element, transforms.Pad):
                    index = i
                    break
            if index >= 0:
                self._transformations[index] = transforms.Pad(self._padding_values)
            else:
                self._transformations.append(transforms.Pad(self._padding_values))
            self._delayed.add(self._reload)
    @property
    def _paddings(self):
        if len(self._padding_values) == 1:
            return self._padding_values[0], self._padding_values[0], self._padding_values[0], self._padding_values[0]
        elif len(self._padding_values) == 2:
            return self._padding_values[0], self._padding_values[0], self._padding_values[1], self._padding_values[1]
        elif len(self._padding_values) == 4:
            return self._padding_values
        else:
            raise AttributeError
    def padding(self, pixels:Union[int,tuple[int,int],tuple[int,int,int,int]]=None, percent:Union[float,tuple[float,float],tuple[float,float,float,float]]=None, surface_ratio:float=None):
        if pixels is not None:
            if isinstance(pixels, int):
                pixels = (pixels, pixels, pixels, pixels)
            elif len(pixels) == 2:
                pixels = (pixels[0], pixels[0], pixels[1], pixels[1])
            elif len(pixels) != 4:
                raise AttributeError
            self._set_padding_values(pixels)
        elif percent is not None:
            if isinstance(percent, float):
                percent = (percent, percent, percent, percent)
            elif len(percent) == 2:
                percent = (percent[0], percent[0], percent[1], percent[1])
            elif len(percent) != 4:
                raise AttributeError
            self._set_padding_values((int(self._height*percent[0]), int(self._height*percent[1]), int(self.width*percent[2]), int(self._height*percent[3])))
        elif surface_ratio is not None:
            ratio = (1.0 - sqrt(surface_ratio))/2
            self._set_padding_values((int(self._height*ratio), int(self._height*ratio), int(self._width*ratio), int(self._width*ratio)))
    def _init_transformations(self):
        self._transformations = []
        self._padding_values = (0,0,0,0)

    # Properties
    _dataset : Optional[LiteralDataSet]
    @property
    def dataset(self):
        class DatasetSelector:
            _self : Dataset
            def __init__(self, _self:Dataset): self._self = _self
            def get(self): return self._self._dataset

            def set(self, dataset:LiteralDataSet):
                if not hasattr(self._self, '_dataset') or self._self._dataset != dataset:
                    if not hasattr(self._self, '_width') or self._self._width is None:
                        self._self.width = DataSetToWidth[dataset]
                    if not hasattr(self._self, '_height') or self._self._height is None:
                        self._self.height = DataSetToHeight[dataset]
                    self._self._delayed.add(self._self._reload)
                self._self._dataset = dataset
            def __eq__(self, dataset:Union[LiteralDataSet,DatasetSelector]):
                if isinstance(dataset, DatasetSelector):
                    super().__eq__(dataset)
                else: self.set(dataset)

            def mnist(self):    self.set('MNIST')
            def flowers(self):  self.set('Flowers')
            def stl10(self):    self.set('STL10')
            def cifar10(self):  self.set('CIFAR10')
        return DatasetSelector(self)

    _reference_type:int
    @property
    def reference(self):
        return ReferenceSelector(self)

    _batch : Optional[int]
    @property
    def batch(self):
        return self._batch
    @batch.setter
    def batch(self, size:int):
        if not hasattr(self, '_batch') or size != self._batch:
            self._delayed.add(self._reload)
            self._batch = size

    _width : Optional[int]
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, pixels:int):
        if pixels is None:
            pixels = DataSetToWidth[self.dataset.get()]
        if not hasattr(self, '_width') or pixels != self._width:
            self._delayed.add(self._reload)
            self._width = pixels

    _height : Optional[int]
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, pixels:int):
        if pixels is None:
            pixels = DataSetToHeight[self.dataset.get()]
        if not hasattr(self, '_height') or pixels != self._height:
            self._delayed.add(self._reload)
            self._height = pixels

    _dtype : Optional[torch.dtype]
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, type:torch.dtype):
        if not hasattr(self, '_dtype') or type != self._dtype:
            self._delayed.add(self._reload)
            self._dtype = type

    _sampler_type : Optional[type(torch.utils.data.Sampler)]
    _sampler_kwargs : Optional[dict]
    @property
    def sampler(self):
        class SamplerSelector:
            _self:Dataset
            def __init__(self, _self:Dataset):
                self._self = _self
            def get(self):
                return self._self._sampler_type
            def set(self, type:type(torch.utils.data.Sampler), **kwargs):
                if not hasattr(self._self, '_sampler_type') or self._self._sampler_type != type or self._self._sampler_kwargs != kwargs:
                    self._self._delayed.add(self._self._reload)
                self._self._sampler_type = type
                self._self._sampler_kwargs = kwargs

            def default(self):
                self.set(torch.utils.data.RandomSampler)
            def parallel(self, rank:int, world_size:int):
                self.set(torch.utils.data.distributed.DistributedSampler, rank=rank, num_replicas=world_size)
        return SamplerSelector(self)

    _threads : Optional[int]
    @property
    def threads(self):
        return self._threads
    @threads.setter
    def threads(self, num_workers:int):
        if not hasattr(self, '_threads') or num_workers != self._threads:
            self._delayed.add(self._reload)
            self._threads = num_workers

    _preload : Optional[int]
    @property
    def preload(self):
        return self._preload
    @preload.setter
    def preload(self, amount:int):
        if not hasattr(self, '_preload') or amount != self._preload:
            self._delayed.add(self._reload)
            self._preload = amount

    @property
    def confusion_allowed(self):
        if self._reference_type == 1 or self._reference_type == 2:
            return False
        return True
    
    def __init__(self, dataset:LiteralDataSet=None, batch:int=None, width:int=None, height:int=None, dtype:torch.dtype=torch.float32, sampler:type(torch.utils.data.Sampler)=torch.utils.data.RandomSampler, threads:int=os.cpu_count(), preload:int=2):
        self._delayed = DelayedFunctions()

        self._reference_type = 0

        if dataset is None: self._dataset = None
        else: self.dataset.set(dataset)

        self.batch  = batch
        self.width  = width
        self.height = height
        self.dtype  = dtype
        self.sampler.set(sampler)
        self.threads = threads
        self.preload = preload

        self._init_transformations()
    @property
    def load(self):
        return LoadSelector(self)

    @staticmethod
    def single(dataset:LiteralDataSet, width:int, height:int, dtype:torch.dtype=torch.float32):
        data = Dataset(dataset, 1, width, height, dtype)
        return next(iter(data.test))