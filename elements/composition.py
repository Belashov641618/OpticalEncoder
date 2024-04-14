import numpy
import torch

from typing import Literal, Callable

from elements.abstracts import AbstractElement, AbstractOptical, AbstractPropagator
from utilities import *

class CompositeModel(torch.nn.Module):
    # Группы модулей
    _elements:tuple[AbstractElement,...]
    pixels:tuple[XYParams[int],...]
    _pixels_synchronizers:tuple[XYParamsSynchronizer,...]
    length:tuple[XYParams[int],...]
    _length_synchronizers:tuple[XYParamsSynchronizer,...]
    def _init_elements(self, *elements:AbstractElement):
        self._elements = elements

        self.pixels = tuple(*[XYParams[int]() for i in range(self.count + 1)])
        self._pixels_synchronizers = tuple(*[XYParamsSynchronizer(param) for param in self.pixels])
        for i, synchronizer in enumerate(self._pixels_synchronizers):
            if i == 0:
                synchronizer.connect(self._elements[0].pixels.parameter_input)
            elif i == self.count:
                synchronizer.connect(self._elements[-1].pixels.parameter_output)
            else:
                synchronizer.connect(self._elements[i-1].pixels.parameter_output)
                synchronizer.connect(self._elements[i].pixels.parameter_input)
        for synchronizer, param in zip(self._pixels_synchronizers, self.pixels):
            synchronizer.connect(param)

        self.length = tuple(*[XYParams[float]() for i in range(self.count + 1)])
        self._length_synchronizers = tuple(*[XYParamsSynchronizer(param) for param in self.length])
        for i, synchronizer in enumerate(self._length_synchronizers):
            if i == 0:
                synchronizer.connect(self._elements[0].length.parameter_input)
            elif i == self.count:
                synchronizer.connect(self._elements[-1].length.parameter_output)
            else:
                synchronizer.connect(self._elements[i-1].length.parameter_output)
                synchronizer.connect(self._elements[i].length.parameter_input)
        for synchronizer, param in zip(self._length_synchronizers, self.length):
            synchronizer.connect(param)
    @property
    def count(self):
        return len(self._elements)
    def element(self, position:int) -> AbstractElement:
        return self._elements[position % self.count]

    _optical:tuple[AbstractOptical,...]
    _optical_group: SpaceParamGroup
    wavelength:SpaceParam[float]
    _wavelength_synchronizer:SpaceParamSynchronizer
    def _init_optical(self):
        self._optical = tuple(*[element for element in self._elements if isinstance(element, AbstractOptical)])
        self._optical_group = SpaceParamGroup()
        for i, element in enumerate(self._optical):
            self._optical_group.merge(element.optical_group)
        self.wavelength = SpaceParam(group=self._optical_group)
        self._wavelength_synchronizer = SpaceParamSynchronizer(self.wavelength, *[element.wavelength for element in self._optical])

    _propagators:tuple[AbstractPropagator,...]
    def _init_propagators(self):
        self._propagators = tuple(*[element for element in self._optical if isinstance(element, AbstractPropagator)])

    # Дополнительные обёртки TODO
    _wrappers:list[None]
    def wrap(self, wrapper:None):
        raise NotImplementedError

    # Основные методы
    def __init__(self, *elements:AbstractElement):
        super().__init__()
        self._init_elements(*elements)
        self._init_optical()

    def _forward(self, field:torch.Tensor):
        for element in self._elements:
            field = element.forward(field)
        return field

    def forward(self, field:torch.Tensor):
        return self._forward(field)

    # Дополнительные методы
    @property
    def device(self):
        if self._propagators:
            return self._propagators[0].device
        else:
            return torch.device('cpu')
    @property
    def dtype(self):
        return self._elements[0].accuracy.tensor_complex
    @property
    def total_length(self):
        return sum([element.distance for element in self._propagators])

    # Методы получения расширенных данных
    def volume(self, field:torch.Tensor, pixels_x:int, pixels_y:int, pixels_z:int, interpolation:IMType=InterpolateModes.bilinear):
        with torch.no_grad():
            result = torch.zeros((pixels_z, pixels_x, pixels_y), dtype=field.dtype, device=torch.device('cpu'))
            result[0] =  interpolate(field.squeeze(), (pixels_x, pixels_y), interpolation).cpu()
            distance_array = torch.linspace(0, self.max_length_x, pixels_z, device=field.device)[1:]
            last_index:int = 0
            for element in self._elements:
                if isinstance(element, AbstractPropagator):
                    element_distance = element.distance
                    for distance in distance_array[last_index:]:
                        if distance > element_distance:
                            element.distance = element_distance
                            break
                        element.distance = distance
                        result[last_index+1] = interpolate(element.forward(field).squeeze(), (pixels_x, pixels_y), interpolation).cpu()
                        last_index += 1
                field = element.forward(field)
            return result.movedim(0,2)

    def profile(self, field:torch.Tensor, orientation:Literal['xz','yz']='xz', reduce:Literal['select','mean','min','max']='select', pixels_xy:int=255, pixels_z:int=255, interpolation:IMType=InterpolateModes.bilinear):
        volume = self.volume(field, pixels_xy, pixels_xy, pixels_z, interpolation)
        if reduce == 'mean':
            plane = (torch.mean(volume, dim=1) if orientation == 'xz' else torch.mean(volume, dim=0))
        elif reduce == 'max':
            plane = (torch.max(volume, dim=1) if orientation == 'xz' else torch.max(volume, dim=0))
        elif reduce == 'min':
            plane = (torch.min(volume, dim=1) if orientation == 'xz' else torch.min(volume, dim=0))
        else:
            plane = (volume[:, pixels_xy // 2, :] if orientation == 'xz' else volume[pixels_xy // 2, :, :])
        return plane

    def planes(self, field:torch.Tensor, pixels_x:int=255, pixels_y:int=255, interpolation:IMType=InterpolateModes.bilinear):
        result = torch.zeros((len(self._propagators), pixels_x, pixels_y), dtype=field.dtype, device=field.device)
        iterator:int = 0
        for element in self._elements:
            if isinstance(element, AbstractPropagator):
                field = element.forward(field)
                result[iterator] = interpolate(field.squeeze(), (pixels_x, pixels_y), interpolation).cpu()
            else:
                field = element.forward(field)
        return result

