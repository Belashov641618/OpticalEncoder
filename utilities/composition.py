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

    def _forward(self, field:torch.Tensor, distance:float=None):
        if distance is None:
            for element in self._elements:
                field = element.forward(field)
            return field
        else:
            for element in self._elements:
                if isinstance(element, AbstractPropagator):
                    if distance - element.distance > distance:
                        field = element.forward(field)
                        distance -= element.distance
                    else:
                        element_distance = element.distance
                        element.distance = distance
                        field = element.forward(field)
                        element.distance = element_distance
                        return field
                else:
                    field = element.forward(field)
            return field

    def forward(self, field:torch.Tensor):
        return self._forward(field)

    # Дополнительные методы
    @property
    def device(self):
        return self._elements[0].device
    @property
    def total_length(self):
        return sum([element.distance for element in self._propagators])
    @property
    def max_length_x(self):
        return max([length.x for length in self.length])
    @property
    def max_length_y(self):
        return max([length.y for length in self.length])
    @property
    def max_pixels_x(self):
        return max([pixels.x for pixels in self.pixels])
    @property
    def max_pixels_y(self):
        return max([pixels.y for pixels in self.pixels])

    # Методы получения расширенных данных
    def volume(self, field:torch.Tensor):
        pass
    def profile(self, field:torch.Tensor, profile_function:Callable[[torch.Tensor], torch.Tensor], orientation:Literal['xz','yz']='xz', pixels_xy:int=512, steps:int=255):
        while len(field.size()) < 4: field.unsqueeze(0)

        step_x = self.max_length_x / pixels_xy
        step_y = self.max_length_y / pixels_xy



        data = torch.zeros()

        distances = numpy.linspace(0, self.total_length, steps)
        for distance in distances:
            pass
