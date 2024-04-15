import numpy
import torch

from typing import Literal, Callable

from elements.abstracts import AbstractElement, AbstractOptical, AbstractPropagator
from utilities import *

class CompositeModel(torch.nn.Module):
    # Группы модулей
    _elements:tuple[AbstractElement,...]
    def _init_elements(self, *elements:AbstractElement):
        self._elements = elements
    @property
    def count(self):
        return len(self._elements)
    def element(self, position:int) -> AbstractElement:
        return self._elements[position % self.count]

    _optical:tuple[AbstractOptical,...]
    def _init_optical(self):
        self._optical = tuple(*[element for element in self._elements if isinstance(element, AbstractOptical)])

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
