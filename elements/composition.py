import numpy
import torch

from typing import Literal, Union

from tqdm import tqdm

from elements.abstracts import AbstractElement, AbstractOptical, AbstractPropagator, AbstractWrapper, AbstractEncoderDecoder, AbstractDetectors, AbstractModulator
from utilities import *
from parameters import FigureWidthHeight, FontLibrary


class CompositeModel(torch.nn.Sequential):
    _elements:tuple[AbstractElement, ...]
    _optical:tuple[AbstractOptical, ...]
    _propagators:tuple[AbstractPropagator, ...]
    _modulators:tuple[AbstractModulator, ...]
    _wrappers:tuple[AbstractEncoderDecoder, ...]
    def _init_groups(self, *modules:torch.nn.Module):
        self._elements = []
        self._optical = []
        self._propagators = []
        self._modulators = []
        self._wrappers = []
        groups_map = [
            (self._elements, torch.nn.Module),
            (self._optical, AbstractOptical),
            (self._propagators, AbstractPropagator),
            (self._modulators, AbstractModulator),
            (self._wrappers, AbstractEncoderDecoder),
        ]
        for module in modules:
            for group, type_ in groups_map:
                if isinstance(module, type_):
                    group.append(module)
        self._elements = tuple(self._elements)
        self._optical = tuple(self._optical)
        self._propagators = tuple(self._propagators)
        self._modulators = tuple(self._modulators)
        self._wrappers = tuple(self._wrappers)
    @property
    def count(self):
        return len(self._elements)
    def element(self, position:int):
        return self._elements[position % self.count]

    def __init__(self, *elements:AbstractElement):
        super().__init__(*elements)
        self._init_groups(*elements)

    def forward(self, field:torch.Tensor, *args, distance:float=None, elements:int=None, **kwargs):
        wrapper_index = 0
        for i, element in enumerate(self._elements):
            if element is self._wrappers:
                wrapper_index += 1
            if distance is not None and isinstance(element, AbstractPropagator):
                distance -= element.distance
                if distance < 0:
                    element_distance = element.distance
                    element.distance = element_distance + distance
                    field = element.forward(field)
                    element.distance = element_distance
                    for wrapper in self._wrappers[wrapper_index+1:]:
                        field = wrapper.forward(field)
                    return field
            field = element.forward(field)
            if elements is not None and i == elements:
                return field
        return field

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
    def total_length(self) -> float:
        return sum([element.distance for element in self._propagators])

    # Методы получения расширенных данных
    def volume(self, field:torch.Tensor, pixels_x:int, pixels_y:int, pixels_z:int, interpolation:IMType=InterpolateModes.bilinear):
        with torch.no_grad():
            result = torch.zeros((pixels_z, pixels_x, pixels_y), dtype=field.dtype, device=torch.device('cpu'))
            result[0] = interpolate(field, (pixels_x, pixels_y), interpolation).squeeze().cpu()
            distance_array = numpy.linspace(0., self.total_length, pixels_z)[1:]
            for i, distance in tqdm(enumerate(distance_array, 1), total=len(distance_array)):
                distance:float
                result[i] = interpolate(self.forward(field, distance=distance), (pixels_x, pixels_y), interpolation).squeeze().cpu()
            return result.movedim(0,2)

    def profile(self, field:torch.Tensor, orientation:Literal['xz','yz']='xz', reduce:Literal['select','mean','min','max']='select', pixels_xy:int=255, pixels_z:int=255, interpolation:IMType=InterpolateModes.bilinear):
        volume = self.volume(field, pixels_xy, pixels_xy, pixels_z, interpolation)
        if reduce == 'mean':
            plane = (torch.mean(volume.abs(), dim=1) if orientation == 'xz' else torch.mean(volume.abs(), dim=0))
        elif reduce == 'max':
            plane = (torch.max(volume.abs(), dim=1) if orientation == 'xz' else torch.max(volume.abs(), dim=0))
        elif reduce == 'min':
            plane = (torch.min(volume.abs(), dim=1) if orientation == 'xz' else torch.min(volume.abs(), dim=0))
        else:
            plane = (volume[:, pixels_xy // 2, :] if orientation == 'xz' else volume[pixels_xy // 2, :, :])
        return plane

    def planes(self, field:torch.Tensor, pixels_x:int=255, pixels_y:int=255, interpolation:IMType=InterpolateModes.bilinear):
        result = torch.zeros((len(self._elements)+1, pixels_x, pixels_y), dtype=field.dtype)
        if self._wrappers:
            def forward_(field_:torch.Tensor, *args, **kwargs):
                return field_
            self._wrappers[0].attach_forward(forward_)
            result[0] = interpolate(self._wrappers[-1].forward(field), (pixels_x, pixels_y), interpolation).squeeze().cpu()
            self._wrappers[0].attach_forward(self._forward)
        else:
            result[0] = interpolate(field, (pixels_x, pixels_y), interpolation).squeeze().cpu()
        for i in tqdm(range(self.count), total=self.count):
            result[i+1] = interpolate(self.forward(field, elements=i), (pixels_x, pixels_y), interpolation).squeeze().cpu()
        return result


_OpticalModels = Union[CompositeModel]
class HybridModel(torch.nn.Module):
    _optical_model:_OpticalModels
    _detectors:AbstractDetectors
    _electronic_model:torch.nn.Module
    @property
    def device(self):
        return self._optical_model.device

    def __init__(self, optical:_OpticalModels, detectors:AbstractDetectors, electronic:torch.nn.Module):
        super().__init__()
        self._optical_model = optical
        self._detectors = detectors
        self._electronic_model = electronic
        self.add_module('optical', self._optical_model)
        self.add_module('detectors', self._detectors)
        self.add_module('electronic', self._electronic_model)

    def forward(self, field:torch.Tensor, *args, **kwargs):
        field = self._optical_model.forward(field)
        signals = self._detectors.forward(field)
        result = self._electronic_model.forward(signals)
        return result

    def visualize(self, image:torch.Tensor):

        plot = TiledPlot(*FigureWidthHeight)
        plot.FontLibrary = FontLibrary
        plot.title('Гибридная модель')

        with torch.no_grad():
            planes = self._optical_model.planes(image, self._optical_model.element(0).pixels.input.x, self._optical_model.element(0).pixels.input.y)
            signals = self._detectors.forward(self._optical_model.forward(image))
            results = self._electronic_model.forward(signals)

            planes = planes.cpu()
            signals = signals.squeeze().cpu()
            results = results.squeeze().cpu()

        total = planes.size(0)
        units = (total - 1) // 2
        
        kwargs = {'aspect':'auto'}
        for col, plane in enumerate(planes[1:-1]):
            plane:torch.Tensor
            axes = plot.axes.add(col, 0)
            axes.imshow(plane.abs(), **kwargs)

        axes = plot.axes.add((0,1), (units-1,units))
        axes.imshow(planes[0].abs(), **kwargs)

        axes = plot.axes.add((units,1), (2*units-1,units))
        axes.imshow(planes[-1].abs(), **kwargs)

        axes = plot.axes.add((0,units+1), (units-1, 2*units))
        axes.bar(numpy.arange(0, len(signals)), signals)

        axes = plot.axes.add((units, units+1), (2*units-1, 2*units))
        axes.bar(numpy.arange(0, len(results)), results)

        plot.show()