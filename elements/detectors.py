import torch
from math import sqrt

from elements.abstracts import AbstractDetectors
from utilities import *

class ClassificationDetectors(AbstractDetectors):
    _detectors:Param[int]
    @property
    def detectors(self):
        return self._detectors.value
    _detectors_buffer:torch.Tensor
    _detectors_filter:Param[filters.Filter]
    def _register_detectors_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device: buffer = buffer.to(self.device)
        if buffer.dtype != self.accuracy.tensor_float: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_detectors_buffer'): self.accuracy.disconnect(self._detectors_buffer)
        self.register_buffer('_detectors_buffer', buffer)
        self.accuracy.connect(self._detectors_buffer)
    def _recalc_detectors_buffer(self):
        filters_array = torch.zeros((self._detectors.value, self._total_pixels_x, self._total_pixels_y))

        rows = int(sqrt(self._detectors.value))
        cols = self._detectors.value // rows
        rest = self._detectors.value - rows * cols

        shift_y = self.length.input.y / (rows + 1)

        x_array_0 = torch.linspace(0, self.length.input.x, self._total_pixels_x)
        y_array_0 = torch.linspace(0, self.length.input.y, self._total_pixels_y)

        i:int = 0
        for row in range(rows):
            y0 = shift_y * (row + 1)
            cols_ = cols
            if rest > 0:
                cols_ += 1
                rest -= 1
            shift_x = self.length.input.x / (cols_ + 1)
            for col in range(cols_):
                x0 = shift_x * (col + 1)
                filters_array[i] = self._detectors_filter.value(x_array_0 - x0, y_array_0 - y0)
                i += 1

        self._register_detectors_buffer(filters_array)
    def _attach_recalc_detectors_buffer(self):
        self.delayed.add(self._recalc_detectors_buffer)
    @property
    def filter(self):
        self.delayed.launch()
        return self._detectors_buffer.clone().detach().cpu()
    @filter.setter
    def filter(self, filter:filters.Filter):
        self._detectors_filter.value = filter

    @property
    def _total_pixels_x(self):
        return super()._total_pixels_x + 2*self._unpadding_x
    @property
    def _total_pixels_y(self):
        return super()._total_pixels_y + 2*self._unpadding_y

    def _change_pixels(self):
        super()._change_pixels()
        self.delayed.add(self._recalc_detectors_buffer)
    def _change_length(self):
        super()._change_length()
        self.delayed.add(self._recalc_detectors_buffer)

    _wavelength_buffer:torch.Tensor
    def _change_wavelength(self):
        self.register_buffer('_wavelength_buffer', self.wavelength.tensor)

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, detectors:int, detector_filter:filters.Filter, spectral_filter:filters.Filter, logger:Logger=None):
        super().__init__(pixels, length, wavelength, spectral_filter, logger=logger)
        self._detectors_filter = Param[filters.Filter](self._attach_recalc_detectors_buffer)
        self._detectors_filter.set(detector_filter)
        self._detectors = Param[int](self._attach_recalc_detectors_buffer)
        self._detectors.set(detectors)
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)

        field = torch.nn.functional.pad(torch.abs(field), self._paddings_difference)
        field = field.reshape(field.shape[0], 1, *field.shape[1:]) * (self._wavelength_buffer.reshape(1, -1, 1, 1) * self._detectors_buffer.reshape(self._detectors_buffer.shape[0], 1, *self._detectors_buffer.shape[1:]))
        field = torch.sum(field, dim=(2,3,4))

        return field

class MatrixDetectors(AbstractDetectors):
    pass