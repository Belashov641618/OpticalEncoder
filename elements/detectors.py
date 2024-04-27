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
    detectors:XYParams[int]

    _detectors_buffer:torch.Tensor
    _detectors_filter:Param[filters.Filter]
    def _register_detectors_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device: buffer = buffer.to(self.device)
        if buffer.dtype != self.accuracy.tensor_float: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_detectors_buffer'): self.accuracy.disconnect(self._detectors_buffer)
        self.register_buffer('_detectors_buffer', buffer)
        self.accuracy.connect(self._detectors_buffer)
    def _recalc_detectors_buffer(self):
        ratio = 0.95
        x_array_0 = torch.linspace(-self.length.input.x/2, self.length.input.x/2, self._total_pixels_x)
        y_array_0 = torch.linspace(-self.length.input.y/2, self.length.input.y/2, self._total_pixels_y)
        filter = self._detectors_filter.value(x_array_0, y_array_0)
        integral = torch.sum(filter)
        pad0 = 0
        pad1 = (min(self._total_pixels_x, self._total_pixels_y) - 1) // 2
        state0 = torch.sum(torch.nn.functional.pad(filter, [-pad0]*4)) / integral < ratio
        state1 = torch.sum(torch.nn.functional.pad(filter, [-pad1]*4)) / integral < ratio
        while abs(pad0 - pad1) > 1:
            pad = int((pad0 + pad1) / 2)
            state = torch.sum(torch.nn.functional.pad(filter, [-pad]*4)) / integral < ratio
            if state == state0: state0, pad0 = state, pad
            else:               state1, pad1 = state, pad
        self._register_detectors_buffer(torch.nn.functional.pad(filter, [-int((pad0 + pad1)/2)]*4))
    def _attach_recalc_detectors_buffer(self):
        self.delayed.add(self._recalc_detectors_buffer)
    @property
    def filter(self):
        self.delayed.launch()

        paddings, dx, dy = self._convolution_params

        result = torch.zeros((self.pixels.input.x, self.pixels.input.y), device=self.device)
        result = torch.nn.functional.pad(result, self._paddings_difference)
        result = torch.nn.functional.pad(result, paddings)

        for i in range(self._detectors.x):
            left = dx * i
            right = left + self._detectors_buffer.shape[0]
            for j in range(self._detectors.y):
                top = dy * j
                bottom = top + self._detectors_buffer.shape[1]
                result[left:right, top:bottom] += self._detectors_buffer
        paddings_ = [-pad0-pad0 for pad0, pad1 in zip(self._paddings_difference, paddings)]
        result = torch.nn.functional.pad(result, paddings_)

        return result.clone().detach().cpu()

    @filter.setter
    def filter(self, filter:filters.Filter):
        self._detectors_filter.value = filter
    @property
    def _convolution_params(self):
        dx = self._total_pixels_x // self._detectors.x
        dy = self._total_pixels_y // self._detectors.y

        diffx = (self._total_pixels_x - dx*(self._detectors.x - 1) - 1)
        diffx_left = diffx // 2
        diffx_right = diffx - diffx_left
        kern_left = (self._detectors_buffer.shape[0] - 1) // 2
        kern_right = self._detectors_buffer.shape[0] - 1 - kern_left
        left = kern_left - diffx_left
        right = kern_right - diffx_right

        diffy = (self._total_pixels_y - dy * (self._detectors.y - 1) - 1)
        diffy_top = diffy // 2
        diffy_bottom = diffy - diffy_top
        kern_top = (self._detectors_buffer.shape[1] - 1) // 2
        kern_bottom = self._detectors_buffer.shape[1] - 1 - kern_top
        top = kern_top - diffy_top
        bottom = kern_bottom - diffy_bottom

        return [left, right, top, bottom], dx, dy

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

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, detectors:IntXY, detector_filter:filters.Filter, spectral_filter:filters.Filter, logger:Logger=None):
        super().__init__(pixels, length, wavelength, spectral_filter, logger=logger)
        self._detectors_filter = Param[filters.Filter](self._attach_recalc_detectors_buffer)
        self._detectors_filter.set(detector_filter)
        self._detectors = XYParams[int](change=self._attach_recalc_detectors_buffer)
        self._detectors.set(detectors)
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)

        field = torch.nn.functional.pad(torch.abs(field), self._paddings_difference)

        paddings, dx, dy = self._convolution_params
        field = torch.nn.functional.pad(field, paddings)

        kernel = self._detectors_buffer.repeat(1, self.wavelength.size, 1, 1) * self._spectral_buffer.reshape(1, -1, 1, 1)
        signals = torch.nn.functional.conv2d(field, kernel, stride=(dx, dy), groups=self.wavelength.size)
        signals = signals[:,:,0:self._detectors.x,0:self._detectors.y]*self._step_x*self._step_y
        signals = torch.mean(signals, dim=1).unsqueeze(1)
        print(signals)
        return signals



