import torch

from utilities import *
from .abstracts import AbstractPropagator

class FurrierPropagation(AbstractPropagator):
    _add_pixels:XYParams[int]
    def _reset_add_pixels(self):
        if not hasattr(self, '_add_pixels'):
            self._add_pixels = XYParams[int](None, None)
            self._add_pixels.set(0)
        self._add_pixels.x = upper_integer(closest_integer((self.length._output.x - self.length._input.x) * self.pixels._input.x / self.length._input.x) / 2)
        self._add_pixels.y = upper_integer(closest_integer((self.length._output.y - self.length._input.y) * self.pixels._input.y / self.length._input.y) / 2)
    _pad_pixels:XYParams[int]
    def _reset_pad_pixels(self):
        if not hasattr(self, '_pad_pixels'):
            self._pad_pixels = XYParams[int](None, None)
            self._pad_pixels.set(0)
        self._pad_pixels.x = upper_integer(self.border_ratio * self.pixels._input.x)
        self._pad_pixels.y = upper_integer(self.border_ratio * self.pixels._input.y)
    @property
    def _padding_x(self):
        return self._pad_pixels.x + (self._add_pixels.x if self._add_pixels.x > 0 else 0)
    @property
    def _padding_y(self):
        return self._pad_pixels.y + (self._add_pixels.y if self._add_pixels.y > 0 else 0)
    @property
    def _paddings(self):
        return self._padding_x, self._padding_x, self._padding_y, self._padding_y
    @property
    def _unpadding_x(self):
        return self._pad_pixels.x - (self._add_pixels.x if self._add_pixels.x < 0 else 0)
    @property
    def _unpadding_y(self):
        return self._pad_pixels.y - (self._add_pixels.y if self._add_pixels.y < 0 else 0)
    @property
    def _unpaddings(self):
        return self._unpadding_x, self._unpadding_x, self._unpadding_y, self._unpadding_y
    @property
    def _total_pixels_x(self):
        return self.pixels.input.x + 2 * self._padding_x
    @property
    def _total_pixels_y(self):
        return self.pixels.input.y + 2 * self._padding_y
    @property
    def _total_length_x(self):
        return self._total_pixels_x * self.length.input.x / self.pixels.input.x
    @property
    def _total_length_y(self):
        return self._total_pixels_y * self.length.input.y / self.pixels.input.y
    @property
    def _step_x(self):
        return self.length.input.x / self.pixels.input.x
    @property
    def _step_y(self):
        return self.length.input.y / self.pixels.input.y

    def _recalc_propagation_buffer(self):
        freq_x = torch.fft.fftfreq(self._total_pixels_x, d=self._step_x, device=self.device, dtype=self.accuracy.tensor_float)
        freq_y = torch.fft.fftfreq(self._total_pixels_y, d=self._step_y, device=self.device, dtype=self.accuracy.tensor_float)
        freq_x_mesh, freq_y_mesh = torch.meshgrid(freq_x, freq_y, indexing='ij')

        wavelength = self.wavelength.tensor.expand(1, 1, -1).movedim(2, 0).to(self.device).to(self.accuracy.tensor_float)
        reflection = (self.reflection.tensor + 1j*self.absorption.tensor).expand(1, 1, -1).movedim(2, 0).to(self.device).to(self.accuracy.tensor_complex)

        K2 = ((1.0 / (wavelength * reflection))**2).to(self.accuracy.tensor_complex)
        Kz = (2. * torch.pi * torch.sqrt(K2 - freq_x_mesh**2 - freq_y_mesh**2)).to(self.accuracy.tensor_complex)

        self.register_buffer('_propagation_buffer', torch.exp(1j * Kz * self.distance))

    def _change_border(self):
        self.delayed.add(self._reset_pad_pixels, -1.)
        self.delayed.add(self._recalc_propagation_buffer)
    @Param
    def border_ratio(self):
        self._change_border()

    def _reset_all(self):
        self.delayed.add(self._reset_add_pixels, -1.)
        self.delayed.add(self._reset_pad_pixels, -1.)
        self.delayed.add(self._recalc_propagation_buffer)
    def _change_pixels(self):
        self._reset_all()
    def _change_length(self):
        self._reset_all()
    def _change_distance(self):
        self.delayed.add(self._recalc_propagation_buffer)
    def _change_absorption(self):
        self.delayed.add(self._recalc_propagation_buffer)
    def _change_reflection(self):
        self.delayed.add(self._recalc_propagation_buffer)
    def _change_wavelength(self):
        self.delayed.add(self._recalc_propagation_buffer)

    interpolation:InterpolateMode

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, distance:float, border_ratio:float=0.5, interpolation:IMType=InterpolateModes.bilinear, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, distance, logger=logger)
        self.border_ratio = border_ratio
        self.interpolation = InterpolateMode(interpolation)
        self._reset_all()
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(field,*args, **kwargs)
        field = torch.nn.functional.pad(field, self._paddings)
        field = torch.fft.fft2(field)
        field = field * self._propagation_buffer
        field = torch.fft.ifft2(field)
        field = torch.nn.functional.pad(field, self._unpaddings)
        field = interpolate(field, (self.pixels._output.x, self.pixels._output.y), mode=self.interpolation.mode)
        return field