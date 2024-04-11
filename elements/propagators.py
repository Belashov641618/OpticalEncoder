import torch

from utilities import *
from .abstracts import AbstractPropagator

class FurrierPropagation(AbstractPropagator):
    _pad_pixels:XYParams[int]
    def _reset_pad_pixels(self):
        self._pad_pixels.x = upper_integer(self.border_ratio * self.pixels.input.x)
        self._pad_pixels.y = upper_integer(self.border_ratio * self.pixels.input.y)
    @property
    def _padding_x(self):
        return self._pad_pixels.x + super()._padding_x
    @property
    def _padding_y(self):
        return self._pad_pixels.y + super()._padding_y
    @property
    def _unpadding_x(self):
        return -self._pad_pixels.x + super()._unpadding_x
    @property
    def _unpadding_y(self):
        return -self._pad_pixels.y + super()._unpadding_y

    def _recalc_propagation_buffer(self):
        freq_x = torch.fft.fftfreq(self._total_pixels_x, d=self._step_x, device=self.device, dtype=self.accuracy.tensor_float)
        freq_y = torch.fft.fftfreq(self._total_pixels_y, d=self._step_y, device=self.device, dtype=self.accuracy.tensor_float)
        freq_x_mesh, freq_y_mesh = torch.meshgrid(freq_x, freq_y, indexing='ij')

        wavelength = self.wavelength.tensor.reshape(-1,1,1).to(self.device)
        reflection = (self.reflection.tensor + 1j*self.absorption.tensor).reshape(-1,1,1).to(self.device)

        K2 = ((1.0 / (wavelength * reflection))**2)
        Kz = (2. * torch.pi * torch.sqrt(K2 - freq_x_mesh**2 - freq_y_mesh**2))

        self._register_propagation_buffer(torch.exp(1j * Kz * self.distance))

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

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, distance:float, border_ratio:float=0.5, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, distance, logger=logger)
        self.border_ratio = border_ratio
        self._pad_pixels = XYParams[int](None, None)
        self._pad_pixels.set(0)
        self._reset_all()
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(field,*args, **kwargs)
        field = torch.nn.functional.pad(field, self._paddings)
        field = torch.fft.fft2(field)
        field = field * self._propagation_buffer
        field = torch.fft.ifft2(field)
        field = torch.nn.functional.pad(field, self._unpaddings)
        field = interpolate(field, (self.pixels.output.x, self.pixels.output.y), mode=self.interpolation.mode)
        return field