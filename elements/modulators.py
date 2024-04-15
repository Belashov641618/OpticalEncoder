import torch

from elements.abstracts import AbstractMask
from utilities import *

class Lens(AbstractMask):
    def _recalc_mask_buffer(self):
        x_array = torch.linspace(-self._total_length_x/2, self._total_length_x/2, self._total_pixels_x, device=self.device, dtype=self.accuracy.tensor_float)
        y_array = torch.linspace(-self._total_length_y/2, self._total_length_y/2, self._total_pixels_y, device=self.device, dtype=self.accuracy.tensor_float)
        x_mesh, y_mesh = torch.meshgrid(x_array, y_array, indexing='ij')

        wavelength = self.wavelength.tensor.reshape(-1,1,1).to(self.device)
        if self.focus.left == self.focus.right:
            # Хроматические аббериации задаются через вариацию фокусного расстояния
            focus = self.focus.tensor.reshape(-1, 1, 1).to(self.device)
            phase = torch.pi * (x_mesh**2 + y_mesh**2) / (wavelength * focus)
        else:
            # Хроматические абберации заданы через вариацию коэффициента преломления
            reflection = (self.reflection.tensor + 1j*self.absorption.tensor).reshape(-1,1,1).to(self.device)
            space_reflection = (self.space_reflection.tensor + 1j*self.space_absorption.tensor).reshape(-1,1,1).to(self.device)
            phase = torch.pi * (x_mesh ** 2 + y_mesh ** 2) / (self.wavelength.effective * self.focus.effective)
            height = self.wavelength.effective * (self.reflection.effective - self.space_reflection.effective) * phase / (2.0*torch.pi)
            phase = 2*torch.pi*height * (reflection - space_reflection) / wavelength
        self._register_mask_buffer(torch.exp(1j*phase))

    focus:SpaceParam[float]
    def _change_focus(self):
        self.delayed.add(self._recalc_mask_buffer)

    def _change_pixels(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_length(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_wavelength(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_reflection(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_absorption(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_space_reflection(self):
        self.delayed.add(self._recalc_mask_buffer)
    def _change_space_absorption(self):
        self.delayed.add(self._recalc_mask_buffer)

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, space_reflection:FloatS, space_absorption:FloatS, focus:FloatS, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, space_reflection, space_absorption, logger=logger)
        self.focus = SpaceParam[float](self._change_focus, group=self.wavelength.group)
        self.focus.set(focus)
        self.accuracy.connect(self.focus.tensor)

    def forward(self, field:torch.Tensor, *args, **kwargs):
        return super().forward(field, *args, **kwargs)