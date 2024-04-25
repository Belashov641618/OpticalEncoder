import torch

from elements.abstracts import AbstractMask, AbstractModulator, AbstractInhomogeneity, AbstractElement
from utilities import *

# Статические маски
class Lens(AbstractMask):
    def _recalc_mask_buffer(self):
        x_array = torch.linspace(-self._total_length_x/2, self._total_length_x/2, self._total_pixels_x, device=self.device, dtype=self.accuracy.tensor_float)
        y_array = torch.linspace(-self._total_length_y/2, self._total_length_y/2, self._total_pixels_y, device=self.device, dtype=self.accuracy.tensor_float)
        x_mesh, y_mesh = torch.meshgrid(x_array, y_array, indexing='ij')

        wavelength = self.wavelength.tensor.reshape(-1,1,1).to(self.device)
        if self.focus.left != self.focus.right:
            # Хроматические аббериации задаются через вариацию фокусного расстояния
            focus = self.focus.tensor.reshape(-1, 1, 1).to(self.device)
            phase = torch.pi * (x_mesh**2 + y_mesh**2) / (wavelength * focus)
            # phase = 4 * torch.pi * torch.sqrt(4 * focus**2 + x_mesh**2 + y_mesh**2) / wavelength
        else:
            # Хроматические абберации заданы через вариацию коэффициента преломления
            reflection = (self.reflection.tensor + 1j*self.absorption.tensor).reshape(-1,1,1).to(self.device)
            space_reflection = (self.space_reflection.tensor + 1j*self.space_absorption.tensor).reshape(-1,1,1).to(self.device)
            phase = torch.pi * (x_mesh ** 2 + y_mesh ** 2) / (self.wavelength.effective * self.focus.effective)
            # phase = 4 * torch.pi * torch.sqrt(4 * self.focus.effective**2 + x_mesh**2 + y_mesh**2) / self.wavelength.effective
            height = (self.wavelength.effective / (self.reflection.effective - self.space_reflection.effective)) * phase / (2.0*torch.pi)
            phase = 2*torch.pi*height * (reflection - space_reflection) / wavelength
        self._register_mask_buffer(torch.exp(-1j*phase))

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
        super().__init__(pixels, length, wavelength, reflection, absorption, space_reflection, space_absorption, logger=logger, finalize=False)
        self.focus = SpaceParam[float](self._change_focus, group=self.wavelength.group)
        self.focus.set(focus)
        self.accuracy.connect(self.focus.tensor)
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        return super().forward(field, *args, **kwargs)


# Простые модуляторы
class AmplitudeModulator(AbstractModulator):
    def _multiplier(self):
        parameters = interpolate(self._normalized(), (self.pixels.input.x, self.pixels.input.y), InterpolateModes.nearest)
        coefficients = torch.nn.functional.pad(parameters, self._paddings_difference).to(self.accuracy.tensor_complex)
        return coefficients

    def __init__(self, pixels:IntIO, length:FloatIO, mask_pixels:IntXY, logger:Logger=None):
        super().__init__(pixels, length, mask_pixels, logger=logger)

class PhaseModulator(AbstractModulator):
    def _multiplier(self):
        parameters = interpolate(self._normalized(), (self.pixels.input.x, self.pixels.input.y), InterpolateModes.nearest)
        coefficients = torch.exp(2j*torch.pi*torch.nn.functional.pad(parameters, self._paddings_difference))
        return coefficients

    def __init__(self, pixels:IntIO, length:FloatIO, mask_pixels:IntXY, logger:Logger=None):
        super().__init__(pixels, length, mask_pixels, logger=logger)

# Физические модуляторы
class PerfectHeightModulator(AbstractModulator, AbstractInhomogeneity):
    @property
    def _max_height(self):
        return torch.max(self.wavelength.tensor / (self.reflection.tensor - self.space_reflection.tensor))

    def _heights(self):
        return interpolate(self._normalized() * self._max_height, (self.pixels.input.x, self.pixels.input.y), InterpolateModes.nearest)

    def _multiplier(self):
        return torch.exp(2j*torch.pi*self._heights()*(self.reflection.tensor - self.space_reflection.tensor + 1j*(self.absorption.tensor - self.space_absorption.tensor)))

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, space_reflection:FloatS, space_absorption:FloatS, mask_pixels:IntXY, logger:Logger=None):
        AbstractInhomogeneity.__init__(self, pixels, length, wavelength, reflection, absorption, space_reflection, space_absorption, logger=logger)
        AbstractModulator.__init__(self, pixels, length, mask_pixels, logger=logger)

# Маска
class Mask(AbstractElement):
    _mask_buffer:torch.Tensor
    def _register_mask_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device: buffer = buffer.to(self.device)
        if buffer.dtype not in (self.accuracy.tensor_float, self.accuracy.tensor_complex):
            if torch.is_complex(buffer): buffer = buffer.to(self.accuracy.tensor_complex)
            else: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_mask_buffer'): self.accuracy.disconnect(self._mask_buffer)
        self.register_buffer('_mask_buffer', buffer)
        self.accuracy.connect(self._mask_buffer)
    @property
    def device(self):
        if hasattr(self, '_mask_buffer'):
            return self._mask_buffer.device
        else:
            return super().device
    @property
    def buffer(self):
        return self._mask_buffer
    @buffer.setter
    def buffer(self, data:torch.Tensor):
        self._register_mask_buffer(data)

    def __init__(self, pixels:IntIO, length:FloatIO, logger:Logger=None):
        super().__init__(pixels, length, logger)
        self._register_mask_buffer(torch.ones((self.pixels.input.x, self.pixels.input.y)))

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)

        return field * self._mask_buffer