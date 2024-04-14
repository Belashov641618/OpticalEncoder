import torch

from utilities import *

class AbstractElement(torch.nn.Module):
    logger:Logger

    delayed:DelayedFunctions
    accuracy:Accuracy

    pixels:IOParams[int]
    def _change_pixels_input(self):
        pass
    def _change_pixels_output(self):
        pass
    def _change_pixels(self):
        pass

    length:IOParams[float]
    def _change_length_input(self):
        pass
    def _change_length_output(self):
        pass
    def _change_length(self):
        pass

    @property
    def device(self):
        return torch.device('cpu')

    _add_pixels:XYParams[int]
    def _reset_add_pixels(self):
        self._add_pixels.x = upper_integer(closest_integer((self.length.output.x - self.length.input.x) * self.pixels.input.x / self.length.input.x) / 2)
        self._add_pixels.y = upper_integer(closest_integer((self.length.output.y - self.length.input.y) * self.pixels.input.y / self.length.input.y) / 2)
    def _reset_add_pixels_(self):
        self.delayed.add(self._reset_add_pixels, -100.0)
    @property
    def _padding_x(self):
        return self._add_pixels.x if self._add_pixels.x > 0 else 0
    @property
    def _padding_y(self):
        return self._add_pixels.y if self._add_pixels.y > 0 else 0
    @property
    def _paddings(self):
        return self._padding_x, self._padding_x, self._padding_y, self._padding_y
    @property
    def _unpadding_x(self):
        return -self._add_pixels.x if self._add_pixels.x < 0 else 0
    @property
    def _unpadding_y(self):
        return -self._add_pixels.y if self._add_pixels.y < 0 else 0
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
    def _step_x(self):
        return self.length.input.x / self.pixels.input.x
    @property
    def _step_y(self):
        return self.length.input.y / self.pixels.input.y
    @property
    def _total_length_x(self):
        return self._total_pixels_x * self._step_x
    @property
    def _total_length_y(self):
        return self._total_pixels_y * self._step_y

    interpolation:InterpolateMode

    def __init__(self, pixels:IntIO, length:FloatIO, logger:Logger=None):
        super().__init__()
        self.accuracy = Accuracy()
        self.delayed = DelayedFunctions()

        self.pixels = IOParams[int](change_input=self._change_pixels_input, change_output=self._change_pixels_output, change=function_combiner(self._change_pixels, self._reset_add_pixels_))
        self.length = IOParams[float](change_input=self._change_length_input, change_output=self._change_length_output, change=function_combiner(self._change_length, self._reset_add_pixels_))

        self.pixels.set(pixels)
        self.length.set(length)

        self._add_pixels = XYParams[int](None, None)
        self._add_pixels.set(0)

        self.interpolation = InterpolateMode(InterpolateModes.bilinear)

        if logger is None:
            logger = Logger(False, prefix='Deleted')
        self.logger = logger

    def forward(self, *args, **kwargs):
        self.delayed.launch()

    def memory(self):
        pass

class AbstractOptical(AbstractElement):
    _optical_group:SpaceParamGroup
    @property
    def optical_group(self):
        return self._optical_group
    @optical_group.setter
    def optical_group(self, group:SpaceParamGroup):
        group.merge(self._optical_group)
        self._optical_group = group

    wavelength:SpaceParam[float]
    def _change_wavelength(self):
        pass

    reflection:SpaceParam[float]
    def _change_reflection(self):
        pass

    absorption:SpaceParam[float]
    def _change_absorption(self):
        pass

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, logger:Logger=None):
        super().__init__(pixels, length, logger=logger)
        self.wavelength = SpaceParam[float](self._change_wavelength)
        self.reflection = SpaceParam[float](self._change_reflection, group=self.wavelength.group)
        self.absorption = SpaceParam[float](self._change_absorption, group=self.wavelength.group)
        self.wavelength.set(wavelength)
        self.reflection.set(reflection)
        self.absorption.set(absorption)
        self.accuracy.connect(self.wavelength.tensor)
        self.accuracy.connect(self.reflection.tensor)
        self.accuracy.connect(self.absorption.tensor)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

class AbstractPropagator(AbstractOptical):
    _propagation_buffer:torch.Tensor
    @property
    def propagation_buffer(self):
        return self._propagation_buffer
    def _register_propagation_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device:
            buffer = buffer.to(self.device)
        if buffer.dtype not in (self.accuracy.tensor_float, self.accuracy.tensor_complex):
            if torch.is_complex(buffer): buffer = buffer.to(self.accuracy.tensor_complex)
            else: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_propagation_buffer'): self.accuracy.disconnect(self._propagation_buffer)
        self.register_buffer('_propagation_buffer', buffer)
        self.accuracy.connect(self._propagation_buffer)
    def _recalc_propagation_buffer(self):
        raise NotImplementedError
    @property
    def device(self):
        if hasattr(self, '_propagation_buffer'):
            return self._propagation_buffer.device
        else:
            return super().device

    def _change_distance(self):
        pass
    _distance:Param[float]
    @property
    def distance(self):
        return self._distance.value
    @distance.setter
    def distance(self, meters:float):
        self._distance.value = meters

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, distance:float, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, logger=logger)
        self._distance = Param[float](self._change_distance)
        self.distance = distance

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)

class AbstractInhomogeneity(AbstractOptical):
    space_reflection:SpaceParam[float]
    def _change_space_reflection(self):
        pass

    space_absorption:SpaceParam[float]
    def _change_space_absorption(self):
        pass

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, space_reflection:FloatS, space_absorption:FloatS, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, logger=logger)
        self.space_reflection = SpaceParam[float](self._change_space_reflection, group=self.wavelength.group)
        self.space_absorption = SpaceParam[float](self._change_space_absorption, group=self.wavelength.group)
        self.space_reflection.set(space_reflection)
        self.space_absorption.set(space_absorption)
        self.accuracy.connect(self.space_reflection.tensor)
        self.accuracy.connect(self.space_absorption.tensor)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

class AbstractMask(AbstractInhomogeneity):
    _mask_buffer:torch.Tensor
    def _register_mask_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device: buffer = buffer.to(self.device)
        if buffer.dtype not in (self.accuracy.tensor_float, self.accuracy.tensor_complex):
            if torch.is_complex(buffer): buffer = buffer.to(self.accuracy.tensor_complex)
            else: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_mask_buffer'): self.accuracy.disconnect(self._mask_buffer)
        self.register_buffer('_mask_buffer', buffer)
        self.accuracy.connect(self._mask_buffer)
    def _recalc_mask_buffer(self):
        raise NotImplementedError
    @property
    def device(self):
        if hasattr(self, '_mask_buffer'):
            return self._mask_buffer.device
        else: return super().device

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, space_reflection:FloatS, space_absorption:FloatS, logger:Logger=None):
        super().__init__(pixels, length, wavelength, reflection, absorption, space_reflection, space_absorption, logger=logger)

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)
        field = torch.nn.functional.pad(field, self._paddings)
        field = field * self._mask_buffer
        field = torch.nn.functional.pad(field, self._unpaddings)
        field = interpolate(field, (self.pixels.output.x, self.pixels.output.y), mode=self.interpolation.mode)
        return field
