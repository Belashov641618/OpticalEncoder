import torch
from typing import Callable, Any, Optional
from utilities import *

# Элементы
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
    def _paddings_difference(self):
        return self._padding_x + self._unpadding_x, self._padding_x + self._unpadding_x, self._padding_y + self._unpadding_y, self._padding_y + self._unpadding_y
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

        self.pixels = IOParams[int](change_input=self._change_pixels_input, change_output=self._change_pixels_output, change=(self._change_pixels, self._reset_add_pixels_))
        self.length = IOParams[float](change_input=self._change_length_input, change_output=self._change_length_output, change=(self._change_length, self._reset_add_pixels_))

        self.pixels.set(pixels)
        self.length.set(length)

        self._add_pixels = XYParams[int]()
        self._add_pixels.set(0)

        self.interpolation = InterpolateMode(InterpolateModes.bilinear)

        if logger is None:
            logger = Logger(False, prefix='Deleted')
        self.logger = logger

    def forward(self, *args, **kwargs):
        self.delayed.launch()

    def memory(self):
        pass

class AbstractSpectral(AbstractElement):
    _optical_group: SpaceParamGroup
    @property
    def optical_group(self):
        return self._optical_group
    @optical_group.setter
    def optical_group(self, group: SpaceParamGroup):
        group.merge(self._optical_group)
        self._optical_group = group

    wavelength: SpaceParam[float]
    def _change_wavelength(self):
        pass

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, logger:Logger=None):
        super().__init__(pixels, length, logger=logger)
        self.wavelength = SpaceParam[float](self._change_wavelength)
        self.wavelength.set(wavelength)
        self.accuracy.connect(self.wavelength.tensor)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

class AbstractOptical(AbstractSpectral):
    reflection:SpaceParam[float]
    def _change_reflection(self):
        pass

    absorption:SpaceParam[float]
    def _change_absorption(self):
        pass

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, logger:Logger=None):
        super().__init__(pixels, length, wavelength, logger=logger)
        self.reflection = SpaceParam[float](self._change_reflection, group=self.wavelength.group)
        self.absorption = SpaceParam[float](self._change_absorption, group=self.wavelength.group)
        self.reflection.set(reflection)
        self.absorption.set(absorption)
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
        print("AbstractPropagator buffer registering")
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

class AbstractModulator(AbstractElement):
    def to(self, *args, **kwargs):
        print(f"AbstractModulator.to triggered with: {args}, {kwargs}")
        return super().to(*args, **kwargs)
    
    mask_pixels:XYParams[int]
    _mask_parameters:torch.nn.Parameter
    def _register_mask_parameters(self, parameters:torch.Tensor):
        if parameters.device != self.device: parameters = parameters.to(self.device)
        if parameters.dtype not in (self.accuracy.tensor_float, self.accuracy.tensor_complex):
            if torch.is_complex(parameters): parameters = parameters.to(self.accuracy.tensor_complex)
            else: parameters = parameters.to(self.accuracy.tensor_float)
        if hasattr(self, '_mask_parameters'):
            self._mask_parameters.copy_(parameters)
        else:
            self._mask_parameters = torch.nn.Parameter(parameters)
            self.accuracy.connect(self._mask_parameters)
    def _recalc_mask_parameters(self):
        parameters = torch.normal(0., 1.0, (self.mask_pixels.x, self.mask_pixels.y))
        self._register_mask_parameters(parameters)
    def _reset_mask_parameters(self):
        self.delayed.add(self._recalc_mask_parameters)
    def _normalized(self) -> torch.Tensor:
        return torch.sigmoid(self._mask_parameters.unsqueeze(0).unsqueeze(0))
    def _multiplier(self) -> torch.Tensor:
        raise NotImplementedError
    @property
    def properties(self):
        return self._mask_parameters.clone().detach().cpu()
    @property
    def device(self):
        if hasattr(self, '_mask_parameters'):
            return self._mask_parameters.device
        else: return super().device

    def __init__(self, pixels:IntIO, length:FloatIO, mask_pixels:IntXY, logger:Logger=None):
        super().__init__(pixels, length, logger=logger)
        self.mask_pixels = XYParams[int](self._reset_mask_parameters).set(mask_pixels)
        self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)
        field = fix_complex(field)
        field = torch.nn.functional.pad(field, self._paddings_difference)
        field = field * fix_complex(self._multiplier())
        field = interpolate(field, (self.pixels.output.x, self.pixels.output.y), mode=self.interpolation.mode)
        return field

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
    def mask(self):
        self.delayed.launch()
        return self._mask_buffer.clone().detach().squeeze().cpu()
    @property
    def device(self):
        if hasattr(self, '_mask_buffer'):
            return self._mask_buffer.device
        else: return super().device

    @property
    def _total_pixels_x(self):
        return super()._total_pixels_x + 2*self._unpadding_x
    @property
    def _total_pixels_y(self):
        return super()._total_pixels_y + 2*self._unpadding_y

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, reflection:FloatS, absorption:FloatS, space_reflection:FloatS, space_absorption:FloatS, logger:Logger=None, finalize:bool=True):
        super().__init__(pixels, length, wavelength, reflection, absorption, space_reflection, space_absorption, logger=logger)
        if finalize: self.delayed.launch()

    def forward(self, field:torch.Tensor, *args, **kwargs):
        super().forward(*args, **kwargs)
        field = fix_complex(field)
        field = torch.nn.functional.pad(field, self._paddings_difference)
        field = field * fix_complex(self._mask_buffer)
        field = interpolate(field, (self.pixels.output.x, self.pixels.output.y), mode=self.interpolation.mode)
        return field

class AbstractDetectors(AbstractSpectral):
    _spectral_buffer:torch.Tensor
    _spectral_filter:Param[filters.Filter]
    def _register_spectral_buffer(self, buffer:torch.Tensor):
        if buffer.device != self.device: buffer = buffer.to(self.device)
        if buffer.dtype != self.accuracy.tensor_float: buffer = buffer.to(self.accuracy.tensor_float)
        if hasattr(self, '_spectral_buffer'): self.accuracy.disconnect(self._spectral_buffer)
        self.register_buffer('_spectral_buffer', buffer)
        self.accuracy.connect(self._spectral_buffer)
    def _recalc_spectral_buffer(self):
        self._register_spectral_buffer(self._spectral_filter.value(self.wavelength.tensor))
    def _attach_recalc_spectral_buffer(self):
        self.delayed.add(self._recalc_spectral_buffer)
    @property
    def spectral(self):
        self.delayed.launch()
        return self._spectral_buffer.clone().detach().cpu()
    @spectral.setter
    def spectral(self, filter:filters.Filter):
        self._spectral_filter.value = filter
    @property
    def device(self):
        if hasattr(self, '_spectral_buffer'):
            return self._spectral_buffer.device
        else:
            return super().device

    def _change_wavelength(self):
        super()._change_wavelength()
        self.delayed.add(self._recalc_spectral_buffer)

    def __init__(self, pixels:IntIO, length:FloatIO, wavelength:FloatS, spectral_filter:filters.Filter, logger:Logger=None):
        super().__init__(pixels, length, wavelength, logger=logger)
        self._spectral_filter = Param[filters.Filter](self._attach_recalc_spectral_buffer)
        self._spectral_filter.set(spectral_filter)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

# Остальное
class AbstractWrapper(torch.nn.Module):
    delayed:DelayedFunctions
    _forward:Optional[Callable[[torch.Tensor, Any, ...], torch.Tensor]]
    def __init__(self, forward:Callable[[torch.Tensor, Any, ...], torch.Tensor]=None):
        super().__init__()
        self.delayed = DelayedFunctions()
        self._forward = forward
    def attach_forward(self, forward:Callable[[torch.Tensor, Any, ...], torch.Tensor]):
        self._forward = forward
    def forward(self, field:torch.Tensor, *args, **kwargs):
        raise NotImplementedError




