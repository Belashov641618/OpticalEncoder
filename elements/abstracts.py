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

    def __init__(self, pixels:IntIO, length:FloatIO, logger:Logger=None):
        super().__init__()
        self.accuracy = Accuracy()
        self.delayed = DelayedFunctions()

        self.pixels = IOParams[int](change_input=self._change_pixels_input, change_output=self._change_pixels_output, change=self._change_pixels)
        self.length = IOParams[float](change_input=self._change_length_input, change_output=self._change_length_output, change=self._change_length)

        self.pixels.set(pixels)
        self.length.set(length)

        if logger is None:
            logger = Logger(False, prefix='Deleted')
        self.logger = logger

    def forward(self, *args, **kwargs):
        self.delayed.launch()

    def memory(self):
        pass

class AbstractOptical(AbstractElement):
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
        self.reflection = SpaceParam[float](self._change_reflection)
        self.absorption = SpaceParam[float](self._change_absorption)
        self.wavelength.set(wavelength)
        self.reflection.set(reflection)
        self.absorption.set(absorption)
        self.wavelength.connect(self.reflection.tensor, self.absorption.tensor)
        self.reflection.connect(self.wavelength.tensor, self.absorption.tensor)
        self.absorption.connect(self.wavelength.tensor, self.reflection.tensor)
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

