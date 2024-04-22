import torch

from math import sqrt
from typing import Union, Literal

from elements.abstracts import AbstractModulator
from elements.composition import CompositeModel
from elements.wrappers import Incoherent, CudaMemoryChunker
from elements.propagators import FurrierPropagation
from elements.modulators import Lens, PhaseModulator, AmplitudeModulator, PerfectHeightModulator
from utilities import *

_Modulators = Union[PhaseModulator, AmplitudeModulator, PerfectHeightModulator]
_ModulatorType = Literal['Phase', 'Amplitude', 'Height']

class DiffractionNeuralNetwork(CompositeModel):
    incoherent: Incoherent
    chunker: CudaMemoryChunker

    propagator: FurrierPropagation
    modulators: tuple[_Modulators, ...]

    wavelength: SpaceParam[float]
    def _change_wavelength(self):
        pass
    def _group_parameters(self):
        pass

    def __init__(self,
                 spatial_coherence: float, time_coherence: float, relaxation_time: float, time_samples: int,
                 wavelength: FloatS, distance_ration: float, sub_pixels: int,
                 modulator_unit_length: float, modulator_units: int, modulators_count: int, modulator_type:_ModulatorType,
                 ):
        super().__init__()
        self.wavelength = SpaceParam[float](self._change_wavelength).set(wavelength)

        length = modulator_unit_length * modulator_units
        pixels = modulators_count * sub_pixels

        distance = length * sqrt((...))

        self.incoherent = Incoherent(spatial_coherence, time_coherence, relaxation_time, time_samples, pixels, length)
        self.chunker = CudaMemoryChunker()

        self.propagator = FurrierPropagation(pixels, length, wavelength, 1.0, 0.0, distance, 0.7)