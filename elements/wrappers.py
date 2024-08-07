from __future__ import annotations
import numpy
import torch

from typing import Union, Iterable, Optional
from matplotlib.patches import Circle

from elements.abstracts import AbstractWrapper, AbstractEncoderDecoder
from utilities.noise import gaussian, GaussianNormalizer
from utilities import *
from parameters import FigureWidthHeight, FontLibrary

class CudaMemoryChunker(AbstractWrapper):
    """
    НЕ ИСПОЛЬЗУЙ ЭТОТ КЛАСС, ОН БОЛЬШЕ НЕ РАБОТАЕТ
    """
    _chunks:int
    _sub_chunks:int
    def __init__(self, chunks:int=1, sub_chunks:int=1):
        super().__init__()
        self._chunks:int = chunks
        self._sub_chunks:int = sub_chunks

    def forward(self, field:torch.Tensor, *args, **kwargs):
        field = fix_complex(field)
        try:
            results = []
            for part_ in torch.chunk(field, self._chunks, dim=0):
                results_ = []
                for part in torch.chunk(part_, self._sub_chunks, dim=1):
                    results__ = self._forward(part, *args, **kwargs)
                    results_.append(results__)
                results.append(torch.cat(results_, dim=1))
            field = torch.cat(results, dim=0)
            return field
        except torch.cuda.OutOfMemoryError as error:
            print('Memory error happened')
            if self._chunks < field.shape[0]:
                self._chunks += 1
                print(f'Splitting batches to {self._chunks} chunks')
            elif self._sub_chunks < field.shape[1]:
                self._sub_chunks += 1
                print(f'Splitting channels to {self._sub_chunks} chunks')
            else:
                print(f'Nothing to do with memory error')
                raise error
        return self.forward(field, *args, **kwargs)


class FullyIncoherentEncoder(AbstractEncoderDecoder):
    def _init__(self, parent:FullyIncoherent):
        super().__init__(parent)
    def forward(self, field:torch.Tensor, *args, **kwargs):
        self._parent.delayed.launch()
        field = field * torch.exp(2j * torch.pi * torch.rand((1,field.shape[1]*self._parent.samples,*field.shape[2:]), device=field.device))
        return field
class FullyIncoherentDecoder(AbstractEncoderDecoder):
    def __init__(self, parent:FullyIncoherent):
        super().__init__(parent)
    def forward(self, field:torch.Tensor, *args, **kwargs):
        self._parent.delayed.launch()
        field = torch.reshape(field, (field.shape[0], -1, self._parent.samples, *field.shape[2:]))
        field = torch.mean(torch.abs(field)**2, dim=2) * torch.exp(1j * torch.angle(field[:, :, 0, :, :]))
        return field
class FullyIncoherent(AbstractWrapper):
    def __init__(self, samples:int):
        super().__init__(init=False)
        self.samples = samples
        self.encoder = FullyIncoherentEncoder(self)
        self.decoder = FullyIncoherentDecoder(self)

class IncoherentEncoder(AbstractEncoderDecoder):
    _generator:GaussianNormalizer
    _parent:Incoherent
    def __init__(self, parent:Incoherent):
        super().__init__(parent)
        self._generator = getattr(self._parent, "_generator")
    def forward(self, field:torch.Tensor, *args, **kwargs):
        self._parent.delayed.launch()
        field = fix_complex(field)

        self._parent.channels = field.size(1)
        Nxy = (field.size(2), field.size(3))
        field = field.reshape(-1, self._parent.channels, 1, *Nxy)

        field = field * torch.exp(2j * torch.pi * self._generator.sample())
        field = field.reshape(-1, self._parent.channels * self._parent.samples, *Nxy)

        return field
class IncoherentDecoder(AbstractEncoderDecoder):
    _parent: Incoherent
    def __init__(self, parent:Incoherent):
        super().__init__(parent)
    def forward(self, field:torch.Tensor, *args, **kwargs):
        self._parent.delayed.launch()
        field = fix_complex(field)

        Nxy = (field.size(2), field.size(3))
        field = field.reshape(-1, self._parent.channels, self._parent.samples, *Nxy)
        field = torch.mean(field.abs() ** 2, dim=2) * torch.exp(1j * torch.angle(field[:, :, 0, :, :]))
        return field
class Incoherent(AbstractWrapper):
    _generator:GaussianNormalizer
    def _reset_generator(self):
        self._generator = GaussianNormalizer(gaussian(
            (self.time_coherence, self.spatial_coherence, self.spatial_coherence),
            (self.samples, self.pixels.x, self.pixels.y),
            ((0, self.time),(0, self.length.x),(0,self.length.y)),
            device=self._generator.device, generator=True), 100)
        self._generator.optimize()
    def _delayed_generator_reset(self):
        self.delayed.add(self._reset_generator)

    _spatial_coherence:Param[float]
    @property
    def spatial_coherence(self):
        return self._spatial_coherence.value
    _time_coherence:Param[float]
    @property
    def time_coherence(self):
        return self._time_coherence.value
    _time:Param[float]
    @property
    def time(self):
        return self._time.value
    _samples:Param[int]
    @property
    def samples(self):
        return self._samples.value
    pixels:XYParams[int]
    length:XYParams[float]

    def sample(self):
        self.delayed.launch()
        return self._generator.sample()

    channels:Optional[int]

    def __init__(self, spatial_coherence:float, time_coherence:float, time:float, samples:int, pixels:IntXY, length:FloatXY):
        super().__init__(init=False)
        self._spatial_coherence = Param[float](self._delayed_generator_reset).set(spatial_coherence)
        self._time_coherence = Param[float](self._delayed_generator_reset).set(time_coherence)
        self._time = Param[float](self._delayed_generator_reset).set(time)
        self._samples = Param[int](self._delayed_generator_reset).set(samples)
        self.pixels = XYParams[int](change=self._delayed_generator_reset).set(pixels)
        self.length = XYParams[float](change=self._delayed_generator_reset).set(length)
        self._channels = None

        self._generator = GaussianNormalizer(gaussian(
            (self.time_coherence, self.spatial_coherence, self.spatial_coherence),
            (self.samples, self.pixels.x, self.pixels.y),
            ((0, self.time), (0, self.length.x), (0, self.length.y)),
            generator=True), 100)

        self.encoder = IncoherentEncoder(self)
        self.decoder = IncoherentDecoder(self)

    def show(self):
        with torch.no_grad():
            sample = self._generator.sample()
            dist, vals = self._generator.distribution_fixed(True)
            dist_, vals_ = self._generator.distribution(True)
            autocorr = autocorrelation(sample, dims=(1,2), mean_dim=0)
            slice_xy = sample[0]
            slice_xt = sample[:,:,0].swapdims(0,1)
            slice_yt = sample[:,0,:].swapdims(0,1)
            dist, vals, dist_, vals_, autocorr, slice_xy, slice_xt, slice_yt = dist.cpu(), vals.cpu(), dist_.cpu(), vals_.cpu(), autocorr.cpu(), slice_xy.cpu(), slice_xt.cpu(), slice_yt.cpu()

        radius, position = correlation_circle(autocorr, ((0, self.length.x), (0, self.length.y)))

        plot = TiledPlot(*FigureWidthHeight)
        plot.FontLibrary = FontLibrary
        plot.FontLibrary.MultiplyFontSize(0.7)
        plot.title('Параметры некогерентности')

        kwargs = {'aspect':'auto'}

        axes = plot.axes.add(0,0)
        axes.imshow(slice_xy, **kwargs, extent=[0, self.length.x, 0, self.length.y])
        plot.graph.title('Срез xy')

        axes = plot.axes.add(1,0)
        axes.imshow(slice_xt, **kwargs, extent=[0, self.length.x, 0, self.time])
        plot.graph.title('Срез xt')

        axes = plot.axes.add(2,0)
        axes.imshow(slice_yt, **kwargs, extent=[0, self.length.y, 0, self.time])
        plot.graph.title('Срез yt')

        axes = plot.axes.add(0,1)
        axes.imshow(autocorr, **kwargs, extent=[0, self.length.x, 0, self.length.y])
        axes.add_patch(Circle(position, radius, fill=False, linestyle='--', color='maroon'))
        plot.graph.title('Автокорреляция xy')

        axes = plot.axes.add(1,1)
        axes.grid(True)
        axes.plot(numpy.linspace(0,self.length.x,autocorr.size(0)), autocorr[:,autocorr.size(1)//2])
        axes.axvline(position[0]-radius, linestyle='--', color='maroon')
        axes.axvline(position[0]+radius, linestyle='--', color='maroon')
        plot.graph.title('Срез автокорреляции')

        axes = plot.axes.add(2,1)
        axes.grid(True)
        axes.plot(vals, dist)
        axes.plot(vals_, dist_, linestyle='--')
        plot.graph.title('Распределение')

        plot.show()