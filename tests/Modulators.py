import torch
import numpy

from elements.abstracts import AbstractMask
from elements.propagators import FurrierPropagation
from elements.modulators import Lens

from utilities import *
from parameters import FigureWidthHeight, FontLibrary

def lens():
    N = 100
    length = 1.0E-3
    wavelength = 500.0E-9
    focus = 100.0E-3

    lens_module = Lens(N, length, wavelength, 1.5, 0.0, 1.0, 0.0, focus)

    random_filed = torch.rand((1,1,N,N)).to(torch.complex64)
    result = lens_module.forward(random_filed).squeeze().cpu()

    plot = TiledPlot(*FigureWidthHeight)
    plot.FontLibrary = FontLibrary
    plot.title(f'Ядро модуля линзы')

    kwargs = {'aspect':'auto'}

    axes = plot.axes.add(0, 0)
    axes.imshow(lens_module.mask.abs(), **kwargs)
    plot.graph.title('Амплитуда')

    axes = plot.axes.add(1, 0)
    axes.imshow(lens_module.mask.angle(), **kwargs)
    plot.graph.title('Фаза')

    axes = plot.axes.add(0, 1)
    axes.imshow(result.abs(), **kwargs)
    plot.graph.title('Амплитуда')

    axes = plot.axes.add(1, 1)
    axes.imshow(result.angle(), **kwargs)
    plot.graph.title('Фаза')

    plot.show()

def lens_focus(module:Lens, steps:int=100):
    pixels = (module.pixels.output.x, module.pixels.output.y, module.pixels.output.x, module.pixels.output.y)
    length = (module.length.output.x, module.length.output.y, module.length.output.x, module.length.output.y)
    wavelength = module.wavelength.effective

    propagator = FurrierPropagation(pixels, length, wavelength, 1.0, 0.0, 2*module.focus.effective, border_ratio=1.0)
    distances = numpy.linspace(0, 2*module.focus.effective, steps)[1:]

    initial = torch.ones((1, 1, pixels[0], pixels[1]))
    initial = module.forward(initial)

    result = torch.zeros((steps, pixels[0]), dtype=torch.complex64)
    result[0] = initial[:,:,pixels[0]//2,:].squeeze().cpu()

    for i, distance in enumerate(distances, 1):
        propagator.distance = distance
        result[i] = propagator.forward(initial)[:,:,pixels[0]//2,:].squeeze().cpu()

    result = result.swapdims(0,1)

    plot = TiledPlot(*FigureWidthHeight)
    plot.FontLibrary = FontLibrary
    plot.width_to_height(numpy.log10(2*module.focus.effective/length[0]) + 1)

    axes = plot.axes.add(0,0)
    axes.imshow(result.abs(), aspect='auto', extent=[0, 2*module.focus.effective, -length[0]/2, +length[0]/2])

    plot.show()