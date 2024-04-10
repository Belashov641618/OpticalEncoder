import numpy
import torch

from belashovplot import TiledPlot

from elements.abstracts import AbstractPropagator
from utilities import Logger, engineering, shifted_log10
from parameters import FigureWidthHeight, FontLibrary

def rectangle(model:AbstractPropagator, Nz:int, width:float=None, height:float=None, nx_max:int=3, ny_max:int=3, plot:TiledPlot=None, logger:Logger=None):
    if logger is None:
        logger = Logger(False)

    pixels_x = model.pixels.input.x
    pixels_y = model.pixels.input.y
    length_x = model.length.input.x
    length_y = model.length.input.y

    wavelength = model.wavelength.effective
    reflection = model.reflection.effective
    distance = model.distance

    if width is None and nx_max is not None:
        period_x = numpy.arcsin(0.5 * length_x / distance) / nx_max
        width = wavelength * reflection / period_x
    elif nx_max is None and width is not None:
        period_x = wavelength*reflection / width
        nx_max = int(numpy.arcsin(0.5*length_x / distance) / period_x)
    else: raise AttributeError('Для оси X необходимо указать либо размер щели width либо количество дифракционных максимумов nx_max')

    if height is None and ny_max is not None:
        period_y = numpy.arcsin(0.5 * length_y / distance) / ny_max
        height = wavelength * reflection / period_y
    elif ny_max is None and height is not None:
        period_y = wavelength*reflection / height
        ny_max = int(numpy.arcsin(0.5*length_y / distance) / period_y)
    else: raise AttributeError('Для оси Y необходимо указать либо размер щели height либо количество дифракционных максимумов ny_max')

    if plot is None:
        plot = TiledPlot(*FigureWidthHeight)
        plot.FontLibrary = FontLibrary
        plot.title(f'Тестирование модуля распространения {type(model).__name__} в щелевой дифракции')

    with torch.no_grad():
        x_mesh, y_mesh = torch.meshgrid(torch.linspace(-length_x/2, +length_x/2, pixels_x, device=model.device, dtype=model.accuracy.tensor_float), torch.linspace(-length_y/2, +length_y/2, pixels_y, device=model.device, dtype=model.accuracy.tensor_float), indexing='ij')
        mask = (-width/2 <= x_mesh) * (x_mesh <= +width/2) * (-height/2 <= y_mesh) * (y_mesh <= +height/2)
        if torch.sum(mask) <= 4: raise ValueError(f'Параметры щели width и/или height слишком малы\nwidth: {width} из критических: {2*length_x/pixels_x}\nheight: {height} из критических: {2*length_y/pixels_y}')
        field = mask * torch.ones((pixels_x, pixels_y), dtype=model.accuracy.tensor_complex, device=model.device)
        field = field.expand(1, model.wavelength.length, -1, -1)
        result = model.forward(field)
        cutX = torch.zeros((Nz, model.pixels.output.y), dtype=model.accuracy.tensor_complex, device=model.device)
        cutY = torch.zeros((Nz, model.pixels.output.x), dtype=model.accuracy.tensor_complex, device=model.device)

        for i, dist in enumerate(numpy.linspace(0, distance, Nz)):
            model.distance = dist
            temp = model.forward(field).squeeze()
            cutX[i] = temp[model.pixels.output.x // 2, :]
            cutY[i] = temp[:, model.pixels.output.y // 2]

        field = field.squeeze().abs().cpu()
        result = result.squeeze().abs().cpu()
        cutX = cutX.squeeze().abs().cpu()
        cutY = cutY.squeeze().abs().cpu()

    format_x, letter_x = engineering.separatedformatter(length_x / 2, 'м')
    format_y, letter_y = engineering.separatedformatter(length_y / 2, 'м')
    format_z, letter_z = engineering.separatedformatter(distance,     'м')

    (field, result, cutX, cutY), shift = shifted_log10(field, result, cutX, cutY, average=0.5, reduce='max')
    plot.description.bottom(f"К амплитудам была применена функция экспозиции: $log_{{10}}(x+{round(shift, 2)})$")

    kwargs = {'aspect':'auto'}
    axes = plot.axes.add(0, 0)
    plot.graph.title('Начальное поле')
    axes.imshow(field, **kwargs, extent=[-length_x/2,+length_x/2,-length_y/2,+length_y/2])
    axes.xaxis.set_major_formatter(format_x)
    plot.graph.label.x(letter_x)
    axes.yaxis.set_major_formatter(format_y)
    plot.graph.label.y(letter_y)

    axes = plot.axes.add(1, 0)
    plot.graph.title('Срез YZ')
    axes.imshow(cutY, **kwargs, extent=[-length_y/2,+length_y/2,distance,0])
    axes.xaxis.set_major_formatter(format_y)
    plot.graph.label.x(letter_y)
    axes.yaxis.set_major_formatter(format_z)
    plot.graph.label.y(letter_z)
    for i in range(1, ny_max+1):
        axes.axline((0, length_y/2), slope=+1.0/(i*wavelength/height), alpha=0.5, linestyle='--')
        axes.axline((0, length_y/2), slope=-1.0/(i*wavelength/height), alpha=0.5, linestyle='--')

    axes = plot.axes.add(0, 1)
    plot.graph.title('Срез XZ')
    axes.imshow(cutX, **kwargs, extent=[-length_x/2,+length_x/2,distance,0])
    axes.xaxis.set_major_formatter(format_x)
    plot.graph.label.x(letter_x)
    axes.yaxis.set_major_formatter(format_z)
    plot.graph.label.y(letter_z)
    for i in range(1, nx_max+1):
        axes.axline((0, length_x/2), slope=+1.0/(i*wavelength/width), alpha=0.5, linestyle='--')
        axes.axline((0, length_x/2), slope=-1.0/(i*wavelength/width), alpha=0.5, linestyle='--')

    axes = plot.axes.add(1, 1)
    plot.graph.title('Результат')
    axes.imshow(result, **kwargs, extent=[-length_x/2,+length_x/2,-length_y/2,+length_y/2])
    axes.xaxis.set_major_formatter(format_x)
    plot.graph.label.x(letter_x)
    axes.yaxis.set_major_formatter(format_y)
    plot.graph.label.y(letter_y)

    return plot




