import torch

from typing import Union

from utilities import *
from parameters import FigureWidthHeight, FontLibrary
from elements.composition import CompositeModel

def propagation(model:CompositeModel, field:Union[torch.Tensor, LiteralDataSet]):
    if not isinstance(field, torch.Tensor): field, _ = Dataset.single(field, model.element(0).pixels.input.x, model.element(0).pixels.input.y, model.dtype)
    while len(field.size()) < 4: field = field.unsqueeze(0)
    field = field.to(model.dtype)

    planes = model.planes(field, model.element(0).pixels.input.x, model.element(0).pixels.input.y)
    profile = model.profile(field, pixels_xy=model.element(0).pixels.input.x, pixels_z=model.element(0).pixels.input.x, reduce='mean')
    if hasattr(profile, 'abs'): profile = profile.abs()

    cols = planes.size(0)

    plot = TiledPlot(*FigureWidthHeight)
    plot.FontLibrary = FontLibrary
    plot.title('Распространение света через модель')

    x_format, x_union = engineering.separatedformatter(model.total_length, 'м', 2)
    y_format, y_union = engineering.separatedformatter(model.element(0).length.input.x, 'м', 2)
    kwargs = {'aspect':'auto'}
    axes = plot.axes.add((0,0), (cols-1,0))
    axes.imshow(dimension_normalization(profile, dim=1), **kwargs, extent=[0, model.total_length, -model.element(0).length.input.x/2, +model.element(0).length.input.x/2])
    axes.xaxis.set_major_formatter(x_format)
    axes.yaxis.set_major_formatter(y_format)
    plot.graph.label.x(x_union)
    plot.graph.label.y(y_union)

    plot.description.row.right('Срез',      0)
    plot.description.row.right('Амплитуда', 1)
    plot.description.row.right('Фаза',      2)

    for col, plane in enumerate(planes):
        extent = [-model.element(col).length.input.x/2, +model.element(col).length.input.x/2, -model.element(col).length.input.y/2, +model.element(col).length.input.y/2] if col != model.count else [-model.element(col-1).length.output.x/2, +model.element(col-1).length.output.x/2, -model.element(col-1).length.output.y/2, +model.element(col-1).length.output.y/2]
        x_format, x_union = engineering.separatedformatter(model.element(col).length.input.x if col != model.count else model.element(col-1).length.output.x, 'м', 2)
        y_format, y_union = engineering.separatedformatter(model.element(col).length.input.y if col != model.count else model.element(col-1).length.output.y, 'м', 2)

        axes = plot.axes.add(col, 1)
        axes.imshow(plane.abs(), **kwargs, extent=extent)
        axes.xaxis.set_major_formatter(x_format)
        axes.yaxis.set_major_formatter(y_format)
        plot.graph.label.x(x_union)
        plot.graph.label.y(y_union)

        axes = plot.axes.add(col, 2)
        axes.imshow(plane.angle(), **kwargs, extent=extent)
        axes.xaxis.set_major_formatter(x_format)
        axes.yaxis.set_major_formatter(y_format)
        plot.graph.label.x(x_union)
        plot.graph.label.y(y_union)
    plot.show()
