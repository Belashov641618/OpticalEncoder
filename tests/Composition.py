import torch

from typing import Union

from utilities import *
from parameters import FigureWidthHeight, FontLibrary
from elements.composition import CompositeModel

def propagation(model:CompositeModel, field:Union[torch.Tensor, LiteralDataSet]):
    if isinstance(field, LiteralDataSet):
        field, _ = Dataset.single(LiteralDataSet, model.pixels[0].x, model.pixels[0].y, model.dtype)
    while len(field.size()) < 4:
        field = field.unsqueeze(0)
    field = field.to(model.dtype)

    planes = model.planes(field, 512, 512)
    profile = model.profile(field, pixels_xy=512, pixels_z=512, reduce='select')

    cols = planes.size(0)

    plot = TiledPlot(*FigureWidthHeight)
    plot.FontLibrary = FontLibrary
    plot.title('Распространение света через модель')

    kwargs = {'aspect':'auto'}
    axes = plot.axes.add((0,0), (cols-1,0))
    axes.imshow(profile, **kwargs)

    for col, plane in enumerate(planes):
        axes = plot.axes.add(col, 1)
        axes.imshow(plane, **kwargs)

    plot.show()
