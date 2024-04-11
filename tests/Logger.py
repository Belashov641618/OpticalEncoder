import torch

from utilities import Logger, DynamicApproximation
from time import sleep
from belashovplot import TiledPlot
from parameters import *

def dynamic_approximation(N:int=20, samples:int=3):
    print('Testing DynamicApproximation class with inner models and mse loss')

    x_array = torch.arange(N)

    approx = DynamicApproximation(N)
    approx.mse()
    plot = TiledPlot(*FigureWidthHeight)
    def construct(row:int, array:torch.Tensor):
        prev = 0
        array += torch.rand_like(array) * array.max() / 10

        for col, batch in enumerate(torch.chunk(array, samples)):
            for element in batch:
                approx.append(element)
            axes = plot.axes.add(col, row)
            axes.grid(True)
            axes.scatter(x_array[:prev+batch.size(0)], array[:prev+batch.size(0)], s=1.0)
            axes.plot(x_array, approx.predict(x_array).detach(), linestyle='--', color='maroon')
            prev += batch.size(0)
        approx.reset()

    approx.linear()
    a = 1.0
    b = 2.0
    construct(0, a*x_array + b)
    print('Linear done')

    approx.linear()
    c = 3.0
    construct(1, a*x_array**2 + b*x_array + c)
    print('Square done')

    approx.linear()
    d = 4.0
    construct(2, a*x_array**3 + b*x_array**2 + c*x_array + d)
    print('Cubic done')

    plot.show()







def cycles():
    pass