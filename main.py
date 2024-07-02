from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
from tests.Modulators import lens_focus
import torch
import matplotlib.pyplot as plt
from belashovplot import TiledPlot

if __name__ == '__main__':
    from utilities import *
    from utilities.datasets import LiteralDataSet
    dataset = Dataset('MNIST', 10, 512, 512, torch.complex64)
    image, image_ = next(iter(dataset.train))
    print(image.shape, image_.shape)




