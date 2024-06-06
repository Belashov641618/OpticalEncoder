from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
from tests.Modulators import lens_focus
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    from utilities import *
    dataset = Dataset('CIFAR10', 64, 512, 512, torch.complex6)
    dataset.train


