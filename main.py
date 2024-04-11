from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
import torch

if __name__ == '__main__':
    # dynamic_approximation()
    rectangle_diffraction(3, 6, 100)