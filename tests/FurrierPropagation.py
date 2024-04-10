import torch

from .uniersal.propagator import rectangle

from elements.propagation import FurrierPropagation



def rectangle_diffraction(nx:int, ny:int, Nz:int=100):
    model = FurrierPropagation(512, 10.0E-3, 500.0E-9, 1.0, 0.0, 1000.0E-3)
    plot = rectangle(model, Nz, nx_max=nx, ny_max=ny)
    plot.show()

if __name__ == '__main__':
    rectangle_diffraction(3, 4, 100)
