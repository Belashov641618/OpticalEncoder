import torch

from .uniersal.propagator import rectangle

from elements.propagation import FurrierPropagation



def rectangle_diffraction(nx:int, ny:int, Nz:int=100):
    model = FurrierPropagation(1024, 2.0E-3, 500.0E-9, 1.0, 0.0, 10.0E-3)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    plot = rectangle(model, Nz, nx_max=nx, ny_max=ny)
    plot.show()
