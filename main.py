from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
import torch


if __name__ == '__main__':
    # dynamic_approximation()
    rectangle_diffraction(3, 6, 100)

    from elements.composition import CompositeModel
    from elements.propagators import FurrierPropagation
    from elements.modulators import Lens
    from tests.Composition import propagation

    N = 512
    length = 1.0E-3
    wavelength = 500.0E-9
    focus = 100.0E-3

    propagation1 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, 2*focus)
    lens = Lens(N, length, wavelength, 1.5, 0.0, 1.0, 0.0, focus)
    propagation2 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, 2*focus)

    model = CompositeModel(propagation1, lens, propagation2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    propagation(model, 'Flowers')