from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
from tests.Modulators import lens_focus
import torch
import matplotlib.pyplot as plt
plt.interactive(False)


if __name__ == '__main__':
    # dynamic_approximation()
    # rectangle_diffraction(3, 6, 100)

    from elements.composition import CompositeModel
    from elements.propagators import FurrierPropagation
    from elements.modulators import Lens, PhaseModulator, AmplitudeModulator
    from tests.Composition import propagation
    from elements.wrappers import Incoherent, CudaMemoryChunker

    N = 180
    length = 5.0E-3
    wavelength = 500.0E-9
    focus = 300.0E-3
    scale = 1
    # 1/a + 1/b = 1/f -> b = af / (a - f)
    # a/b = n -> n = (a - f) / f -> a = f(n + 1)
    # b = f^2(n+1) / fn = f (n + 1) / n
    propagation1 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, focus*(scale + 1), border_ratio=1.0)
    lens1 = Lens(N, length, wavelength, 1.5, 0.0, 1.0, 0.0, focus)
    propagation2 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, focus*(scale + 1)/scale, border_ratio=1.0)
    # propagation3 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, 2 * focus, border_ratio=1.0)
    # lens2 = Lens(N, length, wavelength, 1.5, 0.0, 1.0, 0.0, focus)
    # propagation4 = FurrierPropagation(N, length, wavelength, 1.0, 0.0, 2 * focus, border_ratio=1.0)

    model = CompositeModel(propagation1, lens1, propagation2)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # chunker = CudaMemoryChunker()
    # model.wrap(chunker)
    # incoherent = Incoherent(length / 15, 0.001, 1.0, 50, N, length)
    # model.wrap(incoherent)

    propagation(model, 'MNIST')