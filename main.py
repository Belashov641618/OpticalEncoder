from tests.FurrierPropagation import rectangle_diffraction
from tests.Logger import dynamic_approximation, cycles
from tests.Modulators import lens_focus
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.interactive(False)


if __name__ == '__main__':
    # dynamic_approximation()
    # rectangle_diffraction(3, 6, 100)

    from elements.composition import CompositeModel, HybridModel
    from elements.propagators import FurrierPropagation
    from elements.modulators import Lens, PhaseModulator, AmplitudeModulator
    from tests.Composition import propagation as propagation_test
    from elements.wrappers import Incoherent, CudaMemoryChunker
    from utilities.datasets import Dataset
    from elements.detectors import ClassificationDetectors
    from utilities.filters import Window, Gaussian

    N = 80
    length = 5.0E-3
    wavelength = 500.0E-9
    focus = 300.0E-3

    propagation = FurrierPropagation(N, length, wavelength, 1.0, 0.0, 2*focus, border_ratio=1.0)
    lens = Lens(N, length, wavelength, 1.5, 0.0, 1.0, 0.0, focus)
    modulator = PhaseModulator(N, length, N)
    optical = CompositeModel(propagation, lens, propagation)
    optical.to('cuda' if torch.cuda.is_available() else 'cpu')
    chunker = CudaMemoryChunker()
    optical.wrap(chunker)
    incoherent = Incoherent(length / 20, 0.001, 1.0, 64, N, length)
    optical.wrap(incoherent)

    detectors_amount = 30

    spectral_filter = Window(centers=wavelength, sizes=300.0E-9)
    detectors_filter = Gaussian((length/30, length/30), (0,0))
    detectors = ClassificationDetectors(N, length, wavelength, detectors_amount, detectors_filter, spectral_filter)
    detectors.to(optical.device)

    from belashovplot import TiledPlot
    from parameters import FigureWidthHeight, FontLibrary
    plot = TiledPlot(*FigureWidthHeight)
    plot.FontLibrary = FontLibrary
    plot.axes.add(0,0).imshow(torch.sum(detectors.filter, dim=0), aspect='auto')
    plot.show()

    layers_structure = [detectors_amount, detectors_amount, 15, 10]
    layers = []
    for nodes0, nodes1 in zip(layers_structure[:-1], layers_structure[1:]):
        layers.append(torch.nn.Linear(nodes0, nodes1))
        layers.append(torch.nn.Sigmoid())
    electronic = torch.nn.Sequential(*layers)
    electronic.to(optical.device)

    model = HybridModel(optical, detectors, electronic)
    dataset = Dataset('MNIST', 64, N, N, torch.complex64)

    image, _ = Dataset.single('MNIST', N, N, torch.complex64)
    model.visualize(image)

    # propagation_test(model, 'MNIST')

    # def loss(image1:torch.Tensor, image0:torch.Tensor):
    #     return torch.mean((image0 - image1)**2)
    # dataset = Dataset('MNIST', 64, N, N, torch.complex64)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    # for images, labels in tqdm(dataset.train):
    #     images = images.to(model.device)
    #     loss_value = loss(images.abs() ** 2, model.forward(images).abs())
    #     optimizer.zero_grad()
    #     loss_value.backward()
    #     optimizer.step()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # for images, labels in tqdm(dataset.train):
    #     images = images.to(model.device)
    #     loss_value = loss(images.abs() ** 2, model.forward(images).abs())
    #     optimizer.zero_grad()
    #     loss_value.backward()
    #     optimizer.step()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # for images, labels in tqdm(dataset.train):
    #     images = images.to(model.device)
    #     loss_value = loss(images.abs() ** 2, model.forward(images).abs())
    #     optimizer.zero_grad()
    #     loss_value.backward()
    #     optimizer.step()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # for images, labels in tqdm(dataset.train):
    #     images = images.to(model.device)
    #     loss_value = loss(images.abs() ** 2, model.forward(images).abs())
    #     optimizer.zero_grad()
    #     loss_value.backward()
    #     optimizer.step()
    #
    # propagation_test(model, 'STL10')