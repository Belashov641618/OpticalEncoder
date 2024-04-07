import torch

from ..utilities import Accuracy, DelayedFunctions, InOutParams

class AbstractElement(torch.nn.Module):
    delayed:DelayedFunctions
    accuracy:Accuracy

    pixels:InOutParams[int]
    length:InOutParams[float]

    def __init__(self, in_pixel):
        super().__init__()
        self.accuracy = Accuracy()
        self.delayed = DelayedFunctions()

    def forward(self, *args, **kwargs):
        self.delayed.launch()