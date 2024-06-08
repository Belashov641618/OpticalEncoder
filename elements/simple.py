import torch

from utilities.methods import interpolate
from utilities.basics import IMType, InterpolateModes

class AdjustSize(torch.nn.Module):
    width:int
    height:int
    mode:IMType
    def __init__(self, width:int, height:int, mode:IMType=InterpolateModes.bicubic):
        super().__init__()
        self.width = width
        self.height = height
        self.mode = mode

    def forward(self, field:torch.Tensor, *args, **kwargs):
        return interpolate(field, (self.width, self.height), self.mode)

