import torch
from .methods import interpolate, normilize
from .basics import InterpolateModes

class ImageComparisonMSE:
    def __init__(self):
        pass
    def __call__(self, result:torch.Tensor, target:torch.Tensor):
        if result.shape[2:] != target.shape[2:]:
            target = interpolate(target, result.shape[2:], InterpolateModes.bilinear)
        result = normilize(result.abs())
        target = normilize(target.abs())
        absalute_square_difference = torch.abs(result - target)**2
        return torch.mean(absalute_square_difference)
        