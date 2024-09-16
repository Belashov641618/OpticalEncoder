import torch
from .methods import interpolate, normilize
from .basics import InterpolateModes, IMType
from typing import Callable

class AbstractImageComparison:
     # Parameters
    _normilize:bool
    @property
    def normilize(self):
        return self._normilize
    @normilize.setter
    def normilize(self, state:bool):
        self._normilize = state

    _interpolate_mode:IMType
    @property
    def interpolate(self):
        return self._interpolate_mode
    @interpolate.setter
    def interpolate(self, mode:IMType):
        self._interpolate_mode = mode

    def __init__(self, normilize:bool=True, interpolate:IMType=InterpolateModes.bilinear):
        self.normilize = normilize
        self.interpolate = interpolate

    def _prepare(self, result:torch.Tensor, target:torch.Tensor):
        if result.shape[2:] != target.shape[2:]:
            target = interpolate(target, result.shape[2:], self.interpolate)
        result, target = result.abs(), target.abs()
        if self._normilize:
            result = normilize(result)
            target = normilize(target)
        return result, target
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class MeanImageComparison(AbstractImageComparison):
    _function:Callable
    @property
    def function(self):
        return self._function
    @function.setter
    def function(self, function:Callable):
        self._function = function
    
    def __init__(self, function:Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = function

    def __call__(self, image:torch.Tensor, reference:torch.Tensor):
        image, reference = self._prepare(image, reference)
        difference = self._function(image, reference)
        return torch.mean(difference)

class ImageComparisonMSE(MeanImageComparison):
    def _function_mse(self, image:torch.Tensor, reference:torch.Tensor):
        return (image - reference)**2
    def __init__(self, *args, **kwargs):
        super().__init__(function=self._function_mse, *args, **kwargs)

class ImageComparisonMAE(MeanImageComparison):
    def _function_mae(self, image:torch.Tensor, reference:torch.Tensor):
        return torch.abs(image - reference)
    def __init__(self, *args, **kwargs):
        super().__init__(function=self._function_mae, *args, **kwargs)

class MseCrossEntropyCombination:
    def __init__(self, cross_entropy_to_mse_proportion:float=1.0):
        self.proportion = cross_entropy_to_mse_proportion
    def __call__(self, outputs:torch.Tensor, targets:torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets)
        mse_loss = torch.nn.functional.mse_loss(outputs, torch.nn.functional.one_hot(targets, num_classes=10).float())
        loss = self.proportion*ce_loss + (1.0-self.proportion)*mse_loss
        return loss
