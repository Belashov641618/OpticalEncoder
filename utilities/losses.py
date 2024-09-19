from __future__ import annotations
import torch
from typing import Callable, Any, Optional, Iterable, Literal, Type, Union, TypeVar, Generic
from .methods import interpolate, normilize
from .basics import InterpolateModes, IMType

class LossLinearCombination(torch.nn.Module):
    _multipliers:list[float]
    _loss_functions:list[Union[Callable[[torch.Tensor,torch.Tensor],torch.Tensor],Normalizable.Abstract]]
    def __init__(self, *losses:Union[Callable[[torch.Tensor,torch.Tensor],torch.Tensor],Normalizable.Abstract], proportions:Iterable[Optional[float]]=None):
        super().__init__()
        self._loss_functions = []
        self._multipliers = []
        multiplier = 1.0 / len(losses)
        for loss_function in losses:
            if isinstance(loss_function, torch.nn.Module):
                self.register_module(f"{type(loss_function).__name__}", loss_function)
            self._loss_functions.append(loss_function)
            self._multipliers.append(multiplier)
        if proportions is not None:
            if not isinstance(proportions, tuple): proportions = tuple(proportions)
            self.proportions(*proportions)

    @property
    def count(self):
        return len(self._loss_functions)

    def proportions(self, *values:Optional[float]):
        if len(values) > self.count: raise ValueError(f"Amount of values is too large ({len(values)}), must be not greater than {self.count}")
        elif len(values) < self.count:
            values = [value for value in values] + [None for _ in range(self.count - len(values))]
        else: values = list(values)
        not_none_indexes = []
        none_indexes = []
        for i, value in enumerate(values):
            if value is None:   none_indexes.append(i)
            else:               not_none_indexes.append(i)
        integral = sum(values[i] for i in not_none_indexes)
        if integral > 1.0: raise ValueError(f"Sum of all not none proportion values must be lower or equal 1.0, but it`s equal {integral}")
        rest = 1.0 - integral
        none_multiplier = rest/len(none_indexes)
        for i in none_indexes:
            values[i] = none_multiplier
        self._multipliers = values
    @property
    def coefficients(self):
        return tuple(self._multipliers)
    @coefficients.setter
    def coefficients(self, values:tuple[Optional[float],...]):
        self.proportions(*values)

    def __call__(self, result:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.)
        for i in range(self.count):
            loss += self._multipliers[i] * self._loss_functions[i](result, target)
        return loss

class Normalization:
    class Abstract:
        def __call__(self, signals:torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
    class Softmax(Abstract):
        _dim:int
        def __init__(self, dim:int=1):
            self._dim=dim
        def __call__(self, signals:torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softmax(signals, dim=self._dim)
    class Minmax(Abstract):
        _dim:int
        def __init__(self, dim:int=1):
            self._dim = dim
        def __call__(self, signals:torch.Tensor) -> torch.Tensor:
            minimums, _ = torch.min(signals, dim=self._dim, keepdim=True)
            maximums, _ = torch.max(signals, dim=self._dim, keepdim=True)
            return (signals - minimums) / (maximums - minimums)
    class Max(Abstract):
        _dim:int
        def __init__(self, dim:int=1):
            self._dim = dim
        def __call__(self, signals:torch.Tensor) -> torch.Tensor:
            maximums, _ = torch.max(signals, dim=self._dim, keepdim=True)
            return signals / maximums
    class Sigmoid(Abstract):
        def __call__(self, signals:torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.sigmoid(signals)

    available_types_classes = Type[Abstract],
    available_types = Literal['softmax','minmax','max','sigmoid']
    def __class_getitem__(cls, ntype:available_types) -> available_types_classes:
        if ntype == 'softmax':
            return Normalization.Softmax
        if ntype == 'minmax':
            return Normalization.Minmax
        if ntype == 'max':
            return Normalization.Max
        if ntype == 'sigmoid':
            return Normalization.Sigmoid
        raise KeyError(f"Unknown type {ntype}")
class Normalizable:
    class Abstract(Callable[[torch.Tensor,torch.Tensor],torch.Tensor]):
        _normalization:Optional[Normalization.Abstract]
        def __init__(self, norm:Normalization.Abstract=None):
            if norm is None:
                self._normalization = None
            else:
                self._normalization = norm
        def __call__(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            if self._normalization is None:
                return self._loss(signals, target)
            else:
                return self._loss(self._normalization(signals), target)
        def _loss(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
    class MeanSquareError(Abstract):
        _mse:torch.nn.MSELoss
        def __init__(self, norm:Normalization.Abstract=None, reduction:str='mean'):
            super().__init__(norm)
            self._mse = torch.nn.MSELoss(reduction=reduction)
        def _loss(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            return self._mse(signals, target)
        def __call__(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            return super().__call__(signals, target)
    class CrossEntropy(Abstract):
        _ce:torch.nn.CrossEntropyLoss
        def __init__(self, norm:Normalization.Abstract, *args, **kwargs):
            super().__init__(norm)
            self._ce = torch.nn.CrossEntropyLoss(*args, **kwargs)
        def _loss(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            return self._ce(signals, target)
        def __call__(self, signals:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
            return super().__call__(signals, target)

    available_types = Literal['mse', 'ce']
    def __class_getitem__(cls, ltype:available_types) -> Type[Abstract]:
        if ltype == 'mse':
            return Normalizable.MeanSquareError
        if ltype == 'ce':
            return Normalizable.CrossEntropy
        raise KeyError(f"Unknown type {ltype}")

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
