import torch
import math
from typing import Union, Iterable, Tuple

from utilities.filters import Gaussian
from utilities import distribution

# Различные генераторы многомерных шумов
class Generator:
    """
    Базовый класс любого генератора
    """
    @property
    def dims(self):
        raise NotImplementedError
    @property
    def size(self):
        raise NotImplementedError
    @property
    def device(self):
        raise NotImplementedError
    @property
    def dtype(self):
        raise NotImplementedError
    def numel(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class FourierMask(Generator):
    _mask:torch.Tensor
    @property
    def dims(self):
        return len(self._mask.size())
    @property
    def size(self):
        return self._mask.size()
    @property
    def device(self):
        return self._mask.device
    @property
    def dtype(self):
        return self._mask.dtype

    def numel(self):
        return self._mask.numel()

    def __init__(self, mask:torch.Tensor):
        self._mask = mask

    def sample(self) -> torch.Tensor:
        spectrum = torch.rand(self.size, device=self.device, dtype=self.dtype)*torch.exp(2j*torch.pi*torch.rand(self.size, device=self.device, dtype=self.dtype)) * self._mask
        return torch.sqrt(torch.abs(torch.fft.ifftn(spectrum))).to(self.dtype)


# Функции строящие определённый генератор
def gaussian(areas:Union[Iterable[float],float], counts:Union[Iterable[int],int], limits:Union[Iterable[Tuple[float,float]],Tuple[float, float]]=None, device:torch.device=None, generator:bool=True) -> Union[torch.Tensor, FourierMask]:
    print(areas)
    print(counts)
    print(limits)
    sigmas_:Tuple[float, ...]
    if isinstance(areas, float):    sigmas_ = (areas,)
    else:                           sigmas_ = tuple(areas)

    dims = len(sigmas_)

    counts_:Tuple[int, ...]
    if isinstance(counts, int):     counts_ = (counts, )
    else:                           counts_ = tuple(counts)

    limits_:Tuple[Tuple[float, float], ...]
    if limits is None:              limits_ = tuple([(-1., +1.) for _ in range(dims)])
    elif isinstance(limits, tuple) and len(limits) == 2 and all(isinstance(item, float) for item in limits):
        limits:tuple[float, float]
        limits_ = (limits, )
    else:                           limits_ = tuple(limits)

    if len(sigmas_) != dims or len(counts_) != dims or len(limits_) != dims: raise AssertionError('Lengths of sigmas and counts and limits are not equal')

    sigmas_temp = []
    limits_temp = []
    for sigma, count, limit in zip(sigmas_, counts_, limits_):
        length = limit[1] - limit[0]
        sigmas_temp.append(2.0*math.pi / sigma)
        limits_temp.append((-(count//2)*2*math.pi/length, ((count - 1)//2)*2*math.pi/length))
    sigmas_ = tuple(sigmas_temp)
    limits_ = tuple(limits_temp)

    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coordinates:list[torch.Tensor] = [torch.linspace(limit0, limit1, count, device=device) for (limit0, limit1), count in zip(limits_, counts_)]
    mask = Gaussian(sigmas_)(*coordinates)
    if generator:
        return FourierMask(mask)
    else:
        return FourierMask(mask).sample()


# Нормлизаторы генераторов
class Normalizer(Generator):
    _generator:Generator
    _steps:int

    def _limited(self):
        sample = self._generator.sample()
        return (sample - sample.min()) / (sample.max() - sample.min())
    def distribution(self, vals:bool=False):
        return distribution(self._limited(), self._steps, return_values=vals)
    def distribution_fixed(self, vals:bool=False):
        return distribution(self.sample(), self._steps, return_values=vals)

    @staticmethod
    def _compare_function(a:torch.Tensor, b:torch.Tensor):
        return torch.mean(torch.abs(a-b)**2)
    _parameters : torch.nn.Parameter
    def _function(self, x:torch.Tensor):
        raise NotImplementedError
    def difference(self):
        return self._compare_function(self._function(torch.linspace(0, 1, self._steps, device=self._generator.device)), self.distribution())
    def optimize(self):
        rate:float = 100.0
        self._parameters.requires_grad_(True)
        while rate >= 1.0E-12:
            loss = self.difference()
            loss.backward()
            grad = self._parameters.grad
            with torch.no_grad():
                while rate >= 1.0E-12:
                    self._parameters.copy_(self._parameters - grad * rate)
                    new_loss = self.difference().item()
                    if loss.item() > new_loss:
                        rate *= 1.234213521
                        break
                    else:
                        self._parameters.copy_(self._parameters + grad * rate)
                        rate /= 2.0
            self._parameters.grad.zero_()
            print(new_loss, rate)
        self._parameters.requires_grad_(False)
        print(self._parameters)

    def __init__(self, generator:Generator, steps:int=None):
        if steps is None: steps = int(math.sqrt(generator.numel()))
        self._generator = generator
        self._steps = steps
    def sample(self):
        return self._function(self._limited())
    def function(self):
        array = torch.linspace(0, 1, self._steps, device=self._generator.device)
        return array, self._function(array)

    @property
    def dims(self):
        return self._generator.dims
    @property
    def size(self):
        return self._generator.size
    @property
    def device(self):
        return self._generator.device
    @property
    def dtype(self):
        return self._generator.dtype
    def numel(self):
        return self._generator.numel()

class GaussianNormalizer(Normalizer):
    def _function(self, x:torch.Tensor):
        return 0.5*(torch.erf((x-self._parameters[1])/(self._parameters[0]*1.41421356)) + self._parameters[3])*self._parameters[2]
        # return 0.5*(torch.erf((x-self._parameters[1])/(self._parameters[0]*1.41421356)) + 1)*self._parameters[2] + self._parameters[3]
    def __init__(self, generator:Generator, steps:int=None):
        super().__init__(generator, steps)
        self._parameters = torch.nn.Parameter(torch.tensor((0.14, 0.5, 1.0, 1.0), device=self._generator.device, dtype=self._generator.dtype))

def normalize(generator:Generator, steps:int=None) -> Normalizer:
    normalizers = [
        GaussianNormalizer
    ]
    losses = []
    for i, Norm in enumerate(normalizers):
        Norm = Norm(generator, 20)
        Norm.optimize()
        losses.append(Norm.difference())
    index = losses.index(min(losses))
    Norm = normalizers[index](generator, steps)
    Norm.optimize()
    return Norm