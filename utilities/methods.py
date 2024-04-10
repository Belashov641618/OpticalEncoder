import torch
from typing import Any, Union, List, Literal

from .basics import IMType, InterpolateModes

def closest_integer(value:float) -> int:
    if value < 0:
        return -closest_integer(-value)
    integer = int(value)
    if value - integer >= 0.5:
        integer += 1
    return integer

def upper_integer(value:float) -> int:
    if value < 0:
        return -upper_integer(-value)
    integer = int(value)
    if integer - value > 0:
        integer += 1
    return integer

def interpolate(field:torch.Tensor, size:Any, mode:IMType=InterpolateModes.nearest):
    if field.dtype in [torch.complex32, torch.complex64, torch.complex128]:
        return torch.nn.functional.interpolate(field.real, size, mode=mode, antialias=False, align_corners=True) + 1j*torch.nn.functional.interpolate(field.imag, size, mode=mode, antialias=False, align_corners=True)
    else:
        return torch.nn.functional.interpolate(field, size, mode=mode, antialias=False, align_corners=True)

def shifted_log10(*images:torch.Tensor, shift:float=None, average:float=0.5, reduce:Literal['mean','max','min']='mean', return_shift:bool=True):
    if len(images) > 0:
        if shift is None:
            param = torch.nn.Parameter(torch.tensor(1.))
            optimizer = torch.optim.Adam([param], lr=0.02)
            for i in range(50):
                optimizer.zero_grad()
                loss = []
                for image in images:
                    image_ = torch.log10(image.clone() + 1.0 / param**2)
                    loss.append(torch.mean((image_ - image_.min())/(image_.max() - image_.min())))
                loss = torch.stack(loss)
                if   reduce == 'min':   loss = torch.min(loss)
                elif reduce == 'max':   loss = torch.max(loss)
                elif reduce == 'mean':  loss = torch.mean(loss)
                loss = (loss - average)**2
                loss.backward()
                optimizer.step()
            shift = 1. / torch.sqrt(param).item()
            results_ = shifted_log10(*images, shift=shift)
            if return_shift:    return results_, shift
            else:               return results_
        else:
            results:list[torch.Tensor] = []
            for image_ in images:
                results.append(torch.log10(image_ + shift))
            return tuple(results)
    return
