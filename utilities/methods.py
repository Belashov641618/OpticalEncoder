import torch
from typing import Any
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
