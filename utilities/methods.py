import torch
from typing import Any, Literal

from .basics import IMType, InterpolateModes
from .filters import Gaussian

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
    trigger = True
    for length0, length1 in zip(reversed(field.size()), size):
        trigger *= (length0 == length1)
    if trigger: return field

    align = (True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None)
    if field.dtype in [torch.complex32, torch.complex64, torch.complex128]:
        return torch.nn.functional.interpolate(field.real, size, mode=mode, antialias=False, align_corners=align) + 1j*torch.nn.functional.interpolate(field.imag, size, mode=mode, antialias=False, align_corners=align)
    else:
        return torch.nn.functional.interpolate(field, size, mode=mode, antialias=False, align_corners=align)

def shifted_log10(*images:torch.Tensor, shift:float=None, average:float=0.5, reduce:Literal['mean','max','min']='mean', return_shift:bool=True):
    if len(images) > 0:
        if shift is None:
            param = torch.nn.Parameter(torch.tensor(1.))
            def shift_param():
                return 1.0 / param**2
            def loss_function():
                loss_ = []
                for image in images:
                    image__ = torch.log10(image + shift_param())
                    image__ = torch.mean((image__ - image__.min()) / (image__.max() - image__.min()))
                    loss_.append(image__)
                loss_ = torch.stack(loss_)
                if reduce   == 'min':   loss_ = torch.min(loss_)
                elif reduce == 'max':   loss_ = torch.max(loss_)
                elif reduce == 'mean':  loss_ = torch.mean(loss_)
                loss_ = (loss_ - average) ** 2
                return loss_
            prev_loss = loss_function().item()
            rate = 1.0E+8
            while 1.0E-12 < rate < 1.0E+12:
                loss = loss_function()
                loss.backward()
                with torch.no_grad():
                    while 1.0E-12 < rate < 1.0E+12:
                        param.copy_(param - param.grad.clone().detach() * rate)
                        if loss.item() > prev_loss:
                            param.copy_(param + param.grad.clone().detach() * rate)
                            rate /= 2.0
                        else:
                            rate *= 1.321654987
                            break
                param.grad.zero_()
            shift = shift_param().item()
            results_ = shifted_log10(*images, shift=shift)
            if return_shift:    return results_, shift
            else:               return results_
        else:
            results:list[torch.Tensor] = []
            for image_ in images:
                image_ = torch.log10(image_ + shift)
                results.append((image_ - image_.min()) / (image_.max() - image_.min()))
            if len(results) == 1:
                return results[0]
            return tuple(results)
    return

def dimension_normalization(image:torch.Tensor, dim:int=0):
    maximums = image
    minimums = image
    for d in range(len(image.size())):
        if d != dim:
            maximums, _ = torch.max(maximums, dim=d, keepdim=True)
            minimums, _ = torch.min(minimums, dim=d, keepdim=True)

    image = (image - minimums) / (maximums - minimums)
    return image


def trays_rays(image:torch.Tensor):
    if len(image.size()) == 2:
        image = image.unsqueeze(0).unsqueeze(0)

    derivative_x = +image*5/4 + image.roll(+1,dims=2)*2/3 + image.roll(-1,dims=2)*2/3 - image.roll(+2,dims=2)/24 - image.roll(-2,dims=2)/24
    derivative_y = +image*5/4 + image.roll(+1,dims=3)*2/3 + image.roll(-1,dims=3)*2/3 - image.roll(+2,dims=3)/24 - image.roll(-2,dims=3)/24

    derivative_x = (derivative_x.roll(-2,dims=2) + derivative_x.roll(-1,dims=2) + derivative_x + derivative_x.roll(+1,dims=2) + derivative_x.roll(+2,dims=2))
    derivative_y = (derivative_y.roll(-2,dims=3) + derivative_y.roll(-1,dims=3) + derivative_y + derivative_y.roll(+1,dims=3) + derivative_y.roll(+2,dims=3))

    mask_x = (derivative_x > derivative_x.roll(+1, dims=2)) * (derivative_x > derivative_x.roll(-1, dims=2))
    mask_y = (derivative_y > derivative_y.roll(+1, dims=2)) * (derivative_y > derivative_y.roll(-1, dims=2))

    # derivative_x = (image.roll(-2, dims=2) - image.roll(2, dims=2))/12. + (image.roll(+1, dims=2) - image.roll(-1, dims=2))*2./3.
    # derivative_y = (image.roll(-2, dims=3) - image.roll(2, dims=3))/12. + (image.roll(+1, dims=3) - image.roll(-1, dims=3))*2./3.
    # mask_x = (0 > derivative_x.roll(+1, dims=2)) * (0 > derivative_x.roll(-1, dims=2))
    # mask_y = (0 > derivative_y.roll(+1, dims=2)) * (0 > derivative_y.roll(-1, dims=2))


    mask = mask_x * mask_y

    return mask
