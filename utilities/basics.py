from __future__ import annotations

import torch
from typing import List, Callable, Tuple, Any, Generic, TypeVar, Optional, Union, Literal

class DelayedFunctions:
    _delayed_functions : List[Tuple[Callable, float, Any, Any]]
    def add(self, function:Callable, priority:float=0., *args, **kwargs):
        """
        :param function: Функция
        :param priority: Чем меньше приоритет, тем раньше вызовется функция
        :param args:     Аргументы функции
        :param kwargs:   Параметры функции
        :return:         None
        """
        if not hasattr(self, '_delayed_functions'):
            self._delayed_functions = [(function, priority, args, kwargs)]
        elif (function, priority) not in self._delayed_functions:
            self._delayed_functions.append((function, priority, args, kwargs))
    def launch(self):
        if hasattr(self, '_delayed_functions'):
            self._delayed_functions.sort(key=lambda element: element[1])
            for (function, priority, args, kwargs) in self._delayed_functions:
                function(*args, **kwargs)
            self._delayed_functions.clear()
    def __init__(self):
        self._delayed_functions = []

class Accuracy:
    _bits : int
    tensor_float : torch.dtype
    tensor_complex : torch.dtype

    _connected_float : list[torch.Tensor]
    _connected_complex : list[torch.Tensor]
    def connect(self, tensor:torch.Tensor):
        if not hasattr(self, '_connected_float'):
            self._connected_float = []
        if not hasattr(self, '_connected_complex'):
            self._connected_complex = []
        if tensor.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            tensor.to(dtype=self.tensor_float)
            self._connected_float.append(tensor)
        elif tensor.dtype in [torch.complex32, torch.complex64, torch.complex128]:
            tensor.to(dtype=self.tensor_complex)
            self._connected_complex.append(tensor)
        else: raise TypeError(f'Тензор {tensor} имеет тип {tensor.dtype}')

    def __init__(self):
        self._bits = 32
        self.tensor_float = torch.float32
        self.tensor_complex = torch.complex64

    def set(self, bits:int):
        if bits != self._bits:
            if bits == 16:
                self.tensor_float = torch.float16
                self.tensor_complex = torch.complex32
            elif bits == 32:
                self.tensor_float = torch.float32
                self.tensor_complex = torch.complex64
            elif bits == 64:
                self.tensor_float = torch.float64
                self.tensor_complex = torch.complex128
            else:
                raise ValueError('bits may be 16, 32 or 64')
            self._bits = bits

            for tensor in self._connected_float:
                tensor.to(self.tensor_float)
            for tensor in self._connected_complex:
                tensor.to(self.tensor_complex)

    def get(self):
        return self._bits


IntXY   = Union[int, Tuple[int,int]]
FloatXY = Union[float, Tuple[float,float]]

IntIO = Union[int, Tuple[int,int], Tuple[int,int,int,int]]
FloatIO = Union[float, Tuple[float,float], Tuple[float,float,float,float]]

ParamType = TypeVar('ParamType')
class ChangeableParam(Generic[ParamType]):
    _param : Optional[ParamType]
    _change_function : Optional[Callable]
    def __get__(self):
        return self._param
    def __set__(self, value:ParamType):
        if value != self._param:
            self._param = value
            if self._change_function is not None:
                self._change_function()
    def __init__(self, change_function:Optional[Callable]):
        self._param = None
        self._change_function = change_function

    def __truediv__(self, other):
        if isinstance(other, ChangeableParam):
            return self._param / other._param
        else:
            return self._param / other
    def __rtruediv__(self, other):
        if isinstance(other, ChangeableParam):
            return other._param / self._param
        else:
            return other / self._param
    def __mul__(self, other):
        if isinstance(other, ChangeableParam):
            return self._param * other._param
        else:
            return self._param * other
    def __rmul__(self, other):
        return self*other


    # def __eq__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param == other._param if isinstance(other, ChangeableParam) else self._param == other
    # def __ne__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param != other._param if isinstance(other, ChangeableParam) else self._param != other
    # def __lt__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param <  other._param if isinstance(other, ChangeableParam) else self._param <  other
    # def __gt__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param >  other._param if isinstance(other, ChangeableParam) else self._param >  other
    # def __le__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param <= other._param if isinstance(other, ChangeableParam) else self._param <= other
    # def __ge__(self, other:Union[ParamType, ChangeableParam[ParamType]]):
    #     return self._param >= other._param if isinstance(other, ChangeableParam) else self._param >= other



class XYParams(Generic[ParamType]):
    _x:ChangeableParam[ParamType]
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value:ParamType):
        self._x = value

    _y:ChangeableParam[ParamType]
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value:ParamType):
        self._y = value

    def set(self, data:Union[ParamType, Tuple[ParamType,ParamType]]):
        if isinstance(data, tuple):
            self.x = data[0]
            self.y = data[1]
        else:
            self.x = data
            self.y = data

    def __init__(self, change_x:Optional[Callable], change_y:Optional[Callable]):
        self._x = ChangeableParam[ParamType](change_x)
        self._y = ChangeableParam[ParamType](change_y)

class InOutParams(Generic[ParamType]):
    _input : XYParams[ParamType]
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, *args, **kwargs):
        raise NotImplementedError

    _output : XYParams[ParamType]
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, *args, **kwargs):
        raise NotImplementedError

    def set(self, data:Union[ParamType, Tuple[ParamType,ParamType], Tuple[ParamType,ParamType,ParamType,ParamType]]):
        if isinstance(data, tuple):
            if len(data) == 2:
                self._input.set(data[0])
                self._output.set(data[1])
            if len(data) == 4:
                self._input.x = data[0]
                self._input.y = data[1]
                self._output.x = data[2]
                self._output.y = data[3]
            else: raise AttributeError
        else:
            self._input.set(data)
            self._output.set(data)

    def __init__(self, change_in_x:Optional[Callable], change_in_y:Optional[Callable], change_out_x:Optional[Callable], change_out_y:Optional[Callable]):
        self._input = XYParams[ParamType](change_in_x, change_out_x)
        self._output = XYParams[ParamType](change_in_y, change_out_y)

FloatS = Union[float, tuple[float,float,int]]
class SpaceParam(Generic[ParamType]):
    _value:torch.Tensor
    _change:Callable

    @property
    def tensor(self):
        return self._value
    @tensor.setter
    def tensor(self, value:torch.Tensor):
        raise NotImplementedError
    @property
    def effective(self) -> ParamType:
        return (self.left.item() + self.right.item()) / 2

    @property
    def length(self):
        return self._value.size(0)
    @property
    def left(self):
        return self._value[0]
    @property
    def right(self):
        return self._value[-1]

    def value(self, value:ParamType):
        self.linspace(value, value, self.size())
    def linspace(self, value0:ParamType, value1:ParamType, N:int):
        if value0 != value1:
            temp = torch.linspace(value0, value1, N, device=self._value.device, dtype=self._value.dtype)
        else:
            temp = torch.ones(N, device=self._value.device, dtype=self._value.dtype) * value0
        if temp != self._value:
            self._value = temp
            self._change()
    def set(self, data:Union[ParamType, tuple[ParamType,ParamType,int]]):
        self._value = torch.tensor(0.)
        if isinstance(data, tuple):
            self.linspace(*data)
        else:
            self.value(data)
    def recount(self, N:int):
        self.linspace(self.left, self.right, N)

    _connections:list[SpaceParam]
    def connect(self, *space_params:SpaceParam):
        if not hasattr(self, '_connections'):
            self._connections = []
        for param in space_params:
            self._connections.append(param)
    def adjust(self):
        for param in self._connections:
            if param.length != self.length:
                param.recount(self.length)
    def size(self):
        maximum:int = 1
        for param in self._connections:
            if maximum < param.length:
                maximum = param.length
        return maximum

    def __init__(self, change:Callable):
        self._change = change
        self._connections = []


IMType = Literal['nearest','linear','bilinear','bicubic','trilinear','area','nearest-exact']
class InterpolateMode:
    _mode:IMType
    def __init__(self, _mode:IMType='linear'):
        self._mode = _mode
    def __get__(self):
        return self._mode
    def __set__(self):
        raise NotImplementedError
    @property
    def mode(self):
        return self._mode

    def nearest(self):
        self._mode = 'nearest'
    def linear(self):
        self._mode = 'linear'
    def bilinear(self):
        self._mode = 'bilinear'
    def bicubic(self):
        self._mode = 'bicubic'
    def trilinear(self):
        self._mode = 'trilinear'
    def area(self):
        self._mode = 'area'
    def nearest_exact(self):
        self._mode = 'nearest-exact'
class _InterpolateModes:
    def __init__(self):
        pass
    @property
    def nearest(self):
        return 'nearest'
    @property
    def linear(self):
        return 'linear'
    @property
    def bilinear(self):
        return 'bilinear'
    @property
    def bicubic(self):
        return 'bicubic'
    @property
    def trilinear(self):
        return 'trilinear'
    @property
    def area(self):
        return 'area'
    @property
    def nearest_exact(self):
        return 'nearest-exact'
InterpolateModes = _InterpolateModes()

