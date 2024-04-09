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



ParamType = TypeVar('ParamType')
TypeXY  = Union[ParamType, Tuple[ParamType,ParamType]]
IntXY   = Union[int, Tuple[int,int]]
FloatXY = Union[float, Tuple[float,float]]
TypeIO  = Union[ParamType, Tuple[ParamType,ParamType], Tuple[ParamType,ParamType,ParamType,ParamType]]
IntIO   = Union[int, Tuple[int,int], Tuple[int,int,int,int]]
FloatIO = Union[float, Tuple[float,float], Tuple[float,float,float,float]]

ChangeType = Callable[[], None]
class Param(Generic[ParamType]):
    _value : Optional[ParamType]
    _change_function : Optional[ChangeType]
    def __init__(self, change:ChangeType=None):
        self._change_function = change
        self._value = None
    def set(self, value:ParamType):
        if self._value != value:
            self._value = value
            self._change_function()
    def get(self):
        return self._value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value:ParamType):
        self.set(value)

class XYParams(Generic[ParamType]):
    _change : Optional[ChangeType]

    _x:Param[ParamType]
    @property
    def x(self):
        return self._x.value
    @x.setter
    def x(self, value:ParamType):
        if self._x != value:
            self._x.value = value
            self._change()

    _y:Param[ParamType]
    @property
    def y(self):
        return self._y.value
    @y.setter
    def y(self, value:ParamType):
        if self._y != value:
            self._y.value = value
            self._change()

    def __init__(self, change_x:ChangeType=None, change_y:ChangeType=None, change:ChangeType=None):
        self._change = change
        self._x = Param[ParamType](change_x)
        self._y = Param[ParamType](change_y)

    def set(self, data:TypeXY):
        if isinstance(data, tuple):
            self.x.set(data[0])
            self.y.set(data[1])
            if self._x.value != data[0] or self._y.value != data[1]:
                self._change()
        else:
            self.x.set(data)
            self.y.set(data)
            if self._x.value != data or self._y.value != data:
                self._change()
    def get(self):
        return self.x.get(), self.y.get()

class IOParams(Generic[ParamType]):
    _change         :Optional[ChangeType]

    _input   :XYParams[ParamType]
    @property
    def input(self):
        return self._input

    _output  :XYParams[ParamType]



    def __init__(self, change_input_x:ChangeType=None, change_input_y:ChangeType=None, change_output_x:ChangeType=None, change_output_y:ChangeType=None, change_input:ChangeType=None, change_output:ChangeType=None, change:ChangeType=None):
        self._change = change
        self._change_input = change_input
        self._change_output = change_output
        self._input   = XYParams[ParamType](change_input_x, change_input_y, change_input)
        self._output  = XYParams[ParamType](change_output_x, change_output_y, change_output)

    def set(self, data:TypeIO):
        if isinstance(data, tuple):
            self._input.set(data[0])
            self._output.set(data[1])
        else:
            self._input.set(data)
            self._output.set(data)



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

