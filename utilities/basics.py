from __future__ import annotations

import copy

import torch
from typing import List, Callable, Tuple, Any, Generic, TypeVar, Optional, Union, Literal
from functools import partial

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
    def disconnect(self, tensor:torch.Tensor):
        if torch.is_complex(tensor):
            for i, tensor_ in enumerate(self._connected_complex):
                if tensor_ is tensor:
                    self._connected_complex.pop(i)
                    break
        else:
            for i, tensor_ in enumerate(self._connected_float):
                if tensor_ is tensor:
                    self._connected_float.pop(i)
                    break

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
def function_combiner(function1:Optional[ChangeType], function2:Optional[ChangeType]):
    if function1 is not None:
        if function2 is not None:
            def combined():
                function1()
                function2()
            return combined
        else:
            def combined():
                function1()
            return combined
    elif function2 is not None:
        def combined():
            function2()
        return combined
    else:
        return None

class Param(Generic[ParamType]):
    _value : Optional[ParamType]
    _change_functions : list[ChangeType]
    def append_function(self, *functions:ChangeType):
        for function in functions:
            self._change_functions.append(function)
    def _launch(self):
        for function in self._change_functions:
            function()

    def __init__(self, *changes):
        self._value = None
        self._change_functions = []
        self.append_function(*changes)

    def set(self, value:ParamType):
        if self._value != value:
            self._value = value
            self._launch()
        return self
    def get(self):
        return self._value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value:ParamType):
        self.set(value)

class XYParams(Generic[ParamType]):
    _x:Param[ParamType]
    @property
    def x(self):
        return self._x.value
    @x.setter
    def x(self, value:ParamType):
        self._x.value = value
    @property
    def parameter_x(self):
        return self._x

    _y:Param[ParamType]
    @property
    def y(self):
        return self._y.value
    @y.setter
    def y(self, value:ParamType):
        self._y.value = value
    @property
    def parameter_y(self):
        return self._y

    def __init__(self, change_x:Union[ChangeType,tuple[ChangeType,...]]=(), change_y:Union[ChangeType,tuple[ChangeType,...]]=(), change:Union[ChangeType,tuple[ChangeType,...]]=()):
        if not isinstance(change_x, tuple): change_x = (change_x,)
        if not isinstance(change_y, tuple): change_y = (change_y,)
        if not isinstance(change,   tuple): change   = (change,)
        self._x = Param[ParamType](*change, *change_x)
        self._y = Param[ParamType](*change, *change_y)

    def set(self, data:TypeXY):
        if isinstance(data, tuple):
            self._x.set(data[0])
            self._y.set(data[1])
        else:
            self._x.set(data)
            self._y.set(data)
        return self
    def get(self):
        return self._x.get(), self._y.get()

class IOParams(Generic[ParamType]):
    _change         :Optional[ChangeType]

    _input   :XYParams[ParamType]
    @property
    def input(self):
        return self._input
    @property
    def parameter_input(self):
        return self._input

    _output  :XYParams[ParamType]
    @property
    def output(self):
        return self._output
    @property
    def parameter_output(self):
        return self._output

    def __init__(self,
                 change_input_x:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change_input_y:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change_output_x:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change_output_y:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change_input:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change_output:Union[ChangeType, tuple[ChangeType,...]]=(),
                 change:Union[ChangeType, tuple[ChangeType,...]]=()):
        if not isinstance(change_input_x,   tuple): change_input_x  = (change_input_x,)
        if not isinstance(change_input_y,   tuple): change_input_y  = (change_input_y,)
        if not isinstance(change_output_x,  tuple): change_output_x = (change_output_x,)
        if not isinstance(change_output_y,  tuple): change_output_y = (change_output_y,)
        if not isinstance(change_input,     tuple): change_input    = (change_input,)
        if not isinstance(change_output,    tuple): change_output   = (change_output,)
        if not isinstance(change,           tuple): change          = (change,)
        self._input   = XYParams[ParamType](change_input_x, change_input_y, (*change_input, *change))
        self._output  = XYParams[ParamType](change_output_x, change_output_y, (*change_output, *change))
    def set(self, data:TypeIO):
        if isinstance(data, tuple):
            self._input.set(data[0])
            self._output.set(data[1])
        else:
            self._input.set(data)
            self._output.set(data)
        return self

class SpaceParamGroup:
    _parameters:list[SpaceParam]
    def connect(self, *parameters:SpaceParam):
        for parameter in parameters:
            if parameter not in self._parameters:
                parameter.group = self
                self._parameters.append(parameter)
                parameter.adjust()
    def disconnect(self, *parameters:SpaceParam):
        for parameter in self._parameters:
            if parameter in parameters:
                self._parameters.remove(parameter)
    def stole(self):
        parameters = self._parameters.copy()
        self._parameters.clear()
        return parameters

    def merge(self, other:SpaceParamGroup):
        for parameter in other.stole():
            parameter.group = self
        del other


    @property
    def size(self):
        maximum = 1
        for parameter in self._parameters:
            if parameter.size > maximum:
                maximum = parameter.size
        return maximum

    _trigger:bool
    def adjust(self):
        if not self._trigger:
            self._trigger = True
            size = self.size
            for parameter in self._parameters:
                if parameter.size != size:
                    parameter.linspace(parameter.left, parameter.right, size)
            self._trigger = False
    def __init__(self, *parameters:SpaceParam):
        self._trigger = False
        self._parameters = []
        self.connect(*parameters)
FloatS = Union[float, tuple[float,float,int]]
class SpaceParam(Generic[ParamType]):
    _value:torch.Tensor
    _change:list[ChangeType]
    def append_function(self, *functions:ChangeType):
        for function in functions:
            self._change.append(function)

    def _launch(self):
        for function in self._change:
            function()
        self._group.adjust()

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
    def size(self):
        if len(self._value.size()) == 0:
            return 1
        return self._value.size(0)
    @property
    def left(self):
        return self._value[0]
    @property
    def right(self):
        return self._value[-1]

    def value(self, value:ParamType):
        self.linspace(value, value, self._group.size)
    def linspace(self, value0:ParamType, value1:ParamType, N:int):
        if value0 != value1:
            temp = torch.linspace(value0, value1, N, device=self._value.device, dtype=self._value.dtype)
        else:
            temp = torch.ones(N, device=self._value.device, dtype=self._value.dtype) * value0
        if temp != self._value:
            self._value = temp
            self._launch()
    def set(self, data:Union[ParamType, tuple[ParamType,ParamType,int]]):
        self._value = torch.tensor(0.)
        if isinstance(data, tuple):
            self.linspace(*data)
        else:
            self.value(data)

    _group:SpaceParamGroup
    @property
    def group(self):
        return self._group
    @group.setter
    def group(self, group_:SpaceParamGroup):
        self._group.disconnect(self)
        self._group = group_
    def connect(self, *space_params:SpaceParam):
        self._group.connect(*space_params)
    def adjust(self):
        size = self.group.size
        if self.size != size:
            self.linspace(self.left, self.right, size)

    def __init__(self, *change:ChangeType, group:SpaceParamGroup=None):
        self._value = torch.zeros(1, dtype=torch.float32)
        self._change = []
        self.append_function(*change)
        if group is None:
            group = SpaceParamGroup()
        self._group = group
        self._group.connect(self)


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

