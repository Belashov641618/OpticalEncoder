import torch
from typing import List, Callable, Tuple, Any, Generic, TypeVar, Optional, Union

class DelayedFunctions:
    _delayed_functions : List[Tuple[Callable, float, Any, Any]]
    def add(self, function:Callable, priority:float=0., *args, **kwargs):
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

    def __init__(self):
        self._bits = 32
        self.tensor_float = torch.float32
        self.tensor_complex = torch.complex64

    def set(self, bits:int):
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

    def get(self):
        return self._bits

IntXY   = Union[int, Tuple[int,int]]
FloatXY = Union[float, Tuple[float,float]]

ParamType = TypeVar('ParamType')
class ChangeableParam(Generic[ParamType]):
    _param : Optional[ParamType]
    _change_function : Callable
    @property
    def param(self):
        return self._param
    @param.setter
    def param(self, value:ParamType):
        if value != self._param:
            self._param = value
            self._change_function()
    def __init__(self, change_function:Callable):
        self._param = None
        self._change_function = change_function

class XYParams(Generic[ParamType]):
    _x:ChangeableParam[ParamType]
    @property
    def x(self):
        return self._x.param
    @x.setter
    def x(self, value:ParamType):
        self._x.param = value

    _y:ChangeableParam[ParamType]
    @property
    def y(self):
        return self._y.param
    @y.setter
    def y(self, value:ParamType):
        self._y.param = value

    def set(self, data:Union[ParamType, Tuple[ParamType,ParamType]]):
        if isinstance(data, tuple):
            self.x = data[0]
            self.y = data[1]
        else:
            self.x = data
            self.y = data

    def __init__(self, change_x:Callable, change_y:Callable):
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

    def __init__(self, change_in_x:Callable, change_in_y:Callable, change_out_x:Callable, change_out_y:Callable):
        self._input = XYParams[ParamType](change_in_x, change_out_x)
        self._output = XYParams[ParamType](change_in_y, change_out_y)
