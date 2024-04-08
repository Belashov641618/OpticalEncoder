from __future__ import annotations

from typing import TextIO, Union, List, Iterable, Callable, Iterator
from time import ctime, time
import math
import numpy

Color = int
class _ColorsList:
    Black           :Color = 0
    White           :Color = 15
    Teal            :Color = 6
    Silver          :Color = 7
    Grey            :Color = 8
    Red             :Color = 9
    Maroon          :Color = 1
    Green           :Color = 2
    DarkOrange      :Color = 208
    NavajoWhite     :Color = 223
    LightSteelBlue  :Color = 147
    DarkSeaGreen    :Color = 158
    Plum            :Color = 219
Colors:_ColorsList = _ColorsList()
def fore_ground(color:Color):
    return f"\033[38;5;{color}m"
def back_ground(color:Color):
    return f"\033[48;5;{color}m"
def _test_colors(text:str= 'aboba'):
    for i in range(256):
        print(f"{i}:\033[38;5;{i}m {text}\033[0;0m")

Highlighter = str
class _HighlightersList:
    bold:Highlighter = '\033[1m'
    italic:Highlighter = '\033[3m'
    underline:Highlighter = '\033[4m'
    highlight:Highlighter = '\033[7m'
    cross:Highlighter = '\033[9m'
    underLINE:Highlighter = '\033[21m'
    squared:Highlighter = '\033[51m'
Highlighters:_HighlightersList = _HighlightersList()
def _test_highlighters(text:str= 'aboba'):
    for i in range(256):
        print(f"{i}:\033[{i}m {text}\033[0;0m")

class Style:
    _prefixes:List[str]
    _color:str
    _background_color:str
    _logger:Logger
    def __init__(self, logger:Logger, color:Union[int,Color]=Colors.White, highlights:Union[Highlighter,List[Highlighter]]=None, background_color:Union[int,Color]=None):
        self._logger = logger
        if highlights is None: highlights = []
        elif isinstance(highlights, Highlighter): highlights = [highlights]
        self._prefixes = highlights

        if color is not None:
            self._color = fore_ground(color)
        else:
            self._color = ''

        if background_color is not None:
            self._background_color = back_ground(background_color)
        else:
            self._background_color = ''

    def set_highlights(self, highlights:Union[Highlighter,List[Highlighter]]=None):
        if highlights is None: highlights = []
        elif isinstance(highlights, Highlighter): highlights = [highlights]
        self._prefixes = highlights
    def set_color(self, color:Union[int,Color]):
        if color is not None:
            self._color = fore_ground(color)
        else:
            self._color = ''
    def set_background_color(self, color:Union[int,Color]):
        if color is not None:
            self._background_color = back_ground(color)
        else:
            self._background_color = ''

    def __call__(self, string:str, **kwargs):
        self._logger(string, "".join(self._prefixes + [self._color, self._background_color]), **kwargs)

class EscapePredictor:
    _total:int
    _start:float

    _history:list[float]
    _recalc_after:float
    _parameters:numpy.ndarray
    def _recalc_params(self):
        if not hasattr(self, '_parameters'):
            self._parameters = numpy.ones(5)

        lam:float = 1000.0
        while lam >= 1.0E-6:
            gradients = numpy.zeros(5)
            indices = numpy.arange(len(self._history))
            times = self._time(indices)
            history = numpy.array(self._history)
            diff = times - history

            gradients[4] = 2*numpy.mean(diff * numpy.log(indices) * (self._parameters[0]*indices**3 + self._parameters[1]*indices**2 + self._parameters[2]*indices + self._parameters[3]))
            temp = 1 + numpy.log(indices)*self._parameters[4]
            gradients[3] = 2*numpy.mean(diff * temp)
            temp *= indices
            gradients[2] = 2 * numpy.mean(diff * temp)
            temp *= indices
            gradients[1] = 2 * numpy.mean(diff * temp)
            temp *= indices
            gradients[0] = 2 * numpy.mean(diff * temp)

            loss = numpy.mean(diff**2)
            while lam >= 1.0E-6:
                self._parameters -= gradients * lam
                loss_ = numpy.mean((self._time(indices) - history)**2)
                if loss_ < loss:
                    lam *= 1.213412
                    break
                else:
                    self._parameters += gradients * lam
                    lam /= 2.

    def _time(self, iteration:Union[int,numpy.ndarray]):
        return (self._parameters[0]*iteration**3 + self._parameters[1]*iteration**2 + self._parameters[2]*iteration + self._parameters[3])*(1.0 + self._parameters[4]*numpy.log(iteration))
    def _left(self):
        return self._time(self._total-1)

    def __init__(self, total:int):
        self._total = total
    def __iter__(self):
        self._start = time()
        self._recalc_after = 0.0
        self._history = []
        return self
    def __next__(self):
        self._history.append(time() - self._start)
        if self._history[-1] > self._recalc_after:
            self._recalc_after += 2.0
            self._recalc_params()
        return self._left()

class CycledLogger(Iterable):
    _logger:Logger

    _iterable:Iterable
    _calculator:EscapePredictor
    _total:int
    def __init__(self, logger:Logger, iterable:Iterable, total:int, additions:list[Callable[[],str]]):
        self._logger = logger
        if total is None:
            if hasattr(iterable, '__len__'):
                total = iterable.__len__()
            else: raise AttributeError('Итеририруемый объект не имеет метода __len__, необходимо предоставить параметр total')
        self._iterable = iterable
        self._total = total
        self._calculator = EscapePredictor(total)
    _iterator:Iterator
    def __iter__(self):
        self._iterator = iter(self._iterable)
        self._calculator = iter(self._calculator)
        return self
    def __next__(self):
        value = next(self._iterator)
        time_left = next(self._calculator)

        return value

class Logger:
    ok      :Style
    error   :Style
    warning :Style
    message :Style
    listing :Style
    info    :Style
    def _init_styles(self):
        self.ok         = Style(self, Colors.Green, Highlighters.bold)
        self.error      = Style(self, Colors.Red, Highlighters.bold)
        self.warning    = Style(self, Colors.DarkOrange)
        self.message    = Style(self, Colors.NavajoWhite)
        self.listing    = Style(self, Colors.Silver, Highlighters.italic)
        self.info       = Style(self, Colors.LightSteelBlue)

    def __init__(self, echo:bool=True, prefix:str='', show_time:bool=True):
        pass

    def print(self, string:str):
        pass
    def __call__(self, string:str, params:str, end:str='\n', **kwargs):
        pass

    _cycles :list[CycledLogger]
    @property
    def cycled(self):
        if self._cycles: return True
        else:            return False
    def cycle(self, iterable:Iterable, total:int=None, additions:list[Callable[[],str]]=()):
        self._cycles.append(CycledLogger(self, iterable, total, additions))
        return self._cycles[-1]
    def update_cycles(self):
        pass

class Logger_:
    _sub_loggers : dict[str, Logger]
    def sub(self, key:str= "", log:bool=None, echo:bool=None, time:bool=None):
        if key in self._sub_loggers.keys():
            raise KeyError(f'Дочерний логгер с доболнительным префиксом {key} уже существует')
        if log is None: log = False if self._file is None else True
        if echo is None: echo = self._echo
        if time is None: time = self._time
        logger = Logger(log=log, echo=echo, time=time, path=self._path, prefix=self._prefix + key + ': ')
        self._sub_loggers[key] = logger
        return logger

    _cycles:list[CycledLogger]
    def cycle(self, iterable:Iterable, total:int=None, additions:list[Callable[[],str]]=()):
        self._cycles.append(CycledLogger(self, iterable, total, additions))

    _path:str
    _file:Union[TextIO, None]
    _echo:bool
    _prefix:str
    _time:bool

    def __init__(self, log:bool=False, echo:bool=True, path:str="Logs.txt", time:bool=True, prefix:str= ""):
        self._sub_loggers = {}
        self._echo = echo
        self._time = time
        self._prefix = prefix
        self._path = path
        if log:     self._file = open(self._path, 'w')
        else:       self._file = None
        self._newline = True
        self._init_styles()
    def __delete__(self, instance):
        if self._file is not None:
            self._file.close()
        self._file = None

    @property
    def enable(self):
        class Selector:
            _self : Logger
            def __init__(self, _self:Logger):
                self._self = _self
            def log(self, status:Union[bool, TextIO]=True):
                if isinstance(status, TextIO):
                    if self._self._file is not None: self._self._file.close()
                    self._self._file = status
                elif status and self._self._file is None:
                    self._self._file = open(self._self._path, 'w')
                elif self._self._file is not None:
                    self._self._file.close()
                    self._self._file = None
                for logger in self._self._sub_loggers.values():
                    logger.enable.log(status)
            def echo(self, status:bool=True):
                self._self._echo = status
                for logger in self._self._sub_loggers.values():
                    logger.enable.echo(status)
            def time(self, status:bool=True):
                self._self._time = status
                for logger in self._self._sub_loggers.values():
                    logger.enable.time(status)
        return Selector(self)
    def prefix(self, string:str= ""):
        self._prefix = string
        for key, logger in self._sub_loggers.items():
            logger.prefix(string + key + ': ')

    ok          :Style
    error       :Style
    warning     :Style
    message     :Style
    listing     :Style
    information :Style
    user1       :Style
    user2       :Style
    user3       :Style
    user4       :Style
    def _init_styles(self):
        self.ok =           Style(self, Colors.Green, Highlighters.bold)
        self.error =        Style(self, Colors.Red, Highlighters.bold)
        self.warning =      Style(self, Colors.DarkOrange)
        self.message =      Style(self, Colors.NavajoWhite)
        self.listing =      Style(self, Colors.Silver, Highlighters.italic)
        self.information =  Style(self, Colors.LightSteelBlue)
        self.user1 =        Style(self, Colors.DarkSeaGreen)
        self.user2 =        Style(self, Colors.Plum)
        self.user3 =        Style(self, Colors.DarkSeaGreen)
        self.user4 =        Style(self, Colors.Plum)

    _last_line_length:int

    _newline:bool
    def __call__(self, string:str, params:str, end:str='\n', **kwargs):
        prefix = ''
        if self._newline:
            prefix = self._prefix
            if self._time:
                prefix = f"[{ctime()}] {prefix}"
        tab = '\n' + ' ' * len(prefix)
        string = string.replace('\n', tab)
        if self._echo:
            if self._cycles:
                string = f"{params}{prefix}{string}\033[0;0m"
                pass
                self._last_line_length = 1 #####
            else:
                print(f"{params}{prefix}{string}\033[0;0m", end=end, **kwargs)
        if self._file is not None:
            print(f"{prefix}{string}", end=end, file=self._file, **kwargs)
            self._file.flush()
        if len(end): self._newline = end[-1] == '\n'
        else:        self._newline = string[-1] == '\n'