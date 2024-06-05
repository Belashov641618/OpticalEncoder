import os
import torch
from typing import Callable, Type, Union, Iterable, Any
from pickle import dump
import IPython
import subprocess

from .functions import aims

from utilities import *

def run(aim: aims, *args):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/cash/arguments.pkl", 'wb') as file:
        dump(args, file)
    if IPython.get_ipython():
        IPython.get_ipython().system(f'python3 "{os.path.dirname(os.path.abspath(__file__))}"/functions.py "{aim}"')
    else:
        subprocess.run(["python3", f"{os.path.dirname(os.path.abspath(__file__))}/functions.py", str(aim)])

def train(model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    pass

def confusion(model:torch.nn.Module, dataset:Dataset, classes:int=10):
    pass

def execute(model:torch.nn.Module, data:Union[Dataset, Iterable[torch.Tensor]], function:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor, torch.Tensor],Any]]):
    pass

