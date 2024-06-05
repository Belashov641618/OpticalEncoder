import os
import torch
from typing import Callable, Type, Union, Iterable, Any
from pickle import dump, load
import IPython
import subprocess

from .functions import aims

from utilities import *

_directory = os.path.dirname(os.path.abspath(__file__))

def run(aim:int, *args):
    with open(_directory + "/cash/arguments.pkl", 'wb') as file:
        dump(args, file)
    if IPython.get_ipython():
        IPython.get_ipython().system(f'python3 "{os.path.dirname(os.path.abspath(__file__))}"/functions.py "{aim}"')
    else:
        subprocess.run(["python3", f"{os.path.dirname(os.path.abspath(__file__))}/functions.py", str(aim)])
    with open(_directory + "/cash/results.pkl", 'rb') as file:
        results = load(file)
    return results

def train(model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    model_, loss_history = run(aims.train, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs)
    device = (model.device if hasattr(model, 'device') else next(iter(model.parameters())).device if model.parameters() else torch.device('cpu'))
    model_.to(device)
    model = model_
    return loss_history
def confusion(model:torch.nn.Module, dataset:Dataset, classes:int=10):
    confusion_matrix = run(aims.confusion, model, dataset, classes)
    return confusion_matrix

def execute(model:torch.nn.Module, data:Union[Dataset, Iterable[torch.Tensor]], extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]]):
    results = run(aims.execute, model, data, extract)
    return results

