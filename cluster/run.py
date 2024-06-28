import os
import torch
from typing import Callable, Type, Union, Iterable, Any
from pickle import dump, load
import IPython
import subprocess

from .functions import aims

from utilities import *

_directory = os.path.dirname(os.path.abspath(__file__))

class GPUsSelector:
    ids:list[int]
    def __init__(self, include:Union[int,Iterable[int]]=None, exclude:Union[int,Iterable[int]]=None):
        self.reset()
        if include is not None:
            self.include(*include)
        if exclude is not None:
            self.exclude(*exclude)
    def include(self, *ids):
        for id in ids:
            if id not in self.ids:
                self.ids.append(id)
    def exclude(self, *ids):
        for id in ids:
            if id in self.ids:
                self.ids.remove(id)

    def reset(self):
        self.ids = [i for i in range(torch.cuda.device_count())]

    def tuple(self):
        return tuple(self.ids)
SelectedGPUs = GPUsSelector()

def run(aim:int, *args):
    args = (SelectedGPUs.tuple(), *args)
    with open(_directory + "/cash/arguments.pkl", 'wb') as file:
        dump(args, file)
    if IPython.get_ipython():
        IPython.get_ipython().system(f'python3 "{_directory}"/functions.py "{aim}"')
    else:
        subprocess.run(["python3", f"{_directory}/functions.py", str(aim)])
    with open(_directory + "/cash/results.pkl", 'rb') as file:
        results = load(file)
    return results

def train(model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    model_, loss_history = run(aims.train, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs)
    device = (model.device if hasattr(model, 'device') else next(iter(model.parameters())).device if model.parameters() else torch.device('cpu'))
    model_.to(device)
    model = model_
    SelectedGPUs.reset()
    return loss_history, model
def confusion(model:torch.nn.Module, dataset:Dataset, classes:int=10):
    confusion_matrix = run(aims.confusion, model, dataset, classes)
    SelectedGPUs.reset()
    return confusion_matrix
def execute(model:torch.nn.Module, data:Union[Dataset, Iterable[torch.Tensor]], extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]]):
    results = run(aims.execute, model, data, extract)
    SelectedGPUs.reset()
    return results
def epochs(amount:int, classes:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    models_history, loss_histories, confusion_matrices_history = run(aims.epochs, amount, classes, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs)
    SelectedGPUs.reset()
    return models_history, loss_histories, confusion_matrices_history
