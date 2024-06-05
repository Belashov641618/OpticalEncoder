import os
import sys
import tempfile
import time

import torch
import torch.optim as optim
from enum import Enum
from pickle import load
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler
from typing import Callable, Type

if __name__ == '__main__':
    sys.path.append('..')
from utilities.training import train, confusion
from utilities import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    torch.distributed.destroy_process_group()

class _aims:
    train:int
    confusion:int
    execute:int
    def __init__(self):
        self.train = 1
        self.confusion = 1
        self.execute = 1
aims = _aims()

def _train_flow(rank:int, world_size:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], optimizer_args, optimizer_kwargs):
    print(f"My PID is: {os.getpid()}")
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))
    dataset.sampler.parallel()
    optimizer = optimizer(model.parameters(), *optimizer_args, **optimizer_kwargs)
    train(model, dataset, optimizer, loss_function, echo=(rank == 0))
    cleanup()
def _train(model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(_train_flow, args=(world_size, model, loss_function, dataset, optimizer, optimizer_args, optimizer_kwargs), nprocs=world_size, join=True)

def _confusion_flow(*args):
    raise NotImplementedError
def _confusion(*args):
    raise NotImplementedError

def _execute_flow(*args):
    raise NotImplementedError
def _execute(*args):
    raise NotImplementedError

def main():
    aim = int(sys.argv[1])
    with open(os.path.dirname(os.path.abspath(__file__)) + "/cash/arguments.pkl", 'rb') as file:
        arguments = load(file)
    if aim == aims.train:
        _train(*arguments)
    elif aim == aims.confusion:
        _confusion(*arguments)
    elif aim == aims.execute:
        _execute(*arguments)
    elif aim == -1:
        for i in range(7):
            print(i)
            time.sleep(2)
    else:
        raise AttributeError('Выбран не верный тип операции')
if __name__ == '__main__':
    main()