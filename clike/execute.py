import os
import sys
import tempfile
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
    train = 1
    confusion = 2
    execute = 3
aims = _aims()

def _train_flow(rank, world_size, model:torch.nn.Module, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], dataset:Dataset, optimizer_type:type(torch.optim.Optimizer), learning_rate:float):
    print(f"My PID is: {os.getpid()}")
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))
    dataset.sampler.parallel()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    train(model, dataset, optimizer, loss_function, echo=(rank == 0))
    cleanup()
def _train(model, dataset, optimizer, learning_rate, loss_function):
    # model:torch.nn.Module, dataset:Dataset, optimizer:torch.optim.Optimizer, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]
    print(type(model).__name__)
    print(dataset)
    print(optimizer)
    print(loss_function)
    torch.multiprocessing.spawn(_train_flow, args=(torch.cuda.device_count(), model, loss_function, dataset, optimizer, learning_rate), nprocs=torch.cuda.device_count(), join=True)

def _confusion(*args):
    pass
def _execute(*args):
    pass

if __name__ == '__main__':
    print(sys.argv)
    aim = int(sys.argv[1])
    print(f"Aim is: {aim}")
    with open(os.path.dirname(os.path.abspath(__file__)) + "/cash/arguments.pkl", 'rb') as file:
        arguments = load(file)
    if aim == aims.train:           _train(*arguments)
    elif aims == aims.confusion:    _confusion(*arguments)
    elif aims == aims.execute:      _execute(*arguments)
    else:                           raise AttributeError('Выбран не верный тип операции')