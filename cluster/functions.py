import os
import sys
import tempfile
import time
import numpy
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import IterableDataset, DataLoader
from torchsummary import summary
from pickle import load, dump, dumps, loads
from copy import deepcopy
from typing import Callable, Type, Union, Iterable, Any
from tqdm import tqdm


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    sys.path.append('..')
from utilities.training import train, confusion
from utilities import *
from elements.composition import HybridModel


_directory = os.path.dirname(os.path.abspath(__file__))

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
    epochs:int
    def __init__(self):
        self.train = 1
        self.confusion = 2
        self.execute = 3
        self.epochs = 4
aims = _aims()


def _train_flow(rank:int, world_size:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], optimizer_args, optimizer_kwargs):
    print(f"Training thread#{rank} PID is: {os.getpid()}")
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))
    dataset.sampler.parallel(rank, world_size)
    optimizer = optimizer(model.parameters(), *optimizer_args, **optimizer_kwargs)

    loss_array = train(model, dataset, optimizer, loss_function, echo=(rank == 0))

    loss_array_tensor = torch.tensor(loss_array, dtype=torch.float32, device=rank)
    gathered_loss_tensors = [torch.zeros_like(loss_array_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_loss_tensors, loss_array_tensor)

    torch.distributed.barrier()
    if rank == 0:
        gathered_loss_tensors = [tensor.cpu().numpy() for tensor in gathered_loss_tensors]
        loss_array_reduced = sum(gathered_loss_tensors) / world_size
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            model = model.module.cpu()
            dump((model, loss_array_reduced), file)
    cleanup()
def _train(model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(_train_flow, args=(world_size, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs), nprocs=world_size, join=True)


def _confusion_flow(rank:int, world_size:int, model:torch.nn.Module, dataset:Dataset, classes:int):
    print(f"Confusion thread#{rank} PID is: {os.getpid()}")
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))
    dataset.sampler.parallel(rank, world_size)

    confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))

    confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=rank)
    gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)

    torch.distributed.barrier()
    if rank == 0:
        gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
        confusion_matrix_reduced = sum(gathered_confusion_matrices) / world_size
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump(confusion_matrix_reduced, file)
    cleanup()
def _confusion(model:torch.nn.Module, dataset:Dataset, classes:int):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(_confusion_flow, args=(world_size, model, dataset, classes), nprocs=world_size, join=True)


def _execute_flow(rank:int, world_size:int, model:torch.nn.Module, dataset:DataLoader, extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]], _type:bool):
    print(f"Confusion thread#{rank} PID is: {os.getpid()}")
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))

    results = []
    idx_offset = rank * len(dataset) // world_size
    iterator = tqdm(dataset, disable=rank != 0)
    if _type:
        for idx, (data, correct) in enumerate(iterator, start=idx_offset):
            data:torch.Tensor
            data = data.to(rank)
            correct = correct.to(rank)

            result = model.forward(data)
            result = extract(result, correct)
            results.append((idx, result))
    else:
        for idx, data in enumerate(iterator, start=idx_offset):
            data = data.to(rank)

            result = model.forward(data)
            result = extract(result)
            results.append((idx, result))

    #TODO Переделать, т.к. долго работает
    results_list:list = ([None for _ in range(world_size)] if rank == 0 else None)
    torch.distributed.gather_object(results, results_list, dst=0)

    torch.distributed.barrier()
    if rank == 0:
        results_list:list
        results:list = sum(results_list)
        results.sort(key=lambda x: x[0])
        results = [item[1] for item in results]
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump(results, file)
    cleanup()
def _execute(model:torch.nn.Module, data:Union[Dataset, Iterable[torch.Tensor]], extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]]):
    if isinstance(data, Dataset):
        data.sampler.parallel(0, 8)
        dataloader = data.test
        _type = True
    else:
        class TempDataset(IterableDataset):
            _data:Iterable[torch.Tensor]
            def __init__(self, _data:Iterable[torch.Tensor]):
                self._data = _data
            def __iter__(self):
                for item in self._data:
                    yield item
        dataset = TempDataset(data)
        dataloader = DataLoader(dataset, sampler=DistributedSampler(dataset))
        _type = False
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(_confusion_flow, args=(world_size, model, dataloader, extract, _type), nprocs=world_size, join=True)


def _epochs_flow(rank:int, world_size:int, epochs:int, classes:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], optimizer_args, optimizer_kwargs):
    print(f"Training thread#{rank} PID is: {os.getpid()}")
    if rank == 0:
        if isinstance(model, HybridModel):
            for layer in model._optical_model:
                print(type(layer).__name__)
            print(type(model._detectors).__name__)
            print(type(model._electronic_model).__name__)
        else:
            summary(model, (1, dataset.width, dataset.height), dataset.batch)
    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(rank))
    dataset.sampler.parallel(rank, world_size)
    optimizer = optimizer(model.parameters(), *optimizer_args, **optimizer_kwargs)

    chronograph = Chronograph()
    train_cid = chronograph.add_process('Training')
    confusion_cid = chronograph.add_process('Confusion')
    gather_cid = chronograph.add_process('Gathering')
    dump_cid = chronograph.add_process('Dumping')

    loss_histories = []
    models_history = []
    confusion_matrices_history = []

    chronograph.start(confusion_cid)
    confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))
    confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=rank)
    gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)
    chronograph.end(confusion_cid)

    torch.distributed.barrier()
    if rank == 0:
        gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
        confusion_matrix_reduced = sum(gathered_confusion_matrices) / world_size
        confusion_matrices_history.append(confusion_matrix_reduced)
        print(f"Accuracy in the beggining is {100 * numpy.sum(numpy.diagonal(confusion_matrix_reduced, 0)) / numpy.sum(confusion_matrix_reduced)}")

    for i in range(epochs):
        chronograph.start(train_cid)
        loss_array = train(model, dataset, optimizer, loss_function, echo=(rank == 0), chronograph=chronograph)
        loss_array_tensor = torch.tensor(loss_array, dtype=torch.float32, device=rank)
        gathered_loss_tensors = [torch.zeros_like(loss_array_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_loss_tensors, loss_array_tensor)
        chronograph.end(train_cid)

        chronograph.start(confusion_cid)
        confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))
        confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=rank)
        gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)
        chronograph.end(confusion_cid)

        chronograph.start(gather_cid)
        torch.distributed.barrier()
        if rank == 0:
            gathered_loss_tensors = [tensor.cpu().numpy() for tensor in gathered_loss_tensors]
            loss_array_reduced = sum(gathered_loss_tensors) / world_size
            gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
            confusion_matrix_reduced = sum(gathered_confusion_matrices) / world_size
            loss_histories.append(loss_array_reduced)
            confusion_matrices_history.append(confusion_matrix_reduced)
            models_history.append(deepcopy(model).cpu())
            print(f"Accuracy after epoch {i+1} is {100*numpy.sum(numpy.diagonal(confusion_matrix_reduced, 0))/numpy.sum(confusion_matrix_reduced)}")
        chronograph.end(gather_cid)

    torch.distributed.barrier()
    if rank == 0:
        chronograph.start(dump_cid)
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump((models_history, loss_histories, confusion_matrices_history), file)
        chronograph.end(dump_cid)

    statistic = chronograph.statistic()
    serialized_statistic = dumps(statistic)
    gathered_statistic = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_statistic, serialized_statistic)

    if rank == 0:
        statistics_list = [loads(statistic) for statistic in gathered_statistic if statistic is bytes]
        combined_statistics = sum(statistics_list)
        if hasattr(combined_statistics, 'print'):
            combined_statistics.print()
        else:
            print(combined_statistics)

    cleanup()
def _epochs(epochs:int, classes:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(_epochs_flow, args=(world_size, epochs, classes, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs), nprocs=world_size, join=True)

def main():
    aim = int(sys.argv[1])
    with open(_directory + "/cash/arguments.pkl", 'rb') as file:
        arguments = load(file)
    if aim == aims.epochs:
        _epochs(*(arguments[:-2]), *(arguments[-2]), **(arguments[-1]))
    elif aim == aims.train:
        _train(*(arguments[:-2]), *(arguments[-2]), **(arguments[-1]))
    elif aim == aims.confusion:
        _confusion(*arguments)
    elif aim == aims.execute:
        _execute(*arguments)
    elif aim == -1:
        print("Test cluster run execution")
        for i in range(7):
            print(i)
            time.sleep(2)
    else:
        raise AttributeError('Выбран не верный тип операции')
if __name__ == '__main__':
    main()