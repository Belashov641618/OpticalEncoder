import os
import sys
import time
import datetime
import numpy
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import IterableDataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torchsummary import summary
from pickle import load, dump, dumps, loads
from copy import deepcopy
from typing import Callable, Type, Union, Iterable, Any
from tqdm import tqdm

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utilities.training import train, confusion
from utilities import *
from elements.composition import HybridModel


_directory = os.path.dirname(os.path.abspath(__file__))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['NCCL_SOCKET_NTO'] = f'{5*60}'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1200))
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


def _train_flow(rank:int, gpus:tuple[int,...], model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], optimizer_args, optimizer_kwargs):
    print(f"Training thread#{rank} PID is: {os.getpid()}")
    setup(rank, len(gpus))
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(gpus[rank]))
    dataset.sampler.parallel(rank, len(gpus))
    optimizer = optimizer(model.parameters(), *optimizer_args, **optimizer_kwargs)

    loss_array = train(model, dataset, optimizer, loss_function, echo=(rank == 0))

    loss_array_tensor = torch.tensor(loss_array, dtype=torch.float32, device=gpus[rank])
    gathered_loss_tensors = [torch.zeros_like(loss_array_tensor) for _ in range(len(gpus))]
    torch.distributed.all_gather(gathered_loss_tensors, loss_array_tensor)

    torch.distributed.barrier()
    if rank == 0:
        gathered_loss_tensors = [tensor.cpu().numpy() for tensor in gathered_loss_tensors]
        loss_array_reduced = sum(gathered_loss_tensors) / len(gpus)
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            model = model.module.cpu()
            dump((model, loss_array_reduced), file)
    cleanup()
def _train(gpus:tuple[int,...], model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    torch.multiprocessing.spawn(_train_flow, args=(gpus, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs), nprocs=len(gpus), join=True)


def _confusion_flow(rank:int, gpus:tuple[int,...], model:torch.nn.Module, dataset:Dataset, classes:int):
    print(f"Confusion thread#{rank} PID is: {os.getpid()}")
    setup(rank, len(gpus))
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(gpus[rank]))
    dataset.sampler.parallel(rank, len(gpus))

    confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))

    confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=gpus[rank])
    gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(len(gpus))]
    torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)

    torch.distributed.barrier()
    if rank == 0:
        gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
        confusion_matrix_reduced = sum(gathered_confusion_matrices) / len(gpus)
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump(confusion_matrix_reduced, file)
    cleanup()
def _confusion(gpus:tuple[int,...], model:torch.nn.Module, dataset:Dataset, classes:int):
    torch.multiprocessing.spawn(_confusion_flow, args=(gpus, model, dataset, classes), nprocs=len(gpus), join=True)


def _execute_flow(rank:int, gpus:tuple[int,...], model:torch.nn.Module, dataset:DataLoader, extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]], _type:bool):
    print(f"Confusion thread#{rank} PID is: {os.getpid()}")
    setup(rank, len(gpus))
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(gpus[rank]))

    results = []
    idx_offset = rank * len(dataset) // len(gpus)
    iterator = tqdm(dataset, disable=rank != 0)
    if _type:
        for idx, (data, correct) in enumerate(iterator, start=idx_offset):
            data:torch.Tensor
            data = data.to(gpus[rank])
            correct = correct.to(gpus[rank])

            result = model.forward(data)
            result = extract(result, correct)
            results.append((idx, result))
    else:
        for idx, data in enumerate(iterator, start=idx_offset):
            data = data.to(gpus[rank])

            result = model.forward(data)
            result = extract(result)
            results.append((idx, result))

    #TODO Переделать, т.к. долго работает
    results_list:list = ([None for _ in range(len(gpus))] if rank == 0 else None)
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
def _execute(gpus:tuple[int,...], model:torch.nn.Module, data:Union[Dataset, Iterable[torch.Tensor]], extract:Union[Callable[[torch.Tensor],Any],Callable[[torch.Tensor,torch.Tensor],Any]]):
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
    torch.multiprocessing.spawn(_confusion_flow, args=(gpus, model, dataloader, extract, _type), nprocs=len(gpus), join=True)


def _epochs_flow(rank:int, gpus:tuple[int,...], epochs:int, classes:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], optimizer_args, optimizer_kwargs):
    # print(f"Training thread#{rank} PID is: {os.getpid()}")
    # if rank == 0:
        # if isinstance(model, HybridModel):
            # for layer in model._optical_model:
                # print(type(layer).__name__)
            # print(type(model._detectors).__name__)
            # print(type(model._electronic_model).__name__)
        # else:
            # summary(model, (1, dataset.width, dataset.height), dataset.batch)
    setup(rank, len(gpus))
    model = torch.nn.parallel.DistributedDataParallel(deepcopy(model).to(gpus[rank]))
    dataset.sampler.parallel(rank, len(gpus))
    optimizer = optimizer(model.parameters(), *optimizer_args, **optimizer_kwargs)

    loss_histories = []
    models_history = []
    confusion_matrices_history = []

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True, with_flops=True, with_modules=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log{rank}"),) as prof:
    if dataset.confusion_allowed:
        confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))
        confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=gpus[rank])
        gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(len(gpus))]
        torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)

        torch.distributed.barrier()
        if rank == 0:
            gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
            confusion_matrix_reduced = sum(gathered_confusion_matrices) / len(gpus)
            confusion_matrices_history.append(confusion_matrix_reduced)
            print(f"Accuracy in the beginning is {100 * numpy.sum(numpy.diagonal(confusion_matrix_reduced, 0)) / numpy.sum(confusion_matrix_reduced)}")
        torch.distributed.barrier()
    
    for i in range(epochs):
        # print(f"| {time.time()} : {rank} : Start Training")
        loss_array = train(model, dataset, optimizer, loss_function, echo=(rank == 0))
        loss_array_tensor = torch.tensor(loss_array, dtype=torch.float32, device=gpus[rank])
        gathered_loss_tensors = [torch.zeros_like(loss_array_tensor) for _ in range(len(gpus))]
        # print(f"| {time.time()} : {rank} : Training finished, gathering")
        torch.distributed.all_gather(gathered_loss_tensors, loss_array_tensor)
        
        if dataset.confusion_allowed:
            # print(f"| {time.time()} : {rank} : Gathering finished, start validation")
            confusion_matrix = confusion(model, dataset, classes, echo=(rank == 0))
            confusion_matrices_tensor = torch.tensor(confusion_matrix, dtype=torch.float32, device=gpus[rank])
            gathered_confusion_matrices = [torch.zeros_like(confusion_matrices_tensor) for _ in range(len(gpus))]
            # print(f"| {time.time()} : {rank} : Validation finished, gathering")
            torch.distributed.all_gather(gathered_confusion_matrices, confusion_matrices_tensor)

        # print(f"| {time.time()} : {rank} : Gathering after validation finished, accquiring barrier")
        torch.distributed.barrier()
        # print(f"| {time.time()} : {rank} : Barrier released, starting calculations")
        if rank == 0:
            # print(f"| {time.time()} : {rank} : Meaning loss history")
            gathered_loss_tensors = [tensor.cpu().numpy() for tensor in gathered_loss_tensors]
            loss_array_reduced = sum(gathered_loss_tensors) / len(gpus)
            loss_histories.append(loss_array_reduced)
            if dataset.confusion_allowed:
                # print(f"| {time.time()} : {rank} : Meaning cofusion matrixes")
                gathered_confusion_matrices = [tensor.cpu().numpy() for tensor in gathered_confusion_matrices]
                confusion_matrix_reduced = sum(gathered_confusion_matrices) / len(gpus)
                confusion_matrices_history.append(confusion_matrix_reduced)
            # print(f"| {time.time()} : {rank} : Appending models history")
            model_copy = deepcopy(model.module)
            model_copy = model_copy.cpu()
            models_history.append(model_copy)
            # print(f"| {time.time()} : {rank} : Printing results")
            if dataset.confusion_allowed:
                print(f"Accuracy after epoch {i+1} is {100*numpy.sum(numpy.diagonal(confusion_matrix_reduced, 0))/numpy.sum(confusion_matrix_reduced)}")
            else:
                print(f"Mean loss after epoch {i+1} is {sum(loss_array_reduced)/len(loss_array_reduced)}")
            if i != epochs-1:
                # print(f"| {time.time()} : {rank} : Writing file")
                with open(_directory + "/cash/checkpoints.pkl", 'wb') as file:
                    dump((models_history, loss_histories, confusion_matrices_history), file)
        torch.distributed.barrier()
    torch.distributed.barrier()
    
    # print(f"| {time.time()} : {rank} : Calculations finished")
    if rank == 0:
        # print(f"| {time.time()} : {rank} : Writing file")
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump((models_history, loss_histories, confusion_matrices_history), file)
            
    # print(f"| {time.time()} : {rank} : Barrier accquiring before cleaning up")
    torch.distributed.barrier()
    # print(f"| {time.time()} : {rank} : Barrier released, cleaning up")
    cleanup()
def _epochs(gpus:tuple[int,...], epochs:int, classes:int, model:torch.nn.Module, dataset:Dataset, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], optimizer:Type[torch.optim.Optimizer], *optimizer_args, **optimizer_kwargs):
    print(f"Training main thread PID is: {os.getpid()}")
    exception = None
    models_history, loss_histories, confusion_matrices_history = [], [], []
    merge_histories:bool = False
    while True:
        try:
            if os.path.exists(_directory + "/cash/checkpoints.pkl"):
                os.remove(_directory + "/cash/checkpoints.pkl")
            torch.multiprocessing.spawn(_epochs_flow, args=(gpus, epochs, classes, model, dataset, loss_function, optimizer, optimizer_args, optimizer_kwargs), nprocs=len(gpus), join=True)
        except Exception as error:
            torch.cuda.empty_cache()
            print(error)
            merge_histories = True
            exception = error
            if not os.path.exists(_directory + "/cash/checkpoints.pkl"):
                raise error
            with open(_directory + "/cash/checkpoints.pkl", 'rb') as file:
                mh, lh, cmh = load(file)
            models_history += mh
            loss_histories += lh
            confusion_matrices_history += cmh
            model = models_history[-1]
            epochs -= len(loss_histories)
        break
    if merge_histories:
        with open(_directory + "/cash/results.pkl", 'rb') as file:
            mh, lh, cmh = load(file)
            models_history += mh
            loss_histories += lh
            confusion_matrices_history += cmh
        with open(_directory + "/cash/results.pkl", 'wb') as file:
            dump((models_history, loss_histories, confusion_matrices_history), file)

def main():
    aim = int(sys.argv[1])
    with open(_directory + "/cash/arguments.pkl", 'rb') as file:
        arguments = load(file)
        gpus, *arguments = arguments
    if aim == aims.epochs:
        _epochs(gpus, *(arguments[:-2]), *(arguments[-2]), **(arguments[-1]))
    elif aim == aims.train:
        _train(gpus, *(arguments[:-2]), *(arguments[-2]), **(arguments[-1]))
    elif aim == aims.confusion:
        _confusion(gpus, *arguments)
    elif aim == aims.execute:
        _execute(gpus, *arguments)
    elif aim == -1:
        print("Test cluster run execution")
        for i in range(7):
            print(i)
            time.sleep(2)
    else:
        raise AttributeError('Выбран не верный тип операции')
if __name__ == '__main__':
    main()