import numpy
import torch

from typing import Callable
from tqdm import tqdm
from time import time

from utilities import *

def train(model:torch.nn.Module, dataset:Dataset, optimizer:torch.optim.Optimizer, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], echo:bool=True):

    if hasattr(model, 'device'): device = model.device
    else: device = next(iter(model.parameters())).device

    model.train()
    history = numpy.zeros((len(dataset.train)))

    time_start = time()
    running_loss = 0
    running_loss_proportion = 0.2
    regression = 0

    iterator = tqdm(dataset.train, disable=not echo, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | {elapsed}m - {remaining}m")

    for i, (images, labels) in enumerate(iterator):
        labels = labels.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)

        results = model(images)

        optimizer.zero_grad()
        loss = loss_function(results, labels)
        loss.backward()
        optimizer.step()

        history[i] = loss.item()
        if running_loss == 0:   running_loss = loss.item()
        else:                   running_loss = (1.0 - running_loss_proportion)*running_loss + running_loss_proportion*loss.item()

        if time() - time_start >= 10.0 and i >= 3:
            time_start = time()
            history_slice = history[:i+1]
            iteration_slice = numpy.arange(0, i+1)
            k, b = numpy.polyfit(iteration_slice, history_slice, 1)
            regression = k.item() * 1000.0
        iterator.set_description(f"RL:{scientific(float(running_loss),'',3,space='')}, RPI1K:{scientific(float(regression),'',2,space='')}")
    model.eval()
    return history

def confusion(model:torch.nn.Module, dataset:Dataset, classes:int=10, echo:bool=True):
    if hasattr(model, 'device'): device = model.device
    else: device = next(iter(model.parameters())).device
    
    model.eval()
    matrix = numpy.zeros((classes, classes))
    with torch.no_grad():
        iterator = tqdm(dataset.test, disable=not echo)
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)
            values, indexes = torch.max(model(images), dim=1)
            for label, index in zip(labels, indexes):
                matrix[label.item(), index.item()] += 1
    return matrix