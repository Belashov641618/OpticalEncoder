import torch
import pytorch_lightning as lightning

from typing import Callable, Union, Iterable, Optional, Any

from utilities.methods import normilize


class Basic(lightning.LightningModule):
    _model:torch.nn.Module
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model:torch.nn.Module):
        self._model = model

    _loss_function:Optional[Callable[[torch.Tensor,torch.Tensor],torch.Tensor]]
    @property
    def loss_function(self):
        return self._loss_function
    @loss_function.setter
    def loss_function(self, loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]):
        self._loss_function = loss_function

    _optimizers:Optional[list[torch.optim.Optimizer]]
    def set_optimizers(self, *optimizers:torch.optim.Optimizer):
        self._optimizers = [*optimizers]
    def add_optimizers(self, *optimizers:torch.optim.Optimizer):
        if self._optimizers is None:
            self.set_optimizers(*optimizers)
        else:
            self._optimizers += [*optimizers]
    def list_optimizers(self):
        return self._optimizers
    def remove_optimizers(self, *optimizers:Union[int,torch.optim.Optimizer]):
        indexes = set()
        for optimizer in optimizers:
            if isinstance(optimizer, int) and optimizer not in indexes:
                indexes.add(int)
            elif isinstance(optimizer, torch.optim.Optimizer):
                id = self._optimizers.index(optimizer)
                if id not in indexes:
                    indexes.add(id)
            else:
                raise TypeError
        self._optimizers = [optimizer for i, optimizer in enumerate(self._optimizers) if i not in indexes]


    _schedulers:Optional[list[torch.optim.lr_scheduler.LRScheduler]]
    def set_schedulers(self, *schedulers:torch.optim.lr_scheduler.LRScheduler):
        self._schedulers = [*schedulers]
    def add_schedulers(self, *schedulers:torch.optim.lr_scheduler.LRScheduler):
        if self._schedulers is None:
            self.set_schedulers(*schedulers)
        else:
            self._schedulers += [*schedulers]
    def list_schedulers(self):
        return self._schedulers
    def remove_schedulers(self, *schedulers:Union[int,torch.optim.lr_scheduler.LRScheduler]):
        indexes = set()
        for scheduler in schedulers:
            if isinstance(scheduler, int) and scheduler not in indexes:
                indexes.add(scheduler)
            elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                id = self._schedulers.index(scheduler)
                if id not in indexes:
                    indexes.add(id)
            else:
                raise TypeError
        self._schedulers = [scheduler for i, scheduler in enumerate(self._schedulers) if i not in indexes]

    def __init__(self,
                 model:torch.nn.Module,
                 loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]=None,
                 optimizers:Union[torch.optim.Optimizer,Iterable[torch.optim.Optimizer]]=None,
                 schedulers:Union[torch.optim.lr_scheduler.LRScheduler,Iterable[torch.optim.lr_scheduler.LRScheduler]]=None
                 ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        if optimizers is None: self._optimizers = None
        else: self.set_optimizers(optimizers)
        if schedulers is None: self._schedulers = None
        else: self.set_schedulers(schedulers)

    def forward(self, field:torch.Tensor):
        return self.model.forward(field)

    def training_step(self, batch:tuple[torch.Tensor,torch.Tensor], index:int):
        data, reference = batch
        loss = self._loss_function(self._model.forward(data), reference)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self._optimizers, self._schedulers


class Classification(Basic):
    def __init__(self,
                 model:torch.nn.Module,
                 loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]=None,
                 optimizers:Union[torch.optim.Optimizer,Iterable[torch.optim.Optimizer]]=None,
                 schedulers:Union[torch.optim.lr_scheduler.LRScheduler,Iterable[torch.optim.lr_scheduler.LRScheduler]]=None
                 ):
        super().__init__(model, loss_function, optimizers, schedulers)

    def validation_step(self, batch:tuple[torch.Tensor,torch.Tensor], index:int):
        data, reference = batch
        result = self._model.forward(data)

        loss = self.loss_function(result, reference)

        classes = reference.shape[-1]
        confusion = torch.zeros((classes, classes))
        values, indexes = torch.max(result, dim=1)
        for truth, index in zip(reference, indexes):
            confusion[truth.item(), index.item()] += 1

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', 100*torch.sum(torch.diagonal(confusion,0))/torch.sum(confusion), prog_bar=True)

        return  {'test_loss':loss, 'confusion':confusion}

    # noinspection PyTypeChecker
    def validation_epoch_end(self, outputs:list):
        mean_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        confusion = sum(output['confusion'] for output in outputs)
        accuracy = 100*torch.sum(torch.diagonal(confusion,0))/torch.sum(confusion)

        self.log('mean_test_loss', mean_loss, prog_bar=True)
        self.log('mean_test_accuracy', accuracy, prog_bar=True)


class ImageToImage(Basic):
    def __init__(self,
                 model:torch.nn.Module,
                 loss_function:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]=None,
                 optimizers:Union[torch.optim.Optimizer,Iterable[torch.optim.Optimizer]]=None,
                 schedulers:Union[torch.optim.lr_scheduler.LRScheduler,Iterable[torch.optim.lr_scheduler.LRScheduler]]=None
                 ):
        super().__init__(model, loss_function, optimizers, schedulers)

    def validation_step(self, batch:tuple[torch.Tensor,torch.Tensor], index:int):
        images, references = batch
        images = self._model.forward(images)

        if torch.is_complex(images): images = torch.abs(images)
        if torch.is_complex(references): references = torch.abs(references)

        loss = self.loss_function(images, references)

        images = normilize(images)
        references = normilize(references)

        mae = torch.mean(torch.abs(images - references))
        mse = torch.mean((images - references)**2)
        psnr = torch.mean(20*torch.log10(1.0/((images - references)**2)))

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae',  mae,  prog_bar=True)
        self.log('test_mse',  mse,  prog_bar=True)
        self.log('test_psnr', psnr, prog_bar=True)

        return dict(test_loss=loss, mae=mae, mse=mse, psnr=psnr)

    # noinspection PyTypeChecker
    def validation_epoch_end(self, outputs:list):
        mean_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        mean_mae  = torch.stack([output['test_mae']  for output in outputs]).mean()
        mean_mse  = torch.stack([output['test_mse']  for output in outputs]).mean()
        mean_psnr = torch.stack([output['test_psnr'] for output in outputs]).mean()

        self.log('mean_test_loss', mean_loss, prog_bar=True)
        self.log('mean_test_mae',  mean_mae,  prog_bar=True)
        self.log('mean_test_mse',  mean_mse,  prog_bar=True)
        self.log('mean_test_psnr', mean_psnr, prog_bar=True)