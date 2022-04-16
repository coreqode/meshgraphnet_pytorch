from collections import defaultdict
import math
from tqdm import tqdm, tnrange, tqdm_notebook
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from progress.bar import Bar
import wandb

class BaseModule:
    def __init__(self):
        self.epochs = 100
        self.patience = 20
        self.start_epoch = 0
        self.train_batch_size = 4
        self.val_batch_size = 4
        self.num_workers = 1
        self.val_freq = 1
        self.learning_rate = 0.0001
        self.save_freq = 2
        self.wandb_log_interval = 1000
        self.collate_fn = None
        self.train_shuffle = True
        self.val_shuffle = False
        self.prefetch_factor = 2
        self.pin_memory = True

    def init(self, wandb_log = False, project =None, entity = None):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            self.device = 'cpu'
            print('CUDA is not available.  Training on CPU ...')
        else:
            self.device = 'cuda:0'
            print('CUDA is available!  Training on GPU ...')

        self.define_dataset()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            prefetch_factor = self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn = self.collate_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_workers,
            prefetch_factor = self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn = self.collate_fn
        )
        self.trainset_length = len(self.train_loader)
        self.valset_length = len(self.val_loader)
        self.define_metrics_meter()

        if wandb_log:
            self.wandb = wandb
            wandb.init(project=project, entity=entity)

        else:
            self.wandb = None

    def dump_image_local(self, image, name, path):
        image = np.array(image)
        image = Image.fromarray(image.astype(np.uint8))
        image.save(os.path.join(path, f"{name}.png"))

    def define_model(self):
        pass

    def loss_func(self):
        pass

    def define_dataset(self):
        pass

    def earlystopping(self):
        pass

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def define_scheduler(self):
        pass

    def train_step_zero(self, model_inputs, data):
        self.model.train()
        predictions = self.model(model_inputs)
        losses = self.loss_func(data, predictions)
        return losses

    def train_step(self, model_inputs, data):
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(model_inputs)
        losses = self.loss_func(data, predictions)
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()
        self.update_loss_meter(losses)

        return predictions

    def val_step(self, model_inputs, data):
        self.model.eval()
        predictions = self.model(model_inputs)
        losses = self.loss_func(data, predictions)
        self.update_loss_meter(losses)
        return losses

    def update_loss_meter(self, losses):
        for name, meter in self.loss_meter.items():
            meter.append(losses[name])

    def print_loss_metrics(self):
        strings = []
        for name, loss in self.loss_meter.items():
            _loss = torch.mean(torch.FloatTensor(loss))
            _loss = _loss.detach().cpu().numpy()

            str = f"{name}: {_loss:.4f}"
            strings.append(str)
        lossstr = "| ".join(strings)

        if self.metrics_meter:
            metricsstr = "| ".join(
                [f"{name}: {metric.result().numpy():.3f} " for name,
                 metric in self.metrics_meter.items()]
            )
        else:
            metricsstr = ""
        return lossstr + ("| " + metricsstr)

    def define_loss_meter(self, losses):
        self.loss_meter = {lossname: []
                           for lossname in losses.keys()}

    def update_metrics_meter(self):
        pass

    def define_metrics_meter(self):
        self.metrics_meter = None

    def initepoch(self):
        self.loss_meter = {lossname: []
                           for lossname in self.loss_meter.keys()}
        if self.metrics_meter:
            self.metrics_meter = {metric_name: []
                               for metric_name in self.metrics_meter.keys()}

    def send_to_cuda(self, model_inputs, data):
        if self.device == 'cpu':
            return model_inputs, data
        else:
            for name, _ in model_inputs.items():
                model_inputs[name] = model_inputs[name].to(self.device)

            for name, _ in data.items():
                data[name] = data[name].to(self.device)
            return model_inputs, data

    def train(self):

        self.define_model()
        self.define_optimizer()
        self.define_scheduler()
        self.model.to(self.device)

        if self.wandb:
            self.wandb.watch(self.model)

        print("")
        print(f"Trainset Volume : {self.trainset_length}")
        print(f"Valset Volume : {self.valset_length}")
        print("")

        for epoch in range(self.start_epoch, self.epochs + 1):


            if epoch % self.save_freq == 0:
                if not os.path.isdir('./weights'):
                    os.system('mkdir weights')
                torch.save(self.model.state_dict(), f'./weights/model_{epoch}.pt' )
                if self.wandb:
                    self.wandb.save(f'./weights/model_{epoch}.pt')
            # ----------------------Training Loop -----------------------------#
            # -----------------------------------------------------------------#
            num_iterations = self.trainset_length
            bar = Bar(f"Ep : {epoch} | Training :", max=num_iterations)
            for batch_idx, (data0, data1) in enumerate(self.train_loader):
                if isinstance(data1, list):
                    for i in range(len(data1)):
                        model_inputs = data0[i]
                        data = data1[i]
                        model_inputs, data = self.send_to_cuda(model_inputs, data)

                        if (batch_idx == 0) and (epoch == self.start_epoch):
                            losses = self.train_step_zero(model_inputs, data)
                            self.define_loss_meter(losses)

                        predictions = self.train_step(model_inputs, data)

                        if self.wandb and (batch_idx % self.wandb_log_interval == 0):
                            for name, loss in self.loss_meter.items():
                                _loss = torch.mean(torch.FloatTensor(loss))
                                _loss = _loss.detach().cpu().numpy()
                                self.wandb.log({name: _loss})

                Bar.suffix = f"{batch_idx+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | {self.print_loss_metrics()}"
                bar.next()
            bar.finish()

            self.initepoch();

            if epoch % self.val_freq == 0:
                # ----------------------Validation Loop -----------------------------#
                # -------------------------------------------------------------------#
                num_iterations = self.valset_length
                bar = Bar(f"Ep : {epoch} | Validation :", max=num_iterations)
                with torch.no_grad():
                    for batch_idx, (model_inputs, data) in enumerate(self.val_loader):
                        model_inputs, data = self.send_to_cuda(model_inputs, data)
                        predictions = self.val_step(model_inputs, data)

                        if self.wandb and (batch_idx % self.wandb_log_interval == 0):
                            for name, loss in self.loss_meter.items():
                                _loss = torch.mean(torch.FloatTensor(loss))
                                _loss = _loss.detach().cpu().numpy()
                                self.wandb.log({f'{name}_val': _loss})

                    Bar.suffix = f"{batch_idx+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | {self.print_loss_metrics()}"
                    bar.next()
                bar.finish()



            self.initepoch()
