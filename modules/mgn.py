import sys
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule
from nets.cloth_model import Model
from datasets.dataset import FlagSimpleDataset
from utils import common

class MGN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epoch = 100
        self.data_dir = "./data/"
        self.num_workers = 0
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.prefetch_factor = 2
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = False
        self.node_info = {'NORMAL': 0}
        self.trajectory_length = 20
        self.split_ratio = 0.85
        self.val_freq = 2000

    def define_dataset(self):
        self.train_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='train', split_ratio=self.split_ratio,  node_info=self.node_info, 
                                    augmentation = True)
        
        self.val_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='valid', split_ratio = self.split_ratio, node_info=self.node_info, 
                                    augmentation = False)


    def define_model(self):
        self.model = Model(self.device, size =3)

    def loss_func(self, data, predictions):
        world_pos = data['world_pos']
        prev_world_pos = data['prev|world_pos']
        target_world_pos = data['target|world_pos']
        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self.model.get_output_normalizer()(target_acceleration)

        node_type = data['node_type'][0]
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value]).to(self.device).int())
        error = (target_normalized[0] - predictions) ** 2
        error = torch.sum(error , dim = 1)
        error = torch.mean(error[loss_mask])
        loss = {'mse_loss': error}
        return loss

    def inspect_dataset(self):
        for idx, (model_inputs, data) in enumerate(self.train_loader):
            model_inputs, data = self.send_to_cuda(model_inputs[0], data[0])
            out = self.model(model_inputs, is_training = True)
            print(out)
            break
        
def main():
    h = MGN()
    h.init(wandb_log=False, project='MeshGraphNet', entity='noldsoul')
    h.define_model()
    # h.inspect_dataset()
    h.train()


if __name__ == "__main__":
    main()
