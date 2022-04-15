import sys
#sys.path.append('../')
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule
from nets.cloth_model import Model
from datasets.dataset import FlagSimpleDataset

class MGN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epoch = 100
        self.data_dir = "./data/"
        self.num_workers = 0
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.prefetch_factor = 2
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = False
        self.node_info = {'NORMAL': 0}

    def define_dataset(self):
        self.train_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='train', node_info=self.node_info, 
                                    augmentation = True)
        
        self.val_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='train', node_info=self.node_info, 
                                    augmentation = True)


    def define_model(self):
        self.model = Model(self.device, size =3)

    def loss_function(self, model_inputs, data):
        world_pos = data['world_pos']
        prev_world_pos = data['prev|world_pos']
        target_world_pos = data['target|world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = model.get_output_normalizer()(target_acceleration).to(device)

        # build loss
        node_type = data['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean()
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
    h.inspect_dataset()
    # h.train()


if __name__ == "__main__":
    main()