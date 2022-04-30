import sys
import os
import random
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule
from nets.cloth_model import Model
from datasets.dataset import FlagSimpleDataset
from utils import common, options
from utils.common import sample_points_triangulation

class MGN(BaseModule):
    def __init__(self, parser):
        super().__init__()
        self.epoch = parser.epochs
        self.data_dir = parser.data_dir
        self.save_dir = parser.save_dir
        self.if_sampling = parser.if_sampling
        self.sample_n_points = parser.sample_n_points
        self.num_workers = parser.num_worker
        self.train_batch_size = parser.train_batch_size
        self.val_batch_size = parser.val_batch_size
        self.prefetch_factor = parser.prefetch_factor
        self.learning_rate = parser.learning_rate
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = False
        self.node_info = {'NORMAL': 0}
        self.trajectory_length = parser.trajectory_length
        self.split_ratio = parser.split_ratio
        self.val_freq = 2

    def define_dataset(self):
        self.train_dataset =  FlagSimpleDataset(device=self.device, 
                                    path=os.path.join(self.data_dir, 'flag_simple'), history = True , 
                                    split='train', split_ratio=self.split_ratio,  node_info=self.node_info, 
                                    augmentation = True)

        self.val_dataset =  FlagSimpleDataset(device=self.device, 
                                    path=os.path.join(self.data_dir, 'flag_simple'), history = True , 
                                    split='valid', split_ratio = self.split_ratio, node_info=self.node_info, 
                                    augmentation = False)


    def define_model(self):
        self.model = Model(self.device, size =3, batchsize=self.train_batch_size, if_sampling=self.if_sampling, sample_n_points=self.sample_n_points)

    def loss_func(self, data, predictions, sampled_inputs):
        # world_pos = data['world_pos']
        # prev_world_pos = data['prev|world_pos']
        # target_world_pos = data['target|world_pos']
        
        # print(sampled_inputs['picked_indexes'])
        # cur_position = data['target|world_pos']
        # cp=[]
        # for batch_no in range(data['target|world_pos'].shape[0]):
        #     cp.append(torch.index_select(input=cur_position[batch_no,:,:], dim=0, index=sampled_inputs['picked_indexes'][batch_no].to(self.device)).unsqueeze(0))
        #     print(cp[batch_no][0][:10])
        #     print()
        #     print(sampled_inputs['target|world_pos'][0][:10])
        #     exit()
        if sampled_inputs != "None":
            cur_position = sampled_inputs['world_pos']
            prev_position = sampled_inputs['prev|world_pos']
            target_position = sampled_inputs['target|world_pos']
            node_type = sampled_inputs['node_type']
        if sampled_inputs == "None":
            cur_position = data['world_pos']
            prev_position = data['prev|world_pos']
            target_position = data['target|world_pos']
            node_type = data['node_type']

        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self.model.get_output_normalizer()(target_acceleration)

        loss_mask = torch.eq(node_type[:, :, 0], torch.tensor([common.NodeType.NORMAL.value]).to(self.device).int())
        error = (target_normalized - predictions) ** 2
        error = torch.sum(error , dim = 2)
        error = torch.mean(error[loss_mask])
        
        loss = {'mse_loss': error}
        return loss
    
    def inspect_dataset(self):
        self.model.to(torch.device("cuda:0"))
        self.define_optimizer()
        for idx, (data0, data1) in tqdm(enumerate(self.train_loader)):
            for i in range(len(data1))[:self.trajectory_length]:
                model_inputs = data0[i]
                data = data1[i]
                model_inputs, data = self.send_to_cuda(model_inputs, data)
                predictions = self.model(model_inputs)
                self.optimizer.zero_grad()
                losses = self.loss_func(data, predictions)
                total_loss = sum(losses.values())
                total_loss.backward()
                self.optimizer.step()
                #self.update_loss_meter(losses)
            #break
        
def main():
    parser = options.get_parser()
    h = MGN(parser)
    h.init(wandb_log=False, project='MeshGraphNet', entity='noldsoul')
    h.define_model()
    #h.inspect_dataset()
    h.train()


if __name__ == "__main__":
    main()
