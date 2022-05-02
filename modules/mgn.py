import sys
import os
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

class MGN(BaseModule):
    def __init__(self, parser):
        super().__init__()
        self.epoch = parser.epochs
        self.data_dir = parser.data_dir
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
        self.val_freq = 2000

    def define_dataset(self):
        self.train_dataset =  FlagSimpleDataset(device=self.device, 
                                    path=os.path.join(self.data_dir, 'flag_simple'), history = True , 
                                    split='train', split_ratio=self.split_ratio,  node_info=self.node_info, 
                                    augmentation = True)
        
        self.val_dataset =  FlagSimpleDataset(device=self.device, 
                                    path=os.path.join(self.data_dir, 'flag_simple'), history = True , 
                                    split='valid', split_ratio = self.split_ratio, node_info=self.node_info, 
                                    augmentation = True)

    def define_model(self):
        self.model = Model(self.device, size =3, batchsize=self.train_batch_size)

    def loss_func(self, data, predictions):
        # world_pos = data['world_pos']
        # prev_world_pos = data['prev|world_pos']
        # target_world_pos = data['target|world_pos']
        cur_position = data['world_pos']
        prev_position = data['prev|world_pos']
        target_position = data['target|world_pos']
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self.model.get_output_normalizer()(target_acceleration)

        node_type = data['node_type']
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
            
    def rollout(self):
        import trimesh
        
        dump_path = './output/simple_test_run_val'
        os.makedirs(dump_path, exist_ok = True)
        self.load_checkpoint('./weights/simple_test_run_weights/model_70.pt')
        self.model.to(torch.device("cuda:0"))
        self.model.eval()
        
        data0, data1 = next(iter(self.val_loader))
        model_inputs = data0[0]
        data = data1[0]
        for i in trange(150):
            if i == 0:
                model_inputs, data = self.send_to_cuda(model_inputs, data)
                with torch.no_grad():
                    predictions = self.model(model_inputs)

                current_coord = predictions.detach().cpu()
                prev_coord = model_inputs['world_pos']

            else:
                model_inputs['world_pos'] = current_coord
                model_inputs['prev|world_pos'] = prev_coord
                model_inputs, data = self.send_to_cuda(model_inputs, data)
                with torch.no_grad():
                    predictions = self.model(model_inputs)
                
                prev_coord = current_coord.detach().cpu()
                current_coord = predictions

            error = (model_inputs['target|world_pos'] - predictions) ** 2
            error = torch.sum(error , dim = 2)
            error = torch.mean(error)
            print("error", error)

            faces = data['cells'].detach().cpu().numpy()
            mesh = trimesh.Trimesh(predictions[0].detach().cpu().numpy(), faces[0])
            mesh.export(os.path.join(dump_path, f'{i}.ply'))

    def inference(self):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        import trimesh
        
        dump_path = './output/simple_test_run'
        os.makedirs(dump_path, exist_ok = True)
        self.load_checkpoint('./weights/model_70.pt')
        self.model.to(torch.device("cuda:0"))
        self.model.eval()
        
        for idx, (data0, data1) in tqdm(enumerate(self.train_loader)):
            for i in range(len(data1))[:self.trajectory_length]:
                model_inputs = data0[i]
                data = data1[i]
                cells = data['cells']
                print(cells)
                exit()
                model_inputs, data = self.send_to_cuda(model_inputs, data)
                with torch.no_grad():
                    predictions = self.model(model_inputs).detach().cpu().numpy()
                faces = data['cells'].detach().cpu().numpy()
                mesh = trimesh.Trimesh(predictions[0], faces[0])
                mesh.export(os.path.join(dump_path, f'{i}.ply'))
            break
        
def main():
    parser = options.get_parser()
    h = MGN(parser)
    h.init(wandb_log=False, project='MeshGraphNet', entity='noldsoul')
    h.define_model()
    #h.inspect_dataset()
    #h.train()
    #h.inference()
    h.rollout()


if __name__ == "__main__":
    main()
