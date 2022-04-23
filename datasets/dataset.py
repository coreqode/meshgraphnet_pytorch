import torch
from torch.utils.data import Dataset, IterableDataset
from tfrecord.torch.dataset import TFRecordDataset
import os
import json
import numpy as np
import glob
from natsort import natsorted
import random

class FlagSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, device, path, split, split_ratio, node_info, use_tfrecord = False, history=False, augmentation=True):
        self.path = path
        self.split = split
        self.history = history
        self.augmentation = augmentation
        self.device = device
        self.node_info = node_info
        self.split_ratio = split_ratio

        if use_tfrecord:
            self.dataset = self.get_tfrecord_dataset()
            self.get_meta()
        else:
            filepath  = os.path.join(path, '*.npz')
            self.all_files = natsorted(glob.glob(filepath))
        
        if split == 'train':
            random.shuffle(self.all_files)
            self.all_files = self.all_files[:int(self.split_ratio * len(self.all_files))]
        elif split == 'valid':
            self.all_files = self.all_files[int(self.split_ratio * len(self.all_files)):]
    
    def get_tfrecord_dataset(self):
        if split == 'train':
            tfrecord_path = os.path.join(path, "train.tfrecord")
            index_path = os.path.join(path, "train.idx")
        elif split == 'val':
            tfrecord_path = os.path.join(path, "valid.tfrecord")
            index_path = os.path.join(path,  "valid.idx")

        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None)
        loader = torch.utils.data.DataLoader(tf_dataset, batch_size=1)
        self.dataset = list(iter(loader))
        return self.dataset

    def get_meta(self):
        with open(os.path.join(self.path, 'meta.json'), 'r') as fp:
            self.meta = json.loads(fp.read())
            self.shapes = {}
            self.dtypes = {}
            self.types = {}
            for key, field in self.meta['features'].items():
                self.shapes[key] = field['shape']
                self.dtypes[key] = field['dtype']
                self.types[key] = field['type']

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        sample_path = self.all_files[idx]
        sample = np.load(sample_path)
        node_type = sample['node_type']
        world_pos = sample['world_pos']
        mesh_pos = sample['mesh_pos']
        cells = sample['cells'][0]

        if self.history:
            cells = cells[1: -1]
            mesh_pos = mesh_pos[1: -1]
            node_type = node_type[1: -1]
            prev_world_pos = world_pos[0:-2]
            target_world_pos = world_pos[2:]
            world_pos = world_pos[1: -1]

        data = {'world_pos': world_pos, 'mesh_pos': mesh_pos, 'node_type': node_type, 'cells': cells, 
                    'prev|world_pos': prev_world_pos, 'target|world_pos': target_world_pos}

        if self.augmentation:
            data = self.split_and_preprocess()(data)

        return data, data



    def split_and_preprocess(self, noise_scale = 0.003, noise_gamma = 0.1):
        def add_noise(frame):
            noise = np.random.normal(0, noise_scale, frame['world_pos'].shape)
            mask = (frame['node_type'] == self.node_info['NORMAL']).astype(np.int32)
            mask = np.hstack([mask] * 3)
            noise = np.where(mask, noise, 0)
            frame['world_pos'] += noise
            frame['target|world_pos'] += (1.0 - noise_gamma) * noise
            return frame

        def element_operation(trajectory):
            world_pos = trajectory['world_pos']
            mesh_pos = trajectory['mesh_pos']
            node_type = trajectory['node_type']
            cells = trajectory['cells']
            target_world_pos = trajectory['target|world_pos']
            prev_world_pos = trajectory['prev|world_pos']
            trajectory_steps = []
            for i in range(399):
                wp = world_pos[i]
                mp = mesh_pos[i]
                twp = target_world_pos[i]
                nt = node_type[i]
                c = cells[i]
                pwp = prev_world_pos[i]
                trajectory_step = {'world_pos': wp, 'mesh_pos': mp, 'node_type': nt, 'cells': c,
                                   'target|world_pos': twp, 'prev|world_pos': pwp}
                noisy_trajectory_step = add_noise(trajectory_step)
                trajectory_steps.append(noisy_trajectory_step)
            return trajectory_steps
        return element_operation


if __name__ == '__main__':
    node_info = {'NORMAL': 0 }
    dataset = FlagSimpleDataset(device='cpu', path='../data/flag_simple', history = True , 
                                split='train', node_info=node_info, augmentation = True)


