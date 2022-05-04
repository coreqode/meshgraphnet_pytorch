#!/bin/bash

#python -m modules.mgn --train_batch_size 1 --val_batch_size 1 --trajectory_length 350 --data_dir './data/'
python -m modules.mgn_diffusion --train_batch_size 1 --val_batch_size 1 --trajectory_length 350 --data_dir './data/'


