#!/bin/bash

python -m modules.mgn_diffusion --epochs 400 --train_batch_size 8 --val_batch_size 8 --learning_rate 1e-4 \
--trajectory_length 20 --data_dir './data/' --exp_name 'diffusion_with_mgn_multiple_layers'
