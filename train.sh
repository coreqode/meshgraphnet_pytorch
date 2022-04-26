#!/bin/bash

python -m modules.mgn_diffusion --epochs 100 --train_batch_size 4 --val_batch_size 4 --learning_rate 1e-4 \
--trajectory_length 20 --data_dir './data/' --exp_name 'simple_test_run'
