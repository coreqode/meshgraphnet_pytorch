#!/bin/bash

python -m modules.mgn --epochs 400 --train_batch_size 8 --val_batch_size 8 --learning_rate 1e-4 \
--trajectory_length 120 --data_dir './data/' --exp_name 'trajectory_length_120'
