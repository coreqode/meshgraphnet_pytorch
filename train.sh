#!/bin/bash

python -m modules.mgn --epochs 100 --train_batch_size 8 --val_batch_size 8 --learning_rate 1e-4 \
--trajectory_length 20 --data_dir '/scratch/sidd/data/' --exp_name 'simple_test_run' \
--if_sampling "True" --sample_n_points 200
