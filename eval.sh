#!/bin/bash

python -m modules.mgn --trajectory_length 150 --data_dir '/scratch/sidd/data/' --if_sampling "True" --sample_n_points 200 --train_batch_size 1 --val_batch_size 1 --mode "rollout"
