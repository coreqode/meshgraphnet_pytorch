import argparse

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--trajectory_length', default=20, type=int)
    parser.add_argument('--split_ratio', default=0.85, type=float)
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--exp_name', default='simple_test_run', type=str)
    opts = parser.parse_args()
    return opts