from run.run import run
import torch
import numpy as np
import os

model_name = 'cafe'
args_dict = {
    'dataset_dir': './dataset',
    "dataset_name": "twitter", # Choice of "politifact" "gossipcop" "twitter"
    'mode': 'train',  # Mode can be 'train' or 'test'
    # 'pretrained_path': 'pth/zxz_twitter_epoch2.pth',  # Path to the pretrained model
    'batch_size': 64,
    'lr': 1e-3, 
    'epoch_num' : 3
}

run(model_name, **args_dict)