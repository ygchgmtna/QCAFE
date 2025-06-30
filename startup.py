from run.run import run
import torch
import numpy as np
import os

model_name = 'cafe'
args_dict = {
    'dataset_dir': './dataset',
    "dataset_name": "twitter", # Choice of "politifact" "gossipcop" "twitter"
    'batch_size': 64,
    'lr': 1e-4, 
    'epoch_num' : 30
}

run(model_name, **args_dict)