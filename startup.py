from run.run import run
import torch
import numpy as np
import os

model_name = 'cafe'
dataset_name = "weibo"  # Choose from "politifact", "gossipcop", "twitter", "weibo"

args_dict = {
    'dataset_dir': './dataset',
    'dataset_name': dataset_name, 
    'mode': 'test',  # Mode can be 'train' or 'test'
    'pretrained_path': f"pth/qaf4_{dataset_name}_epoch3_batch71.pth",  # Path to the pretrained model
    # 'pretrained_path':"pth/best_model_twitter_88_bs=64.pth",
    'batch_size': 64,
    'lr': 1e-3, 
    'epoch_num' :5,
    # 'best_path': f"zxz2+crx+q_{dataset_name}",  # Path to save the best model
    'best_path': f"qaf4_{dataset_name}"
}

run(model_name, **args_dict)