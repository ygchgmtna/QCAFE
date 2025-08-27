from run.run import run
import torch
import numpy as np
import os

model_name = 'qattention' # Choose from "cafe". "qattention"
dataset_name = "twitter"  # Choose from "politifact", "gossipcop", "twitter", "weibo"

args_dict = {
    'dataset_dir': '../data',
    'dataset_name': dataset_name, 
    'mode': 'train',  # Mode can be 'train' or 'test'
    # 'pretrained_path': f"pth/qaf4_{dataset_name}_epoch1.pth",  # Path to the pretrained model
    # 'pretrained_path':"pth/qaf4b_1_weibo_3e_4_epoch1.pth",
    'batch_size': 256,
    'lr': 6e-4, 
    'epoch_num' :2,
    # 'best_path': f"zxz2+crx+q_{dataset_name}",  # Path to save the best model
    'best_path': f"qafv2_{dataset_name}_6e_4_epoch2_3"
}

run(model_name, **args_dict)