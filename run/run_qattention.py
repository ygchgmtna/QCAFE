# run_qattention.py
import torch
from torch.utils.data import DataLoader
from typing import List
import yaml

from utils.util import dict2str
from train.qattention_trainer import QAttentionTrainer
from evaluate.evaluator import Evaluator
from preprocess.dataset.cafe_dataset import CafeDataset
from model.qattention import QAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_qattention(dataset_dir: str,
                   dataset_name: str = "politifact",
                   mode: str = "train",
                   batch_size=64,
                   lr=1e-3,
                   weight_decay=0,
                   epoch_num=2,
                   metrics: List = None,
                   pretrained_path=None,
                   best_path: str = None,
                   device=device):

    # --- Load Data ---
    train_set = CafeDataset(f"{dataset_dir}/{dataset_name}/train_text_with_label.npz",
                            f"{dataset_dir}/{dataset_name}/train_image_with_label.npz")
    test_set = CafeDataset(f"{dataset_dir}/{dataset_name}/test_text_with_label.npz",
                           f"{dataset_dir}/{dataset_name}/test_image_with_label.npz")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # --- Build Model ---
    model = QAttention().to(device)

    # --- Optimizer ---
    optim_task_detection = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_task_detection,
                                                  lr_lambda=lambda epoch: 1.0 if epoch < 1 else 1e-5 / lr)

    # --- Trainer ---
    evaluator = Evaluator(metrics)
    trainer = QAttentionTrainer(model,
                            evaluator,
                            optim_task_detection,
                            scheduler=scheduler,
                            best_path=best_path,
                            device=device)


    # --- Train/Test ---
    if mode == "train":
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            trainer.logger.info(f"Loaded pretrained model from {pretrained_path}")

        trainer.fit(train_loader, epoch_num)

        if test_loader:
            result = trainer.evaluate(test_loader)
            trainer._show_data_size(train_loader, test_loader=test_loader)
            trainer.logger.info(f"test result: {dict2str(result)}")

    elif mode == "test":
        if not pretrained_path:
            raise ValueError("pretrained_path must be provided for testing mode.")
        result = trainer.evaluate(test_loader, pretrained_path=pretrained_path)
        trainer._show_data_size(train_loader, test_loader=test_loader)
        trainer.logger.info(f"test result: {dict2str(result)}")


def run_qattention_from_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_qattention(**config)
