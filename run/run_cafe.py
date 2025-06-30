from typing import List

import yaml
import torch
from torch.utils.data import DataLoader

from utils.util import dict2str
from train.cafe_trainer import CafeTrainer
from evaluate.evaluator import Evaluator
from preprocess.dataset.cafe_dataset import CafeDataset
from model.cafe import CAFE

__all__ = ['run_cafe', 'run_cafe_from_yaml']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def run_cafe(dataset_dir: str,
             dataset_name: str = "politifact",
             batch_size=64,
             lr=1e-3,
             weight_decay=0,
             epoch_num=2,
             metrics: List = None,
             device=device):
    """
    run CAFE
    Args:
        dataset_dir (str): path of data,including training data and testing data.
        batch_size (int): batch size, default=64
        lr (float): learning rate, default=0.001
        weight_decay (float): weight_decay, default=0
        epoch_num(int): number of epochs, default=50
        metrics (List): evaluation metrics,
            if None, ['accuracy', 'precision', 'recall', 'f1'] is used,
            default=None
        device (str): device to run model, default='cuda:0'
    """
    # ---  Load Data  ---
    train_set = CafeDataset(
        "{}/{}/train_text_with_label.npz".format(dataset_dir, dataset_name),
        "{}/{}/train_image_with_label.npz".format(dataset_dir, dataset_name))
    test_set = CafeDataset(
        "{}/{}/test_text_with_label.npz".format(dataset_dir, dataset_name),
        "{}/{}/test_image_with_label.npz".format(dataset_dir, dataset_name)) 
                                   
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)
    # print(f'\ntest_loader: {len(test_loader)}\n')

    model = CAFE().to(device)

    optim_task_similarity = torch.optim.Adam(
        model.similarity_module.parameters(), lr=lr, weight_decay=weight_decay)

    sim_params_id = list(map(id, model.similarity_module.parameters()))
    base_params = filter(lambda p: id(p) not in sim_params_id,
                         model.parameters())
    optim_task_detection = torch.optim.Adam(base_params,
                                            lr=lr,
                                            weight_decay=weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_task_detection, T_0=1, T_mult=1)

    evaluator = Evaluator(metrics)

    trainer = CafeTrainer(model,
                          evaluator,
                          optim_task_detection,
                          optim_task_similarity,
                        #   scheduler=scheduler,
                          device=device)
    
    # 训练入口
    trainer.fit(train_loader, epoch_num)

    if test_loader is not None:
        test_result = trainer.evaluate(test_loader)
        trainer._show_data_size(train_loader, test_loader=test_loader)
        trainer.logger.info(f"test result: {dict2str(test_result)}")


def run_cafe_from_yaml(path: str) -> None:
    """
    run EANN from yaml config file

    Args:
        path (str): yaml config file path
    """

    with open(path, 'r', encoding='utf-8') as _f:
        _config = yaml.load(_f, Loader=yaml.FullLoader)
        run_cafe(**_config)
