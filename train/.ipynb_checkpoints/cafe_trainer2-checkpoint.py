from train.trainer import BaseTrainer
from typing import Optional, Dict, Any
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys, os

from evaluate.evaluator import Evaluator
from model.cafe import CAFE
from utils.util import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CafeTrainer(BaseTrainer):
    """
    Trainer for CAFE model with tow model,
    which inherits from BaseTrainer and modifies the '_train_epoch' method.
    """
    def __init__(self,
                 model: CAFE,
                 evaluator: Evaluator,
                 detection_optimizer: Optimizer,
                 similarity_optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 clip_grad_norm: Optional[Dict[str, Any]] = None,
                 device=device,
                 early_stopping: Optional[EarlyStopping] = None,
                 best_path: Optional[str] = None):
        """
        Args:
            model (CAFE): the first faknow abstract model to train
            evaluator (Evaluator):  faknow evaluator for evaluation
            detection_optimizer (Optimizer): pytorch optimizer for training of the detection model
            similarity_optimizer (Optimizer): pytorch optimizer for training of the similarity model
            scheduler (_LRScheduler): learning rate scheduler. Defaults=None.
            clip_grad_norm (Dict[str, Any]): key args for
                torch.nn.utils.clip_grad_norm_. Defaults=None.
            device (str): device to use. Defaults='cuda:0'.
            early_stopping (EarlyStopping): early stopping for training.
                If None, no early stopping will be performed. Defaults=None.
        """

        super().__init__(model, evaluator, detection_optimizer, scheduler,
                         clip_grad_norm, device, early_stopping)
        self.similarity_optimizer = similarity_optimizer
        self.best_f1 = 0.0
        self.best_batch = -1
        if best_path is not None:
            self.base_path = os.path.join("pth", best_path)
        else:
            self.base_path = os.path.join("pth", "best_model")

    def _train_epoch(self, loader: DataLoader,
                     epoch: int) -> Dict[str, float]:
        """
         training for one epoch, including gradient clipping

         Args:
             loader (DataLoader): training data
             epoch (int): current epoch

         Returns:
             Union[float, Dict[str, float]]: loss of current epoch.
                 If multiple losses,
                 return a dict of losses with loss name as key.
         """
        self.model.train()

        total_acc, total_f1, count = 0.0, 0.0, 0
        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc=f"Epoch {epoch+1}",
                  dynamic_ncols=True,
                  leave=False,
                  file=sys.stdout) as pbar:
            loss = None
            for batch_id, batch_data in pbar:
                batch_data = self._move_data_to_device(batch_data)

                similarity_loss = self.model.similarity_module.calculate_loss(batch_data)
                self.similarity_optimizer.zero_grad()
                similarity_loss.backward()
                self.similarity_optimizer.step()

                detection_loss = self.model.calculate_loss(batch_data)
                self.optimizer.zero_grad()
                detection_loss.backward()
                self.optimizer.step()

                loss = {
                    'similarity_loss': similarity_loss.item(),
                    'detection_loss': detection_loss.item()
                }
                with torch.no_grad():
                    logits = self.model.forward(batch_data['text'], batch_data['image'])
                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    labels = batch_data['label'].detach().cpu().numpy()
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average='macro')

                total_acc += acc
                total_f1 += f1
                count += 1

                pbar.set_postfix_str(
                    f"sim={similarity_loss.item():.4f}, det={detection_loss.item():.4f}, acc={acc:.4f}, f1={f1:.4f}"
                )

                loss = {
                    'similarity_loss': similarity_loss.item(),
                    'detection_loss': detection_loss.item()
                }

                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_batch = batch_id + 1

                    if f1 > 0.8:  # 仍保留你设定的下限
                        final_path = f"{self.base_path}_epoch{epoch+1}_batch{self.best_batch}.pth"
                        torch.save(self.model.state_dict(), final_path)
                        self.logger.info(f"Saved best model to {final_path} with batch-F1: {f1:.4f}")
                        self.best_path = final_path

            avg_acc = total_acc / count
            avg_f1 = total_f1 / count

            self.logger.info(f"[Epoch {epoch+1}] Avg Train Acc: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")

            # 保存模型
            if not os.path.isdir("pth"):
                os.makedirs("pth")

            if avg_f1 > self.best_f1:
                self.best_f1 = avg_f1
                self.best_epoch = epoch + 1
                
                if avg_f1 > 0.8:
                    final_path =f"{self.base_path}_epoch{self.best_epoch}.pth"
                    torch.save(self.model.state_dict(), final_path)
                    self.logger.info(f"Saved best model to {final_path} with F1: {avg_f1:.4f}")
                    self.best_path = final_path

        return loss