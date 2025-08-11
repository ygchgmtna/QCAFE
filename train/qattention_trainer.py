# train/qattention_trainer.py
from train.trainer import BaseTrainer
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import sys, os

from utils.util import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QAttentionTrainer(BaseTrainer):
    """
    Trainer for QAttention model
    """
    def __init__(self,
                 model,
                 evaluator,
                 detection_optimizer,
                 scheduler=None,
                 clip_grad_norm=None,
                 device=device,
                 early_stopping=None,
                 best_path=None):
        super().__init__(model, evaluator, detection_optimizer, scheduler,
                         clip_grad_norm, device, early_stopping)
        self.best_f1 = 0.0
        if best_path is not None:
            self.base_path = os.path.join("pth", best_path)
        else:
            self.base_path = os.path.join("pth", "best_model")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_acc, total_f1, count = 0.0, 0.0, 0

        with tqdm(enumerate(loader),
                  total=len(loader),
                  ncols=100,
                  desc=f"Epoch {epoch+1}",
                  dynamic_ncols=True,
                  leave=False,
                  file=sys.stdout) as pbar:

            for batch_id, batch_data in pbar:
                batch_data = self._move_data_to_device(batch_data)

                # 仅计算 detection loss
                detection_loss = self.model.calculate_loss(batch_data)
                self.optimizer.zero_grad()
                detection_loss.backward()
                self.optimizer.step()

                # 计算指标
                with torch.no_grad():
                    logits = self.model.forward(batch_data['text'], batch_data['image'])
                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    labels = batch_data['label'].detach().cpu().numpy()
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average='macro')

                total_acc += acc
                total_f1 += f1
                count += 1

                pbar.set_postfix_str(f"det={detection_loss.item():.4f}, acc={acc:.4f}, f1={f1:.4f}")

                # 保存 best
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    if f1 > 0.8:
                        final_path = f"{self.base_path}_epoch{epoch+1}_batch{batch_id+1}.pth"
                        torch.save(self.model.state_dict(), final_path)
                        self.logger.info(f"Saved best model to {final_path} with F1: {f1:.4f}")
                        self.best_path = final_path

            avg_acc = total_acc / count
            avg_f1 = total_f1 / count
            self.logger.info(f"[Epoch {epoch+1}] Avg Train Acc: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")

        return {"detection_loss": detection_loss.item()}
