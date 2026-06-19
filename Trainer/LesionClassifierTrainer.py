"""
病灶多标签分类训练器
支持：
1. 多标签分类训练循环
2. 早停机制
3. 学习率调度
4. 混合精度训练
5. 实时指标监控
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from typing import Dict, List, Optional

# 绝对导入
from Conf.Config import *
# 导入新增的 Logger（若尚未创建则先创建文件）
try:
    from Utils.TrainingLogger import TrainingLogger
except ImportError:
    # 若尚未创建，跳过，但运行时需要
    pass


class LesionClassifierTrainer:
    """病灶分类模型训练器"""

    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: str = DEVICE,
            epochs: int = EPOCHS,
            save_path: str = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            patience: int = EARLY_STOP_PATIENCE,
            grad_clip: float = GRAD_CLIP,
            use_amp: bool = USE_AMP,
            show_realtime_viz: bool = True,
            logger: Optional[object] = None   # <--- 新增 logger 参数
    ):
        """
        Args:
            model: 分类模型
            loss_fn: 损失函数
            optimizer: 优化器
            device: 计算设备
            epochs: 训练轮数
            save_path: 模型保存路径
            scheduler: 学习率调度器
            patience: 早停耐心值
            grad_clip: 梯度裁剪阈值
            use_amp: 是否使用混合精度
            show_realtime_viz: 是否显示实时可视化
            logger: TrainingLogger 实例，用于记录训练日志
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.save_path = save_path or os.path.join(WEIGHT_SAVE_DIR, "best_lesion.pth")
        self.scheduler = scheduler
        self.patience = patience
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.show_realtime_viz = show_realtime_viz
        self.logger = logger  # <--- 保存 logger

        # 混合精度 scaler（处理 CUDA 不可用的情况）
        if use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # 最佳指标
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        print(f"\n{'=' * 60}")
        print(f"病灶分类训练器初始化完成")
        print(f"{'=' * 60}")
        print(f"设备: {device}")
        print(f"训练轮数: {epochs}")
        print(f"早停耐心值: {patience}")
        print(f"混合精度: {use_amp}")
        print(f"{'=' * 60}")

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        计算多标签分类指标
        """
        preds_binary = (predictions >= POSITIVE_THRESHOLD).astype(int)
        num_classes = predictions.shape[1]
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(num_classes):
            tp = np.sum((preds_binary[:, i] == 1) & (targets[:, i] == 1))
            fp = np.sum((preds_binary[:, i] == 1) & (targets[:, i] == 0))
            fn = np.sum((preds_binary[:, i] == 0) & (targets[:, i] == 1))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        sample_tp = np.sum((preds_binary == 1) & (targets == 1))
        sample_fp = np.sum((preds_binary == 1) & (targets == 0))
        sample_fn = np.sum((preds_binary == 0) & (targets == 1))
        sample_precision = sample_tp / (sample_tp + sample_fp + 1e-8)
        sample_recall = sample_tp / (sample_tp + sample_fn + 1e-8)
        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall + 1e-8)

        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'sample_precision': sample_precision,
            'sample_recall': sample_recall,
            'sample_f1': sample_f1,
            'per_class_f1': f1_scores
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp and torch.cuda.is_available() and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_predictions = []
        all_targets = []

        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for batch in progress_bar:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(probs)
            all_targets.append(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = self.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss

        return metrics

    def Train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        完整训练流程
        """
        print(f"\n🚀 开始训练...")
        print(f"{'=' * 60}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)

            # 打印进度
            print(f"\nEpoch [{epoch}/{self.epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Macro F1:   {val_metrics['macro_f1']:.4f}")
            print(f"  Sample F1:  {val_metrics['sample_f1']:.4f}")
            print(f"  LR:         {current_lr:.6f}")

            # ========== 新增：记录日志 ==========
            if self.logger is not None:
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    macro_f1=val_metrics['macro_f1'],
                    sample_f1=val_metrics['sample_f1'],
                    lr=current_lr,
                    best_sample_f1=self.best_f1
                )

            # 检查是否为最佳模型
            current_f1 = val_metrics['sample_f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_model(epoch, val_metrics)
                print(f"  ✅ 新的最佳模型! F1: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏸️  耐心计数: {self.patience_counter}/{self.patience}")

            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  早停触发! 最佳 Epoch: {self.best_epoch}, F1: {self.best_f1:.4f}")
                break

            print(f"{'=' * 60}")

        total_time = time.time() - start_time
        total_minutes = total_time / 60

        # ========== 新增：保存最终总结 ==========
        if self.logger is not None:
            self.logger.save_final_summary(self.best_epoch, self.best_f1, total_minutes)

        print(f"\n🎉 训练完成!")
        print(f"  总耗时: {total_minutes:.2f} 分钟")
        print(f"  最佳 Epoch: {self.best_epoch}")
        print(f"  最佳 F1: {self.best_f1:.4f}")

        return self.history

    def save_model(self, epoch: int, metrics: Dict):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_f1': self.best_f1
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(checkpoint, self.save_path)
        print(f"  💾 模型已保存: {self.save_path}")

    def load_model(self, path: str = None):
        """加载模型"""
        path = path or self.save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        self.best_epoch = checkpoint.get('epoch', 0)
        print(f"✅ 模型已加载: {path}")
        print(f"   Best F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")