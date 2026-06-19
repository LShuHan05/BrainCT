import torch
import torch.nn.functional as F
import tqdm
import os
import numpy as np

# ⚠️ 【修改】改为绝对导入
from Utils.EvalMetrics import MetricCalculator
from Utils.TrainingVisualizer import RealTimeVisualizer
from Conf.Config import (
    USE_ADAPTIVE_THRESHOLD, THRESHOLD_SEARCH_STEP, THRESHOLD_SEARCH_RANGE,
    USE_OHEM, OHEM_RATIO,
    USE_WARM_RESTARTS, RESTART_T_0, RESTART_T_MULT
)


class Trainer:
    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            device,
            save_path,
            epochs,
            scheduler=None,
            patience=10,
            grad_clip=1.0,
            use_amp=False,
            show_realtime_viz=True
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.epochs = epochs
        self.save_path = save_path
        self.patience = patience
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        self.show_realtime_viz = show_realtime_viz
        if self.show_realtime_viz:
            viz_dir = os.path.join(os.path.dirname(save_path), '..', 'viz')
            self.visualizer = RealTimeVisualizer(save_dir=viz_dir)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "dice": [],
            "iou": []
        }
        self.best_dice = 0.0
        self.early_stop_counter = 0
        self.optimal_threshold = 0.5

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        bar = tqdm.tqdm(loader, desc="Training")

        for img, gt in bar:
            img = img.to(self.device)
            gt = gt.to(self.device)

            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(img)
                    if isinstance(pred, dict):
                        loss = self.loss_fn(pred, gt)
                    else:
                        loss = self.loss_fn(pred, gt)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(img)
                if isinstance(pred, dict):
                    pred_logits = pred['main']
                else:
                    pred_logits = pred

                if isinstance(pred, dict):
                    loss_raw = self.loss_fn(pred, gt)
                else:
                    loss_raw = self.loss_fn(pred, gt)

                if USE_OHEM:
                    pixel_loss = F.binary_cross_entropy_with_logits(pred_logits, gt, reduction='none')
                    pixel_loss_flat = pixel_loss.view(-1)
                    num_keep = int(OHEM_RATIO * pixel_loss_flat.numel())
                    _, indices = torch.topk(pixel_loss_flat, num_keep)
                    ohem_mask = torch.zeros_like(pixel_loss_flat).scatter_(0, indices, 1.0)
                    ohem_mask = ohem_mask.view(pixel_loss.shape)
                    loss = (pixel_loss * ohem_mask).sum() / (ohem_mask.sum() + 1e-7)
                else:
                    loss = loss_raw

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() if not isinstance(loss, dict) else loss['main'].item()
            bar.set_postfix(loss=loss.item() if not isinstance(loss, dict) else loss['main'].item())

        return total_loss / len(loader)

    @torch.no_grad()
    def search_best_threshold(self, loader):
        print("\n🔍 正在搜索最优分割阈值...")
        all_preds = []
        all_targets = []
        for img, gt in tqdm.tqdm(loader, desc="Collecting predictions"):
            img = img.to(self.device)
            pred = self.model(img)
            if isinstance(pred, dict):
                pred = pred['main']
            pred = torch.sigmoid(pred).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(gt.cpu().numpy())
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_targets = np.concatenate([t.flatten() for t in all_targets])

        best_dice = 0.0
        best_th = 0.5
        low, high, step = THRESHOLD_SEARCH_RANGE[0], THRESHOLD_SEARCH_RANGE[1], THRESHOLD_SEARCH_STEP
        for th in np.arange(low, high + step, step):
            pred_bin = (all_preds > th).astype(np.float32)
            intersection = (pred_bin * all_targets).sum()
            dice = (2. * intersection + 1e-7) / (pred_bin.sum() + all_targets.sum() + 1e-7)
            if dice > best_dice:
                best_dice = dice
                best_th = th
        print(f"✅ 最优阈值: {best_th:.3f} (Dice={best_dice:.4f})")
        return best_th

    @torch.no_grad()
    def val_one_epoch(self, loader, optimal_threshold=None):
        self.model.eval()
        total_loss = 0
        total_pre = 0
        total_rec = 0
        total_dice = 0
        total_acc = 0
        total_iou = 0

        if optimal_threshold is None:
            optimal_threshold = self.optimal_threshold

        val_bar = tqdm.tqdm(loader, desc="Validating")
        for img, gt in val_bar:
            img = img.to(self.device)
            gt = gt.to(self.device)
            pred = self.model(img)
            if isinstance(pred, dict):
                pred = pred['main']
            loss = self.loss_fn(pred, gt) if not isinstance(pred, dict) else self.loss_fn({'main': pred}, gt)
            total_loss += loss.item()
            val_bar.set_postfix(val_loss=loss.item())
            metrics = MetricCalculator.calculate_metrics_with_threshold(pred, gt, optimal_threshold)
            p, r, d, a, iou = metrics
            total_pre += p
            total_rec += r
            total_dice += d
            total_acc += a
            total_iou += iou

        avg_loss = total_loss / len(loader)
        avg_pre = total_pre / len(loader)
        avg_rec = total_rec / len(loader)
        avg_dice = total_dice / len(loader)
        avg_acc = total_acc / len(loader)
        avg_iou = total_iou / len(loader)
        return avg_loss, avg_pre, avg_rec, avg_dice, avg_acc, avg_iou

    def Train(self, train_loader, val_loader):
        print(f"🚀 开始训练，设备：{self.device}")
        print(f"最优模型将保存至：{self.save_path}\n")
        if self.patience > 0:
            print(f"⏱️  早停机制已启用，耐心值：{self.patience} epochs\n")
        if USE_ADAPTIVE_THRESHOLD:
            print("🔧 自适应阈值已启用，每个 epoch 将重新搜索最优阈值\n")
        if USE_OHEM:
            print(f"⚡ 在线难例挖掘（OHEM）已启用，比例: {OHEM_RATIO}\n")

        if USE_WARM_RESTARTS and not isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=RESTART_T_0, T_mult=RESTART_T_MULT
            )
            print(f"🔄 使用 CosineAnnealingWarmRestarts 学习率调度器 (T_0={RESTART_T_0}, T_mult={RESTART_T_MULT})\n")

        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"======== Epoch {epoch+1}/{self.epochs} ========")
            print(f"{'='*60}\n")

            train_loss = self.train_one_epoch(train_loader)

            if USE_ADAPTIVE_THRESHOLD:
                self.optimal_threshold = self.search_best_threshold(val_loader)

            val_loss, pre, rec, dice, acc, iou = self.val_one_epoch(val_loader, self.optimal_threshold)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["acc"].append(acc)
            self.history["precision"].append(pre)
            self.history["recall"].append(rec)
            self.history["dice"].append(dice)
            self.history["iou"].append(iou)

            if self.show_realtime_viz:
                current_epoch = epoch + 1
                metrics_dict = {'acc': acc, 'precision': pre, 'recall': rec, 'dice': dice, 'iou': iou}
                self.visualizer.print_epoch_summary(
                    epoch=current_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics_dict,
                    best_dice=self.best_dice
                )
                if current_epoch % 3 == 0 or current_epoch == self.epochs:
                    try:
                        self.visualizer.plot_metrics(self.history, current_epoch)
                    except Exception as e:
                        print(f"⚠️  图表更新失败: {e}")

            print(f"\n📈 当前轮次指标 (阈值={self.optimal_threshold:.3f}):")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"   Acc: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f}")
            print(f"   Dice: {dice:.4f} | IoU: {iou:.4f}")

            if dice > self.best_dice:
                self.best_dice = dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'optimal_threshold': self.optimal_threshold,
                }, self.save_path)
                print(f"\n✅ 已保存最优模型！Dice: {dice:.4f} (阈值={self.optimal_threshold:.3f})\n")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"\n⚠️  未提升，连续 {self.early_stop_counter} 个epoch无改善")
                if self.patience > 0 and self.early_stop_counter >= self.patience:
                    print(f"\n🛑 触发早停！验证Dice在{self.patience}个epoch内未提升")
                    break

        print(f"\n{'='*60}")
        print(f"🎯 训练结束！最佳验证Dice: {self.best_dice:.4f} (对应阈值={self.optimal_threshold:.3f})")
        print(f"{'='*60}")

        if self.show_realtime_viz:
            self.visualizer.plot_metrics(self.history, self.epochs,
                                         os.path.join(self.visualizer.save_dir, 'final_report.png'))
        return self.history
