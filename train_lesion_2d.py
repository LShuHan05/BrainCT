# train_lesion_2d.py
"""
2D 切片 + ResNet50 预训练 多标签病灶分类（改进版）
- 增加 FC 层 Dropout(0.5)
- 权重衰减 1e-4 → 5e-4
- 使用 ReduceLROnPlateau 替代 CosineAnnealing
- 增加对比度/亮度数据增强
- 增加冻结骨干网络预热策略（前 5 个 epoch 只训练 FC 层）
- EDH 权重 10 → 15
- 早停 patience 从 12 降低到 10（更早停止，减少过拟合）
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from Conf.Config import *
import torchvision.models as models

# ---------- 固定随机种子 ----------
torch.manual_seed(42)
np.random.seed(42)

# ---------- 2D 数据集 ----------
class NiftiSliceDataset(Dataset):
    """从 NIfTI 体积中随机抽取轴向切片，返回单张 2D 图像 + 9 类标签"""
    def __init__(self, metadata_csv, nifti_dir, target_size=(256,256), 
                 num_slices=NUM_SLICES_PER_VOLUME, transform=True):
        self.df = pd.read_csv(metadata_csv)
        self.nifti_dir = nifti_dir
        self.target_size = target_size
        self.num_slices = num_slices
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['case_id']
        nifti_path = os.path.join(self.nifti_dir, f"{case_id}.nii.gz")
        img = nib.load(nifti_path)
        volume = img.get_fdata().astype(np.float32)
        # 归一化
        volume = np.clip(volume, -100, 100)
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)

        # 随机抽取轴向切片（axis=0）
        D, H, W = volume.shape
        if self.transform:
            # 训练时随机抽取 num_slices 张
            slice_indices = np.random.choice(D, self.num_slices, replace=False)
        else:
            # 验证/测试时取中间切片或均匀采样
            slice_indices = np.linspace(0, D-1, self.num_slices, dtype=int)

        # 收集切片并堆叠为 (num_slices, H, W)
        slices = []
        for si in slice_indices:
            slice_img = volume[si, :, :]
            # resize 到 target_size
            h, w = slice_img.shape
            if (h, w) != self.target_size:
                zoom_factors = (self.target_size[0]/h, self.target_size[1]/w)
                slice_img = zoom(slice_img, zoom_factors, order=1)
            slices.append(slice_img)

        # 转换为 (num_slices, 1, H, W)
        slices = np.stack(slices, axis=0)  # (num_slices, H, W)
        slices = slices[:, np.newaxis, :, :]  # 加通道维

        # 数据增强（翻转/旋转/对比度）
        if self.transform:
            for i in range(slices.shape[0]):
                if np.random.rand() > 0.5:
                    slices[i] = np.fliplr(slices[i]).copy()
                if np.random.rand() > 0.5:
                    slices[i] = np.flipud(slices[i]).copy()
                # 小旋转
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    from scipy.ndimage import rotate
                    slices[i] = rotate(slices[i].squeeze(), angle, reshape=False, order=1)
                    slices[i] = slices[i][np.newaxis, ...]
                # 【新增】对比度/亮度调整
                if np.random.rand() > 0.5:
                    contrast = np.random.uniform(0.8, 1.2)
                    brightness = np.random.uniform(-0.1, 0.1)
                    slices[i] = slices[i] * contrast + brightness
                    slices[i] = np.clip(slices[i], 0, 1)

        # 转 tensor
        slices_tensor = torch.from_numpy(slices).float()  # (num_slices, 1, H, W)

        # 标签（9类）
        labels = [row[f'label_{i}'] for i in range(9)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return slices_tensor, labels_tensor

# ---------- 数据加载器 collate 函数 ----------
def collate_slices(batch):
    """将多个样本的切片拼接为一个大 batch，标签重复"""
    all_slices = []
    all_labels = []
    for slices, labels in batch:
        all_slices.append(slices)
        all_labels.extend([labels] * slices.shape[0])
    all_slices = torch.cat(all_slices, dim=0)  # (total_slices, 1, H, W)
    all_labels = torch.stack(all_labels, dim=0)  # (total_slices, 9)
    return all_slices, all_labels

# ---------- 模型定义（ResNet50 预训练 + Dropout） ----------
class ResNet50_2D(nn.Module):
    def __init__(self, num_classes=9, input_channels=1, pretrained=True, dropout_rate=0.5):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # 修改第一层接受单通道
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        # 修改最后一层：增加 Dropout 防止过拟合
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ---------- Asymmetric Loss（加权 Focal Loss） ----------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            weight = self.pos_weight.to(logits.device) * targets + (1 - targets)
        else:
            weight = 1.0
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight_neg = (1 - p_t) ** self.gamma_neg
        weight_pos = (1 - p_t) ** self.gamma_pos
        weights = targets * weight_pos + (1 - targets) * weight_neg
        weights = torch.clamp(weights, min=self.clip)
        loss = weights * bce * weight
        return loss.mean()

# ---------- 评估指标 ----------
def compute_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds >= threshold).astype(int)
    tp = (preds_bin * targets).sum(axis=0)
    fp = ((preds_bin) * (1 - targets)).sum(axis=0)
    fn = ((1 - preds_bin) * targets).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.mean(), recall.mean(), f1.mean()

# ---------- 单折训练 ----------
def train_fold(fold_idx, train_indices, val_indices, dataset, fold_log_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/5")
    print(f"  训练: {len(train_indices)}, 验证: {len(val_indices)}")
    print(f"{'='*50}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        collate_fn=collate_slices,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        collate_fn=collate_slices,
        num_workers=NUM_WORKERS//2, pin_memory=PIN_MEMORY,
        prefetch_factor=2, persistent_workers=True
    )

    model = ResNet50_2D(num_classes=NUM_CLASSES, input_channels=1, 
                        pretrained=True, dropout_rate=0.5).to(DEVICE)
    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 类别权重（EDH 权重 15）
    pos_weight = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, pos_weight=pos_weight)

    # 优化器：权重衰减从 1e-4 提升到 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # 【新增】ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    scaler = GradScaler() if USE_AMP else None

    history = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [], 'train_f1': [],
        'val_precision': [], 'val_recall': []
    }
    best_f1 = 0.0
    patience_counter = 0
    # 【新增】预热策略：前 5 个 epoch 只训练 FC 层（冻结 backbone）
    freeze_epochs = 5

    for epoch in range(1, EPOCHS_3D + 1):
        # ----- 冻结/解冻控制 -----
        if epoch <= freeze_epochs:
            # 冻结 backbone（除 FC 外所有层）
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print(f"  🔒 预热阶段: 仅训练 FC 层 (Epoch {epoch}/{freeze_epochs})")
        else:
            # 解冻所有层
            for param in model.parameters():
                param.requires_grad = True

        # ---------- 训练 ----------
        model.train()
        train_loss = 0
        all_train_preds, all_train_targets = [], []
        for x, y in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch}", leave=False):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            if USE_AMP:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():
                all_train_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_train_targets.append(y.cpu().numpy())
        train_loss /= len(train_loader)
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)
        train_prec, train_rec, train_f1 = compute_metrics(all_train_preds, all_train_targets)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                if USE_AMP:
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                all_val_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_val_targets.append(y.cpu().numpy())
        val_loss /= len(val_loader)
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)
        val_prec, val_rec, val_f1 = compute_metrics(all_val_preds, all_val_targets)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['train_f1'].append(train_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val F1={val_f1:.4f}, LR={current_lr:.6f}")

        # 【修改】ReduceLROnPlateau 需要传入 val_loss
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_log_dir, f"best_fold_{fold_idx+1}.pth"))
            print(f"  ✅ 新最佳 F1={best_f1:.4f}")
        else:
            patience_counter += 1
            # 【修改】patience 从 12 降低到 10
            if patience_counter >= 10:
                print(f"  ⏳ 早停触发 (patience=10)")
                break

        if epoch % 10 == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.subplot(1, 2, 2)
            plt.plot(history['val_f1'], label='Val F1')
            plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best={best_f1:.3f}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(fold_log_dir, f"fold_{fold_idx+1}_epoch{epoch}.png"), dpi=150)
            plt.close()

    print(f"  Fold {fold_idx+1} 最佳 F1={best_f1:.4f}")
    return best_f1, history

# ---------- 日志重定向 ----------
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# ---------- 主函数 ----------
def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    log_dir = Path("/mnt/workspace/BrainCT/logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_2d_{timestamp}.log"
    sys.stdout = Tee(str(log_file))

    print(f"📝 日志将保存到: {log_file}")
    print(f"🚀 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("🚀 2D 切片 + ResNet50 预训练 5折交叉验证（改进版）")
    print("   - FC Dropout=0.5")
    print("   - Weight Decay=5e-4")
    print("   - ReduceLROnPlateau (patience=5, factor=0.5)")
    print("   - 对比度/亮度增强")
    print("   - 预热策略: 前5个epoch仅训练FC层")
    print("   - EDH权重: 15")
    print("   - Early Stop Patience=10")
    print(f"📍 设备: {DEVICE}, 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"📊 模型: ResNet50 (ImageNet预训练)")
    print(f"📈 学习率: 1e-4, Weight Decay: 5e-4")
    print(f"📦 Batch Size: {BATCH_SIZE_TRAIN}, 每例切片数: {NUM_SLICES_PER_VOLUME}")
    print("=" * 70)

    dataset = NiftiSliceDataset(METADATA_CSV, NIFTI_DIR, target_size=TARGET_SIZE_2D,
                                num_slices=NUM_SLICES_PER_VOLUME, transform=True)
    print(f"📚 数据集: {len(dataset)} 例 (每例 {NUM_SLICES_PER_VOLUME} 张切片)")

    # 分层标签
    labels_df = dataset.df
    bleed_cols = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    has_bleed = labels_df[bleed_cols].sum(axis=1) > 0
    y_stratify = has_bleed.astype(int).values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, y_stratify)):
        fold_log_dir = os.path.join(TRAINING_LOG_DIR, f"2d_run_{timestamp}_fold{fold_idx+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        best_f1, history = train_fold(fold_idx, train_idx, val_idx, dataset, fold_log_dir)
        fold_results.append(best_f1)
        all_histories.append(history)

    print("\n" + "=" * 70)
    print("📊 5折交叉验证结果")
    for i, f1 in enumerate(fold_results):
        print(f"  Fold {i+1}: {f1:.4f}")
    print(f"  平均 F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"  最高 F1: {np.max(fold_results):.4f}")

    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(all_histories):
        plt.plot(hist['val_f1'], label=f'Fold {i+1}', alpha=0.6)
    plt.axhline(y=np.mean(fold_results), color='r', linestyle='--',
                label=f'Mean={np.mean(fold_results):.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(TRAINING_LOG_DIR, "2d_kfold_summary.png"), dpi=150)
    plt.close()

    print(f"\n✅ 结果保存至: {TRAINING_LOG_DIR}")
    print(f"✅ 训练完成！日志已保存到: {log_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()