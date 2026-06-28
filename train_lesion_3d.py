# train_lesion_3d_v3.py
# K-Fold 交叉验证 + 轻量级模型 + 自适应学习率

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

# ----------------- 配置 -----------------
METADATA_CSV = "metadata/dataset_metadata.csv"
NIFTI_DIR = "datasets_nifti"
BATCH_SIZE = 8
EPOCHS_PER_FOLD = 80
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9
TARGET_SHAPE = (128, 128, 64)
N_FOLDS = 5
EARLY_STOP_PATIENCE = 12
PLOT_EVERY = 10
LOG_BASE = "文档/训练记录"
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
# ----------------------------------------

# 创建日志目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_BASE, f"run_{timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
shutil.copy2(os.path.abspath(__file__), os.path.join(LOG_DIR, "train_script.py"))

log_txt_path = os.path.join(LOG_DIR, "train_log.txt")

class TeeLogger:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

logger = TeeLogger(log_txt_path)
sys.stdout = logger

# ----------------- 数据集 -----------------
class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, nifti_dir, target_shape=TARGET_SHAPE, transform=True):
        self.df = pd.read_csv(metadata_csv)
        self.nifti_dir = nifti_dir
        self.target_shape = target_shape
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['case_id']
        nifti_path = os.path.join(self.nifti_dir, f"{case_id}.nii.gz")
        img = nib.load(nifti_path)
        volume = img.get_fdata().astype(np.float32)

        volume = np.clip(volume, -100, 100)
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)

        curr_shape = volume.shape
        zoom_factors = [TARGET_SHAPE[0]/curr_shape[0],
                        TARGET_SHAPE[1]/curr_shape[1],
                        TARGET_SHAPE[2]/curr_shape[2]]
        volume = zoom(volume, zoom_factors, order=1)

        if self.transform:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=0).copy()
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=1).copy()
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=2).copy()
            # 小幅度随机旋转（仅限轴位面）
            if np.random.rand() > 0.5:
                from scipy.ndimage import rotate
                angle = np.random.uniform(-10, 10)
                for i in range(volume.shape[0]):
                    volume[i] = rotate(volume[i], angle, reshape=False, order=1)

        tensor = torch.from_numpy(volume).unsqueeze(0).float()
        labels = [row[f'label_{i}'] for i in range(NUM_CLASSES)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return tensor, labels_tensor

# ----------------- 轻量级 3D 模型（参数量 ~5M） -----------------
class Lightweight3DClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        # 编码器
        self.enc1 = self._block(in_channels, 16, maxpool=True)
        self.enc2 = self._block(16, 32, maxpool=True)
        self.enc3 = self._block(32, 64, maxpool=True)
        self.enc4 = self._block(64, 128, maxpool=True)
        
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate)
        )
        
        # 全局池化 + 分类
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, num_classes)
        )
        
    def _block(self, in_ch, out_ch, maxpool=True):
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        ]
        if maxpool:
            layers.append(nn.MaxPool3d(2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------- 损失 + 指标 -----------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight_neg = (1 - p_t) ** self.gamma_neg
        weight_pos = (1 - p_t) ** self.gamma_pos
        weights = targets * weight_pos + (1 - targets) * weight_neg
        weights = torch.clamp(weights, min=self.clip)
        return (weights * bce).mean()

def compute_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds >= threshold).astype(int)
    tp = (preds_bin * targets).sum(axis=0)
    fp = ((preds_bin) * (1 - targets)).sum(axis=0)
    fn = ((1 - preds_bin) * targets).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.mean(), recall.mean(), f1.mean()

# ----------------- K-Fold 单折训练 -----------------
def train_fold(fold_idx, train_indices, val_indices, dataset, fold_log_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/{N_FOLDS}")
    print(f"  训练样本: {len(train_indices)}, 验证样本: {len(val_indices)}")
    print(f"{'='*50}")
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = Lightweight3DClassifier().to(DEVICE)
    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS_PER_FOLD + 1):
        # 训练
        model.train()
        train_loss = 0
        all_train_preds, all_train_targets = [], []
        for x, y in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():
                all_train_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_train_targets.append(y.cpu().numpy())
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_val_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_val_targets.append(y.cpu().numpy())
        val_loss /= len(val_loader)
        
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)
        
        _, _, train_f1 = compute_metrics(all_train_preds, all_train_targets)
        _, _, val_f1 = compute_metrics(all_val_preds, all_val_targets)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}, LR={lr:.6f}")
        
        scheduler.step(val_loss)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_log_dir, f"best_fold_{fold_idx+1}.pth"))
            print(f"  ✅ 新最佳 F1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  ⏳ 早停触发 (耐心 {EARLY_STOP_PATIENCE})")
                break
        
        if epoch % PLOT_EVERY == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.subplot(1, 2, 2)
            plt.plot(history['val_f1'], label='Val F1')
            plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best F1={best_f1:.3f}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(fold_log_dir, f"fold_{fold_idx+1}_epoch{epoch}.png"), dpi=150)
            plt.close()
    
    print(f"  Fold {fold_idx+1} 完成，最佳 F1={best_f1:.4f} (Epoch {best_epoch})")
    return best_f1, history

# ----------------- 主函数 -----------------
def main():
    print("="*70)
    print(f"🚀 K-Fold 交叉验证训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 设备: {DEVICE}")
    print(f"📁 日志目录: {LOG_DIR}")
    print(f"📊 折数: {N_FOLDS}, Epochs/折: {EPOCHS_PER_FOLD}")
    print(f"📈 学习率: {LR}, Weight Decay: {WEIGHT_DECAY}")
    print(f"🎯 早停耐心值: {EARLY_STOP_PATIENCE}")
    print(f"🧠 模型类型: Lightweight3DClassifier (~5M 参数)")
    print("="*70)

    # 加载数据
    dataset = NiftiDataset(METADATA_CSV, NIFTI_DIR, transform=True)
    print(f"📚 数据集总数: {len(dataset)}")

    # K-Fold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        fold_log_dir = os.path.join(LOG_DIR, f"fold_{fold_idx+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        
        best_f1, history = train_fold(fold_idx, train_idx, val_idx, dataset, fold_log_dir)
        fold_results.append(best_f1)
        all_histories.append(history)

    # 汇总
    print("\n" + "="*70)
    print("📊 K-Fold 汇总")
    print("="*70)
    for i, f1 in enumerate(fold_results):
        print(f"  Fold {i+1}: {f1:.4f}")
    print(f"  平均 F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

    # 保存汇总
    with open(os.path.join(LOG_DIR, "summary.txt"), 'w') as f:
        f.write(f"K-Fold Results ({N_FOLDS} folds):\n")
        for i, f1 in enumerate(fold_results):
            f.write(f"  Fold {i+1}: {f1:.4f}\n")
        f.write(f"  Mean: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}\n")

    # 绘制汇总图
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(all_histories):
        plt.plot(hist['val_f1'], label=f'Fold {i+1}', alpha=0.6)
    plt.axhline(y=np.mean(fold_results), color='r', linestyle='--', label=f'Mean F1={np.mean(fold_results):.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('K-Fold Validation F1 Curves')
    plt.savefig(os.path.join(LOG_DIR, "kfold_summary.png"), dpi=150)
    plt.close()

    print(f"\n✅ 所有日志保存至: {LOG_DIR}")
    print("="*70)

    # 建议：使用全部数据训练最终模型
    print("\n💡 建议: 使用全部数据重新训练最终模型 (运行 train_final_model.py)")

    sys.stdout = logger.stdout
    logger.close()

if __name__ == "__main__":
    main()