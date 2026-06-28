# trainer_optimized.py
# 3D 病灶多分类训练器（v3 稳定版 + 稀有类过采样）

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from scipy.ndimage import zoom, gaussian_filter, map_coordinates

from Conf.Config import *

torch.backends.cudnn.benchmark = True


class NiftiDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, nifti_dir, target_shape=TARGET_SHAPE_3D, transform=True):
        self.df = pd.read_csv(metadata_csv)
        self.nifti_dir = nifti_dir
        self.target_shape = target_shape
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _elastic_transform(self, volume, alpha=1, sigma=1):
        shape = volume.shape
        dx = gaussian_filter(np.random.randn(*shape), sigma, mode='constant', cval=0) * alpha
        dy = gaussian_filter(np.random.randn(*shape), sigma, mode='constant', cval=0) * alpha
        dz = gaussian_filter(np.random.randn(*shape), sigma, mode='constant', cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (x + dx, y + dy, z + dz)
        return map_coordinates(volume, indices, order=1, mode='reflect')

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

        if self.transform:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=0).copy()
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=1).copy()
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=2).copy()
            if np.random.rand() > 0.7:
                volume = self._elastic_transform(volume, alpha=1, sigma=1)
            if np.random.rand() > 0.9:
                noise = np.random.normal(0, 0.005, volume.shape)
                volume = np.clip(volume + noise, 0, 1)

        tensor = torch.from_numpy(volume).unsqueeze(0).float()
        labels = [row[f'label_{i}'] for i in range(9)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return tensor, labels_tensor


class Large3DClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=9, dropout_rate=DROPOUT_RATE_3D):
        super().__init__()
        channels = MODEL_CHANNELS
        self.enc1 = self._block(in_channels, channels[0], maxpool=True, dropout=dropout_rate)
        self.enc2 = self._block(channels[0], channels[1], maxpool=True, dropout=dropout_rate)
        self.enc3 = self._block(channels[1], channels[2], maxpool=True, dropout=dropout_rate)
        self.enc4 = self._block(channels[2], channels[3], maxpool=True, dropout=dropout_rate)
        self.enc5 = self._block(channels[3], channels[4], maxpool=True, dropout=dropout_rate)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels[4], channels[5], 3, padding=1),
            nn.BatchNorm3d(channels[5]),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate * 0.5)
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(channels[5], 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(512, num_classes)
        )

    def _block(self, in_ch, out_ch, maxpool=True, dropout=0.3):
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        ]
        if maxpool:
            layers.append(nn.MaxPool3d(2))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout * 0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.bottleneck(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    D, H, W = x.shape[2], x.shape[3], x.shape[4]

    cut_d = int(D * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))

    if cut_d < 1 or cut_h < 1 or cut_w < 1 or cut_d >= D or cut_h >= H or cut_w >= W:
        return x, y, y, 1.0

    cx = np.random.randint(0, D - cut_d)
    cy = np.random.randint(0, H - cut_h)
    cz = np.random.randint(0, W - cut_w)

    x[:, :, cx:cx+cut_d, cy:cy+cut_h, cz:cz+cut_w] = x[index, :, cx:cx+cut_d, cy:cy+cut_h, cz:cz+cut_w]
    lam = 1 - (cut_d * cut_h * cut_w) / (D * H * W)
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def compute_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds >= threshold).astype(int)
    tp = (preds_bin * targets).sum(axis=0)
    fp = ((preds_bin) * (1 - targets)).sum(axis=0)
    fn = ((1 - preds_bin) * targets).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.mean(), recall.mean(), f1.mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    axes[0,1].plot(epochs, history['val_f1'], 'm-', label='Val F1')
    axes[0,1].set_title('Validation F1')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    axes[1,0].plot(epochs, history['train_f1'], 'g-', label='Train F1')
    axes[1,0].set_title('Train F1')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    axes[1,1].plot(epochs, history['val_precision'], 'c--', label='Val Prec')
    axes[1,1].plot(epochs, history['val_recall'], 'y--', label='Val Rec')
    axes[1,1].set_title('Val Precision & Recall')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_fold(fold_idx, train_indices, val_indices, dataset, fold_log_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/5")
    print(f"  训练: {len(train_indices)}, 验证: {len(val_indices)}")
    print(f"{'='*50}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # ========== 构建过采样采样器 ==========
    if USE_OVERSAMPLING:
        # 计算每个训练样本的权重：根据其9类标签中稀有类的数量
        # 使用CLASS_WEIGHTS_3D作为各类的权重，样本权重 = sum(标签 * 权重)
        class_weights = torch.tensor(CLASS_WEIGHTS_3D)
        sample_weights = []
        for idx in train_indices:
            row = dataset.df.iloc[idx]
            labels = [row[f'label_{i}'] for i in range(9)]
            weight = sum(labels[i] * class_weights[i] for i in range(9))
            # 确保每个样本至少有权重1.0（防止全阴性样本权重为0）
            weight = max(weight, 1.0)
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False  # sampler自带shuffle
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS//2, pin_memory=PIN_MEMORY,
        prefetch_factor=2, persistent_workers=True
    )

    model = Large3DClassifier(dropout_rate=DROPOUT_RATE_3D).to(DEVICE)

    if USE_TORCH_COMPILE:
        print("⚡ 启用 torch.compile ...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️ torch.compile 失败: {e}，使用普通模式")

    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AsymmetricLoss(
        gamma_neg=4, gamma_pos=1, clip=0.05,
        pos_weight=torch.tensor(CLASS_WEIGHTS_3D).to(DEVICE)
    )

    if OPTIMIZER_3D == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR_3D, momentum=MOMENTUM_3D,
                              weight_decay=WEIGHT_DECAY_3D, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR_3D, weight_decay=WEIGHT_DECAY_3D)

    total_epochs = EPOCHS_3D
    warmup_epochs = WARMUP_EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if USE_AMP else None
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    history = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [], 'train_f1': [],
        'val_precision': [], 'val_recall': []
    }
    best_f1 = 0.0
    patience_counter = 0
    grad_accum_steps = GRAD_ACCUM_STEPS

    for epoch in range(1, EPOCHS_3D + 1):
        model.train()
        train_loss = 0
        all_train_preds, all_train_targets = [], []
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch}", leave=False)):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            if USE_MIXUP and np.random.rand() < MIXUP_PROB:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=MIXUP_ALPHA)
            elif USE_CUTMIX and np.random.rand() < CUTMIX_PROB:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=CUTMIX_ALPHA)
            else:
                y_a, y_b, lam = y, y, 1.0

            if USE_AMP:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y_a) * lam + criterion(logits, y_b) * (1 - lam)
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(x)
                loss = criterion(logits, y_a) * lam + criterion(logits, y_b) * (1 - lam)
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item()
            with torch.no_grad():
                all_train_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_train_targets.append(y.cpu().numpy())

            if ema is not None:
                ema.update()

        if (batch_idx + 1) % grad_accum_steps != 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)
        train_prec, train_rec, train_f1 = compute_metrics(all_train_preds, all_train_targets)

        if ema is not None:
            ema.apply_shadow()
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
        if ema is not None:
            ema.restore()

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

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), os.path.join(fold_log_dir, f"best_fold_{fold_idx+1}.pth"))
                ema.restore()
            else:
                torch.save(model.state_dict(), os.path.join(fold_log_dir, f"best_fold_{fold_idx+1}.pth"))
            print(f"  ✅ 新最佳 F1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE_3D:
                print(f"  ⏳ 早停触发 (patience={EARLY_STOP_PATIENCE_3D})")
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


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    log_dir = Path("/mnt/workspace/BrainCT/logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    sys.stdout = Tee(str(log_file))

    print(f"📝 日志将保存到: {log_file}")
    print(f"🚀 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("🚀 5折交叉验证训练（v3稳定版 + 稀有类过采样）")
    print(f"📍 设备: {DEVICE}, 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"📊 模型: Large3DClassifier (~7M参数)")
    print(f"📈 学习率: {LR_3D}, Weight Decay: {WEIGHT_DECAY_3D}")
    print(f"🎯 Dropout: {DROPOUT_RATE_3D}, Label Smoothing: {LABEL_SMOOTHING}")
    print(f"📦 Batch Size: {BATCH_SIZE_TRAIN}, Workers: {NUM_WORKERS}")
    print(f"🔄 过采样: {USE_OVERSAMPLING}")
    print("=" * 70)

    dataset = NiftiDataset(METADATA_CSV, NIFTI_DIR, transform=True)
    print(f"📚 数据集: {len(dataset)} 例")

    labels_df = dataset.df
    bleed_cols = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    has_bleed = labels_df[bleed_cols].sum(axis=1) > 0
    y_stratify = has_bleed.astype(int).values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, y_stratify)):
        fold_log_dir = os.path.join(TRAINING_LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fold{fold_idx+1}")
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
    plt.savefig(os.path.join(TRAINING_LOG_DIR, "kfold_summary.png"), dpi=150)
    plt.close()

    print(f"\n✅ 结果保存至: {TRAINING_LOG_DIR}")
    print(f"✅ 训练完成！日志已保存到: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()