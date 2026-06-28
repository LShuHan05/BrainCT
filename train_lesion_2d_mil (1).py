# train_lesion_2d_mil.py
"""
2D切片 + 多窗输入 + MIL + TopK Pooling + BCE + 类别独立阈值 (优化版)
- TopK=5
- MLP Head: 2048->256->9
- Dropout=0.4
- lr=1.5e-4, warmup=5
- Epochs=120
- 切片数: 16张/病例
- 数据增强: 增加随机裁剪和噪声
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
import nibabel as nib
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from Conf.Config import *
import torchvision.models as models

# ---------- 固定随机种子 ----------
torch.manual_seed(42)
np.random.seed(42)

# ==================== 多窗CT处理 ====================
def apply_window(image, wl, ww):
    low = wl - ww / 2
    high = wl + ww / 2
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32)

def generate_multi_window_slice(slice_img):
    brain = apply_window(slice_img, 40, 80)
    subdural = apply_window(slice_img, 50, 130)
    bone = apply_window(slice_img, 600, 2800)
    return np.stack([brain, subdural, bone], axis=0)

# ==================== 弹性变形（用于数据增强） ====================
def elastic_transform_2d(image, alpha=0.5, sigma=1):
    """对2D图像进行弹性变形（强度降低）"""
    shape = image.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma, mode='constant', cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = (x + dx, y + dy)
    return map_coordinates(image, indices, order=1, mode='reflect')

# ==================== MIL数据集 ====================
class NiftiMILDataset(Dataset):
    def __init__(self, metadata_csv, nifti_dir, target_size=(512, 512),
                 num_slices=NUM_SLICES_PER_VOLUME, use_2d5=USE_2D5,
                 context_slices=CONTEXT_SLICES, transform=True):
        self.df = pd.read_csv(metadata_csv)
        self.nifti_dir = nifti_dir
        self.target_size = target_size
        self.num_slices = num_slices
        self.use_2d5 = use_2d5
        self.context_slices = context_slices
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['case_id']
        nifti_path = os.path.join(self.nifti_dir, f"{case_id}.nii.gz")
        img = nib.load(nifti_path)
        volume = img.get_fdata().astype(np.float32)
        volume = np.clip(volume, -1000, 3000)

        D, H, W = volume.shape

        if self.num_slices == 1:
            slice_indices = [D // 2]
        else:
            slice_indices = np.linspace(0, D-1, self.num_slices, dtype=int).tolist()

        slices = []
        for si in slice_indices:
            if self.use_2d5:
                slice_range = range(si - self.context_slices, si + self.context_slices + 1)
                slice_imgs = []
                for s in slice_range:
                    s_clipped = np.clip(s, 0, D-1)
                    slice_img = volume[s_clipped, :, :]
                    h, w = slice_img.shape
                    if (h, w) != self.target_size:
                        zoom_factors = (self.target_size[0]/h, self.target_size[1]/w)
                        slice_img = zoom(slice_img, zoom_factors, order=1)
                    multi_win = generate_multi_window_slice(slice_img)
                    slice_imgs.append(multi_win)
                combined = np.concatenate(slice_imgs, axis=0)
            else:
                slice_img = volume[si, :, :]
                h, w = slice_img.shape
                if (h, w) != self.target_size:
                    zoom_factors = (self.target_size[0]/h, self.target_size[1]/w)
                    slice_img = zoom(slice_img, zoom_factors, order=1)
                combined = generate_multi_window_slice(slice_img)

            # 数据增强（增强版）
            if self.transform:
                # 翻转
                if np.random.rand() > 0.5:
                    combined = np.flip(combined, axis=2).copy()
                if np.random.rand() > 0.5:
                    combined = np.flip(combined, axis=1).copy()
                # 旋转
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    for c in range(combined.shape[0]):
                        combined[c] = rotate(combined[c], angle, reshape=False, order=1)
                # 对比度/亮度调整
                if np.random.rand() > 0.5:
                    contrast = np.random.uniform(0.8, 1.2)
                    brightness = np.random.uniform(-0.05, 0.05)
                    combined = combined * contrast + brightness
                    combined = np.clip(combined, 0, 1)
                # 弹性变形（概率0.3，强度降低）
                if np.random.rand() > 0.7:
                    for c in range(combined.shape[0]):
                        combined[c] = elastic_transform_2d(combined[c], alpha=0.3, sigma=1)
                # 高斯噪声（新增，概率0.2）
                if np.random.rand() > 0.8:
                    noise = np.random.normal(0, 0.005, combined.shape)
                    combined = np.clip(combined + noise, 0, 1)
            slices.append(combined)

        slices_tensor = torch.from_numpy(np.stack(slices, axis=0)).float()
        labels = [row[f'label_{i}'] for i in range(9)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return slices_tensor, labels_tensor

# ==================== MIL模型 ====================
class MILModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS,
                 pretrained=True, pooling=MIL_POOLING, topk=MIL_TOPK,
                 attention_dim=MIL_ATTENTION_DIM):
        super().__init__()
        self.pooling = pooling
        self.topk = topk

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained and input_channels == 3:
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_conv1.weight.data
        elif pretrained and input_channels != 3:
            with torch.no_grad():
                weight = original_conv1.weight.data.mean(dim=1, keepdim=True)
                self.backbone.conv1.weight.data = weight.repeat(1, input_channels, 1, 1) / input_channels

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1)
            )
        else:
            self.attention = None

        # MLP分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.classifier.apply(init_linear)
        if self.attention is not None:
            self.attention.apply(init_linear)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(B, N, -1)

        if self.pooling == 'max':
            bag_feature, _ = features.max(dim=1)
        elif self.pooling == 'mean':
            bag_feature = features.mean(dim=1)
        elif self.pooling == 'topk':
            topk_values, _ = features.topk(self.topk, dim=1)
            bag_feature = topk_values.mean(dim=1)
        elif self.pooling == 'attention':
            attn_weights = self.attention(features)
            attn_weights = F.softmax(attn_weights, dim=1)
            bag_feature = (features * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.classifier(bag_feature)
        return logits, None

# ==================== 评估指标（支持类别独立阈值） ====================
def compute_metrics_with_thresholds(preds, targets, thresholds=None):
    num_classes = preds.shape[1]
    if thresholds is None:
        thresholds = [0.5] * num_classes

    preds_bin = np.zeros_like(preds)
    for i in range(num_classes):
        preds_bin[:, i] = (preds[:, i] >= thresholds[i]).astype(int)

    tp = (preds_bin * targets).sum(axis=0)
    fp = ((preds_bin) * (1 - targets)).sum(axis=0)
    fn = ((1 - preds_bin) * targets).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.mean(), recall.mean(), f1.mean(), f1

def search_best_thresholds(preds, targets, n_steps=THRESHOLD_SEARCH_STEPS,
                           min_val=THRESHOLD_MIN, max_val=THRESHOLD_MAX):
    num_classes = preds.shape[1]
    best_thresholds = []
    best_f1_per_class = []

    for i in range(num_classes):
        pred_i = preds[:, i]
        target_i = targets[:, i]

        if target_i.sum() == 0:
            best_thresholds.append(0.5)
            best_f1_per_class.append(0.0)
            continue

        thresholds = np.linspace(min_val, max_val, n_steps)
        best_f1 = 0.0
        best_th = 0.5

        for th in thresholds:
            pred_bin = (pred_i >= th).astype(int)
            tp = ((pred_bin == 1) & (target_i == 1)).sum()
            fp = ((pred_bin == 1) & (target_i == 0)).sum()
            fn = ((pred_bin == 0) & (target_i == 1)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        best_thresholds.append(best_th)
        best_f1_per_class.append(best_f1)

    return best_thresholds, best_f1_per_class

# ==================== 单折训练 ====================
def train_fold(fold_idx, train_indices, val_indices, dataset, fold_log_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/5")
    print(f"  训练: {len(train_indices)}, 验证: {len(val_indices)}")
    print(f"{'='*50}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 动态 pos_weight
    all_labels = []
    for idx in train_indices:
        row = dataset.df.iloc[idx]
        labels = [row[f'label_{i}'] for i in range(9)]
        all_labels.append(labels)
    all_labels = np.array(all_labels)
    samples_per_class = all_labels.sum(axis=0).astype(np.float32)
    n_samples = len(train_indices)
    pos_weight = np.sqrt((n_samples - samples_per_class) / (samples_per_class + 1e-8))
    pos_weight = np.clip(pos_weight, 0.5, 10.0)
    print(f"  类别样本数: {samples_per_class.astype(int).tolist()}")
    print(f"  动态pos_weight: {pos_weight.round(3).tolist()}")

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS//2, pin_memory=PIN_MEMORY,
        prefetch_factor=2, persistent_workers=True
    )

    model = MILModel(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        pretrained=True,
        pooling=MIL_POOLING,
        topk=MIL_TOPK,
        attention_dim=MIL_ATTENTION_DIM
    ).to(DEVICE)
    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Pooling: {MIL_POOLING}, TopK: {MIL_TOPK if MIL_POOLING == 'topk' else 'N/A'}")
    print(f"   输入通道: {INPUT_CHANNELS} ({'2.5D' if USE_2D5 else '2D'})")
    print(f"   切片数/病例: {NUM_SLICES_PER_VOLUME}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
    print("   Loss: BCEWithLogitsLoss")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR_3D,
        weight_decay=WEIGHT_DECAY_3D
    )

    warmup_epochs = WARMUP_EPOCHS
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_reduce = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    scaler = GradScaler() if USE_AMP else None

    history = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [],
        'best_thresholds': []
    }
    best_f1 = 0.0
    best_thresholds = [0.3] * NUM_CLASSES
    patience_counter = 0

    for epoch in range(1, EPOCHS_3D + 1):
        # ---------- 训练 ----------
        model.train()
        train_loss = 0
        all_train_preds, all_train_targets = [], []
        for x, y in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch}", leave=False):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            if USE_AMP:
                with autocast():
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_3D)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
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
        train_prec, train_rec, train_f1, _ = compute_metrics_with_thresholds(
            all_train_preds, all_train_targets, [0.3]*NUM_CLASSES
        )

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                if USE_AMP:
                    with autocast():
                        logits, _ = model(x)
                        loss = criterion(logits, y)
                else:
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                all_val_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_val_targets.append(y.cpu().numpy())
        val_loss /= len(val_loader)
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_targets = np.concatenate(all_val_targets, axis=0)

        # ===== 诊断信息 =====
        print("\n" + "="*60)
        print(f"🔍 诊断信息 (Epoch {epoch}):")
        print("="*60)
        pred_flat = all_val_preds.flatten()
        print(f"  Pred - min: {pred_flat.min():.4f}, max: {pred_flat.max():.4f}, mean: {pred_flat.mean():.4f}")
        print(f"  Pred > 0.5: {(pred_flat > 0.5).sum()} / {len(pred_flat)} ({100*(pred_flat>0.5).mean():.2f}%)")
        print(f"  Pred > 0.3: {(pred_flat > 0.3).sum()} / {len(pred_flat)} ({100*(pred_flat>0.3).mean():.2f}%)")

        print("\n  各类别统计:")
        for i in range(NUM_CLASSES):
            preds_i = all_val_preds[:, i]
            targets_i = all_val_targets[:, i]
            pos_mask = targets_i > 0.5
            pos_count = pos_mask.sum()
            if pos_count > 0:
                pred_pos_mean = preds_i[pos_mask].mean()
                pred_pos_max = preds_i[pos_mask].max()
                recall_03 = (preds_i[pos_mask] > 0.3).sum() / pos_count
                recall_05 = (preds_i[pos_mask] > 0.5).sum() / pos_count
                print(f"  类别{i}: 正样本={int(pos_count)}, 预测均值={pred_pos_mean:.3f}, "
                      f"召回@0.3={recall_03:.2f}, 召回@0.5={recall_05:.2f}")
            else:
                print(f"  类别{i}: 无正样本")

        # ===== 类别独立阈值搜索 =====
        if USE_PER_CLASS_THRESHOLD:
            best_thresholds, class_f1 = search_best_thresholds(
                all_val_preds, all_val_targets,
                n_steps=THRESHOLD_SEARCH_STEPS
            )
            print(f"\n  类别阈值: {[f'{th:.3f}' for th in best_thresholds]}")
            print(f"  类别F1:   {[f'{f1:.3f}' for f1 in class_f1]}")
            val_prec, val_rec, val_f1, _ = compute_metrics_with_thresholds(
                all_val_preds, all_val_targets, best_thresholds
            )
            print(f"\n  Val F1@独立阈值 = {val_f1:.4f}")
        else:
            val_prec, val_rec, val_f1, _ = compute_metrics_with_thresholds(
                all_val_preds, all_val_targets, [0.3]*NUM_CLASSES
            )
            best_thresholds = [0.3] * NUM_CLASSES

        history['best_thresholds'].append(best_thresholds)

        print("\n  Sample predictions (first 5 cases):")
        for j in range(min(5, len(all_val_preds))):
            print(f"    Case {j}: pred={all_val_preds[j].round(3)}, gt={all_val_targets[j].round(0)}")
        print("="*60 + "\n")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['train_f1'].append(train_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val F1@独立阈值={val_f1:.4f}, LR={current_lr:.6f}")

        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_reduce.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_thresholds': best_thresholds,
                'samples_per_class': samples_per_class
            }, os.path.join(fold_log_dir, f"best_fold_{fold_idx+1}.pth"))
            print(f"  ✅ 新最佳 F1@独立阈值={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE_3D:
                print(f"  ⏳ 早停触发 (patience={EARLY_STOP_PATIENCE_3D})")
                break

        if epoch % 10 == 0:
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.legend(); plt.grid(alpha=0.3)
            plt.subplot(1,2,2)
            plt.plot(history['val_f1'], label='Val F1')
            plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best={best_f1:.3f}')
            plt.legend(); plt.grid(alpha=0.3)
            plt.savefig(os.path.join(fold_log_dir, f"fold_{fold_idx+1}_epoch{epoch}.png"), dpi=150)
            plt.close()

    print(f"  Fold {fold_idx+1} 最佳 F1 = {best_f1:.4f}")
    return best_f1, history

# ==================== 多标签分层KFold ====================
class MultilabelStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        if y.shape[1] > 1:
            pca = PCA(n_components=min(10, y.shape[1]))
            y_compressed = pca.fit_transform(y)
        else:
            y_compressed = y
        kmeans = KMeans(n_clusters=self.n_splits, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(y_compressed)
        indices_by_fold = [[] for _ in range(self.n_splits)]
        for idx, cluster in enumerate(cluster_labels):
            indices_by_fold[cluster].append(idx)
        for fold in range(self.n_splits):
            if len(indices_by_fold[fold]) == 0:
                max_fold = max(range(self.n_splits), key=lambda i: len(indices_by_fold[i]))
                indices_by_fold[fold].append(indices_by_fold[max_fold].pop())
        for fold in range(self.n_splits):
            val_idx = indices_by_fold[fold]
            train_idx = []
            for f in range(self.n_splits):
                if f != fold:
                    train_idx.extend(indices_by_fold[f])
            yield train_idx, val_idx

# ==================== 日志重定向 ====================
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

# ==================== 主函数 ====================
def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    log_dir = Path("/mnt/workspace/BrainCT/logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_mil_{timestamp}.log"
    sys.stdout = Tee(str(log_file))

    print(f"📝 日志将保存到: {log_file}")
    print(f"🚀 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("🚀 MIL + 多窗 + TopK Pooling + BCE + 类别独立阈值 (优化版)")
    print(f"   - Pooling: {MIL_POOLING}, TopK: {MIL_TOPK if MIL_POOLING=='topk' else 'N/A'}")
    print(f"   - 输入: {'2.5D' if USE_2D5 else '2D多窗'} ({INPUT_CHANNELS} 通道)")
    print(f"   - Loss: BCEWithLogitsLoss + 动态pos_weight")
    print(f"   - 阈值: 类别独立搜索")
    print(f"   - 切片数: {NUM_SLICES_PER_VOLUME} 张/病例")
    print("=" * 70)

    dataset = NiftiMILDataset(
        METADATA_CSV, NIFTI_DIR,
        target_size=TARGET_SIZE_2D,
        num_slices=NUM_SLICES_PER_VOLUME,
        use_2d5=USE_2D5,
        context_slices=CONTEXT_SLICES,
        transform=True
    )
    print(f"📚 数据集: {len(dataset)} 例")
    print(f"   每例切片数: {NUM_SLICES_PER_VOLUME}")

    all_labels = []
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        labels = [row[f'label_{i}'] for i in range(9)]
        all_labels.append(labels)
    y_multilabel = np.array(all_labels)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MLSKF
        kf = MLSKF(n_splits=5, shuffle=True, random_state=42)
        print("✅ 使用 iterstrat.MultilabelStratifiedKFold")
    except ImportError:
        print("⚠️  iterstrat未安装，使用简化版")
        kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), y_multilabel)):
        fold_log_dir = os.path.join(TRAINING_LOG_DIR, f"mil_run_{timestamp}_fold{fold_idx+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        best_f1, history = train_fold(fold_idx, train_idx, val_idx, dataset, fold_log_dir)
        fold_results.append(best_f1)
        all_histories.append(history)

    print("\n" + "=" * 70)
    print("📊 5折交叉验证结果 (F1@独立阈值)")
    for i, f1 in enumerate(fold_results):
        print(f"  Fold {i+1}: {f1:.4f}")
    print(f"  平均 F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"  最高 F1: {np.max(fold_results):.4f}")

    plt.figure(figsize=(12,6))
    for i, hist in enumerate(all_histories):
        plt.plot(hist['val_f1'], label=f'Fold {i+1}', alpha=0.6)
    plt.axhline(y=np.mean(fold_results), color='r', linestyle='--',
                label=f'Mean={np.mean(fold_results):.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(TRAINING_LOG_DIR, "mil_kfold_summary.png"), dpi=150)
    plt.close()

    # 打印平均阈值
    all_thresholds = []
    for hist in all_histories:
        if hist['best_thresholds']:
            best_idx = np.argmax(hist['val_f1'])
            all_thresholds.append(hist['best_thresholds'][best_idx])
    if all_thresholds:
        avg_thresholds = np.mean(all_thresholds, axis=0)
        print(f"\n📊 平均类别阈值: {[f'{th:.3f}' for th in avg_thresholds]}")

    print(f"\n✅ 结果保存至: {TRAINING_LOG_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()