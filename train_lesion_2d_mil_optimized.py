# train_lesion_2d_mil_optimized.py
"""
2D切片 + 多窗输入 + MIL + TopK Pooling + 可选候选切片/独立头 (快速2折验证)
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

# ==================== 候选切片筛选（可选） ====================
def compute_slice_score(slice_img):
    std = np.std(slice_img)
    from scipy.ndimage import sobel
    grad_x = sobel(slice_img, axis=0)
    grad_y = sobel(slice_img, axis=1)
    edge = np.sqrt(grad_x**2 + grad_y**2).mean()
    high_hu_ratio = (slice_img > 200).sum() / slice_img.size
    return std * 10 + edge * 100 + high_hu_ratio * 100

def select_candidate_slices(volume, num_slices, target_size):
    D, H, W = volume.shape
    if D <= num_slices:
        indices = np.linspace(0, D-1, num_slices, dtype=int).tolist()
    else:
        scores = []
        for z in range(D):
            slice_img = volume[z, :, :]
            if slice_img.max() - slice_img.min() < 10:
                scores.append(-1e9)
            else:
                scores.append(compute_slice_score(slice_img))
        sorted_indices = np.argsort(scores)[::-1]
        indices = sorted_indices[:num_slices].tolist()
        indices = sorted(indices)
    slices = []
    for si in indices:
        slice_img = volume[si, :, :]
        h, w = slice_img.shape
        if (h, w) != target_size:
            zoom_factors = (target_size[0]/h, target_size[1]/w)
            slice_img = zoom(slice_img, zoom_factors, order=1)
        slices.append(slice_img)
    return np.stack(slices, axis=0)

# ==================== 弹性变形 ====================
def elastic_transform_2d(image, alpha=0.5, sigma=1):
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
                 context_slices=CONTEXT_SLICES, transform=True,
                 candidate_selection=CANDIDATE_SELECTION):
        self.df = pd.read_csv(metadata_csv)
        self.nifti_dir = nifti_dir
        self.target_size = target_size
        self.num_slices = num_slices
        self.use_2d5 = use_2d5
        self.context_slices = context_slices
        self.transform = transform
        self.candidate_selection = candidate_selection

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
        if self.candidate_selection and self.num_slices > 1:
            selected_slices = select_candidate_slices(volume, self.num_slices, self.target_size)
        else:
            if self.num_slices == 1:
                slice_indices = [D // 2]
            else:
                slice_indices = np.linspace(0, D-1, self.num_slices, dtype=int).tolist()
            selected_slices = []
            for si in slice_indices:
                slice_img = volume[si, :, :]
                h, w = slice_img.shape
                if (h, w) != self.target_size:
                    zoom_factors = (self.target_size[0]/h, self.target_size[1]/w)
                    slice_img = zoom(slice_img, zoom_factors, order=1)
                selected_slices.append(slice_img)
            selected_slices = np.stack(selected_slices, axis=0)

        slices_multi = []
        for i in range(selected_slices.shape[0]):
            slice_img = selected_slices[i]
            multi_win = generate_multi_window_slice(slice_img)
            slices_multi.append(multi_win)
        combined = np.stack(slices_multi, axis=0)

        if self.transform:
            for i in range(combined.shape[0]):
                if np.random.rand() > 0.5:
                    combined[i] = np.flip(combined[i], axis=2).copy()
                if np.random.rand() > 0.5:
                    combined[i] = np.flip(combined[i], axis=1).copy()
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    for c in range(combined.shape[1]):
                        combined[i, c] = rotate(combined[i, c], angle, reshape=False, order=1)
                if np.random.rand() > 0.5:
                    contrast = np.random.uniform(0.8, 1.2)
                    brightness = np.random.uniform(-0.05, 0.05)
                    combined[i] = combined[i] * contrast + brightness
                    combined[i] = np.clip(combined[i], 0, 1)
                if np.random.rand() > 0.7:
                    for c in range(combined.shape[1]):
                        combined[i, c] = elastic_transform_2d(combined[i, c], alpha=0.3, sigma=1)
                if np.random.rand() > 0.8:
                    noise = np.random.normal(0, 0.005, combined[i].shape)
                    combined[i] = np.clip(combined[i] + noise, 0, 1)

        slices_tensor = torch.from_numpy(combined).float()
        labels = [row[f'label_{i}'] for i in range(9)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return slices_tensor, labels_tensor

# ==================== MIL模型 ====================
class MILModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS,
                 pretrained=True, pooling=MIL_POOLING, topk=MIL_TOPK,
                 attention_dim=MIL_ATTENTION_DIM,
                 independent_heads=USE_INDEPENDENT_HEADS):
        super().__init__()
        self.pooling = pooling
        self.topk = topk
        self.independent_heads = independent_heads

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

        if independent_heads:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(self.feature_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1)
                ) for _ in range(num_classes)
            ])
        else:
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
        if self.independent_heads:
            for head in self.heads:
                head.apply(init_linear)
        else:
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

        if self.independent_heads:
            logits = torch.cat([head(bag_feature) for head in self.heads], dim=1)
        else:
            logits = self.classifier(bag_feature)
        return logits, None

# ==================== 评估函数 ====================
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
    print(f"Fold {fold_idx+1}/2")   # 修改为2折
    print(f"  训练: {len(train_indices)}, 验证: {len(val_indices)}")
    print(f"{'='*50}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

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
        attention_dim=MIL_ATTENTION_DIM,
        independent_heads=USE_INDEPENDENT_HEADS
    ).to(DEVICE)

    if PRETRAIN_WEIGHT_PATH and os.path.exists(PRETRAIN_WEIGHT_PATH):
        print(f"📥 加载医学预训练: {PRETRAIN_WEIGHT_PATH}")
        state_dict = torch.load(PRETRAIN_WEIGHT_PATH, map_location='cpu')
        if 'backbone' in state_dict:
            model.backbone.load_state_dict(state_dict['backbone'], strict=False)
        else:
            model.backbone.load_state_dict(state_dict, strict=False)
        print("✅ 加载完成")
    else:
        print("ℹ️ 使用ImageNet预训练")

    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Pooling: {MIL_POOLING}, TopK: {MIL_TOPK if MIL_POOLING == 'topk' else 'N/A'}")
    print(f"   输入通道: {INPUT_CHANNELS} ({'2.5D' if USE_2D5 else '2D多窗'})")
    print(f"   切片数/病例: {NUM_SLICES_PER_VOLUME}")
    print(f"   候选切片筛选: {CANDIDATE_SELECTION}")
    print(f"   独立分类头: {USE_INDEPENDENT_HEADS}")

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
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'train_f1': [], 
               'val_precision': [], 'val_recall': [], 'best_thresholds': []}
    best_f1 = 0.0
    best_thresholds = [0.3] * NUM_CLASSES
    patience_counter = 0

    for epoch in range(1, EPOCHS_3D + 1):
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

        if USE_PER_CLASS_THRESHOLD:
            best_thresholds, class_f1 = search_best_thresholds(
                all_val_preds, all_val_targets,
                n_steps=THRESHOLD_SEARCH_STEPS
            )
        val_prec, val_rec, val_f1, _ = compute_metrics_with_thresholds(
            all_val_preds, all_val_targets, best_thresholds
        )

        history['best_thresholds'].append(best_thresholds)
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

# ==================== 多标签分层KFold（支持任意折数） ====================
class MultilabelStratifiedKFold:
    def __init__(self, n_splits=2, shuffle=True, random_state=42):  # 默认2折
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
    log_file = log_dir / f"training_mil_optimized_{timestamp}.log"
    sys.stdout = Tee(str(log_file))

    print(f"📝 日志将保存到: {log_file}")
    print(f"🚀 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("🚀 MIL + 多窗 + TopK Pooling (2折快速验证)")
    print(f"   - Pooling: {MIL_POOLING}, TopK: {MIL_TOPK if MIL_POOLING=='topk' else 'N/A'}")
    print(f"   - 输入: {'2.5D' if USE_2D5 else '2D多窗'} ({INPUT_CHANNELS} 通道)")
    print(f"   - Loss: BCEWithLogitsLoss + 动态pos_weight")
    print(f"   - 阈值: 类别独立搜索")
    print(f"   - 切片数: {NUM_SLICES_PER_VOLUME} 张/病例")
    print(f"   - 候选切片筛选: {CANDIDATE_SELECTION}")
    print(f"   - 独立分类头: {USE_INDEPENDENT_HEADS}")
    if PRETRAIN_WEIGHT_PATH:
        print(f"   - 医学预训练权重: {PRETRAIN_WEIGHT_PATH}")
    else:
        print(f"   - 预训练: ImageNet (默认)")
    print("=" * 70)

    dataset = NiftiMILDataset(
        METADATA_CSV, NIFTI_DIR,
        target_size=TARGET_SIZE_2D,
        num_slices=NUM_SLICES_PER_VOLUME,
        use_2d5=USE_2D5,
        context_slices=CONTEXT_SLICES,
        transform=True,
        candidate_selection=CANDIDATE_SELECTION
    )
    print(f"📚 数据集: {len(dataset)} 例")
    print(f"   每例切片数: {NUM_SLICES_PER_VOLUME}")
    print(f"   候选切片筛选: {CANDIDATE_SELECTION}")

    all_labels = []
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        labels = [row[f'label_{i}'] for i in range(9)]
        all_labels.append(labels)
    y_multilabel = np.array(all_labels)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MLSKF
        kf = MLSKF(n_splits=2, shuffle=True, random_state=42)   # 改为2折
        print("✅ 使用 iterstrat.MultilabelStratifiedKFold (2折)")
    except ImportError:
        print("⚠️  iterstrat未安装，使用简化版 (2折)")
        kf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    fold_results = []
    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), y_multilabel)):
        fold_log_dir = os.path.join(TRAINING_LOG_DIR, f"mil_optimized_{timestamp}_fold{fold_idx+1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        best_f1, history = train_fold(fold_idx, train_idx, val_idx, dataset, fold_log_dir)
        fold_results.append(best_f1)
        all_histories.append(history)

    print("\n" + "=" * 70)
    print("📊 2折交叉验证结果 (F1@独立阈值)")
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
    plt.savefig(os.path.join(TRAINING_LOG_DIR, "mil_2fold_summary.png"), dpi=150)
    plt.close()

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