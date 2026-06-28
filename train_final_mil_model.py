# train_final_mil_model.py
"""
使用全部数据训练最终 MIL 模型（无交叉验证，用于部署）
从 Config 读取所有超参数，自动继承候选切片、独立头等优化配置
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from Conf.Config import *
from train_lesion_2d_mil_optimized import (
    NiftiMILDataset, MILModel, 
    compute_metrics_with_thresholds, search_best_thresholds
)

# ---------- 固定随机种子 ----------
torch.manual_seed(42)
np.random.seed(42)

def train_full():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 70)
    print("🚀 最终模型训练（使用全部数据）")
    print("=" * 70)
    print(f"   切片数/病例: {NUM_SLICES_PER_VOLUME}")
    print(f"   候选切片筛选: {CANDIDATE_SELECTION}")
    print(f"   独立分类头: {USE_INDEPENDENT_HEADS}")
    print(f"   预训练: {'医学权重' if PRETRAIN_WEIGHT_PATH else 'ImageNet'}")
    print("=" * 70)

    # 1. 加载完整数据集
    dataset = NiftiMILDataset(
        metadata_csv=METADATA_CSV,
        nifti_dir=NIFTI_DIR,
        target_size=TARGET_SIZE_2D,
        num_slices=NUM_SLICES_PER_VOLUME,
        use_2d5=USE_2D5,
        context_slices=CONTEXT_SLICES,
        transform=True,
        candidate_selection=CANDIDATE_SELECTION
    )
    print(f"📚 总样本数: {len(dataset)}")

    # 2. 划分 90% 训练，10% 验证（用于最终确定阈值，但不是用于调参）
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS//2, pin_memory=PIN_MEMORY
    )
    print(f"   训练集: {len(train_ds)}, 验证集: {len(val_ds)} (仅用于确定阈值)")

    # 3. 计算动态 pos_weight（基于训练集）
    all_labels = []
    for idx in range(len(train_ds)):
        row = dataset.df.iloc[train_ds.indices[idx]]  # 获取原始索引
        labels = [row[f'label_{i}'] for i in range(9)]
        all_labels.append(labels)
    all_labels = np.array(all_labels)
    samples_per_class = all_labels.sum(axis=0).astype(np.float32)
    n_samples = len(train_ds)
    pos_weight = np.sqrt((n_samples - samples_per_class) / (samples_per_class + 1e-8))
    pos_weight = np.clip(pos_weight, 0.5, 10.0)
    print(f"   动态 pos_weight: {pos_weight.round(3).tolist()}")

    # 4. 构建模型
    model = MILModel(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        pretrained=True,
        pooling=MIL_POOLING,
        topk=MIL_TOPK,
        attention_dim=MIL_ATTENTION_DIM,
        independent_heads=USE_INDEPENDENT_HEADS
    ).to(DEVICE)

    # 加载医学预训练（如果指定）
    if PRETRAIN_WEIGHT_PATH and os.path.exists(PRETRAIN_WEIGHT_PATH):
        print(f"📥 加载医学预训练: {PRETRAIN_WEIGHT_PATH}")
        state_dict = torch.load(PRETRAIN_WEIGHT_PATH, map_location='cpu')
        if 'backbone' in state_dict:
            model.backbone.load_state_dict(state_dict['backbone'], strict=False)
        else:
            model.backbone.load_state_dict(state_dict, strict=False)
        print("✅ 加载完成")

    print(f"🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 优化器与损失
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR_3D, weight_decay=WEIGHT_DECAY_3D)
    
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

    best_val_f1 = 0.0
    best_thresholds = [0.5] * NUM_CLASSES
    patience_counter = 0

    # 6. 训练循环
    for epoch in range(1, EPOCHS_3D + 1):
        model.train()
        train_loss = 0
        all_train_preds, all_train_targets = [], []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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

        # 验证（用于监控和确定阈值）
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

        # 搜索最佳阈值
        if USE_PER_CLASS_THRESHOLD:
            best_thresholds, _ = search_best_thresholds(
                all_val_preds, all_val_targets,
                n_steps=THRESHOLD_SEARCH_STEPS
            )
        val_prec, val_rec, val_f1, _ = compute_metrics_with_thresholds(
            all_val_preds, all_val_targets, best_thresholds
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val F1={val_f1:.4f}, LR={current_lr:.6f}")

        # 调度器
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_reduce.step(val_loss)

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            final_path = os.path.join(WEIGHT_SAVE_DIR, "final_best_lesion.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_val_f1,
                'best_thresholds': best_thresholds,
            }, final_path)
            print(f"  ✅ 新最佳 F1={best_val_f1:.4f}, 已保存")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE_3D:
                print(f"  ⏳ 早停触发")
                break

    print("\n" + "=" * 70)
    print("🎉 最终模型训练完成！")
    print(f"   最佳验证 F1: {best_val_f1:.4f}")
    print(f"   最佳阈值: {[f'{th:.3f}' for th in best_thresholds]}")
    print(f"   模型保存至: {WEIGHT_SAVE_DIR}/final_best_lesion.pth")
    print("=" * 70)

if __name__ == "__main__":
    train_full()