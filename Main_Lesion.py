"""
脑部病灶多标签分类 - 主训练脚本
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Conf.Config import *
from Datasets.CQ500Dataset import CQ500SliceDataset, CQ500ClassificationDataset
from Model.LesionClassifier import LesionClassifier2D, LesionClassifier3D
from Loss.MultiLabelLoss import CombinedMultiLabelLoss
from Trainer.LesionClassifierTrainer import LesionClassifierTrainer
from Utils.TrainingLogger import TrainingLogger
from Conf.Config import TRAINING_LOG_DIR

def main():
    print("=" * 60)
    print("🧠 脑部病灶多标签分类训练")
    print("=" * 60)
    print(f"任务模式: {TASK_MODE}")
    print(f"设备: {DEVICE}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"标签: {LESION_LABELS}")
    print("=" * 60)

    # ==================== 数据加载 ====================
    print("\n📊 加载数据集...")

    if USE_3D_VOLUME:
        # 3D 体积分类
        dataset = CQ500ClassificationDataset(
            data_root=CQ500_DATA_ROOT,
            annotation_file=CQ500_ANNOTATION_FILE,
            transform=True,
            target_size=TARGET_SIZE_3D,
            use_3_slices=USE_THREE_VIEWS
        )
    else:
        # 2D 切片分类
        dataset = CQ500SliceDataset(
            data_root=CQ500_DATA_ROOT,
            annotation_file=CQ500_ANNOTATION_FILE,
            transform=True,
            target_size=TARGET_SIZE_2D
        )

    # 划分训练集和验证集
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ==================== 模型 ====================
    print("\n🏗️  构建模型...")

    if USE_3D_VOLUME:
        model = LesionClassifier3D(
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS
        )
    else:
        model = LesionClassifier2D(
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS,
            use_three_views=USE_THREE_VIEWS
        )

    # ==================== 损失函数 ====================
    print("\n⚖️  配置损失函数...")

    # 计算类别权重（处理不平衡）
    pos_weight = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)

    loss_fn = CombinedMultiLabelLoss(
        bce_weight=0.5,
        focal_weight=0.5,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        label_smoothing=0.1,
        pos_weight=pos_weight
    )

    # ==================== 优化器 ====================
    print("\n🔧 配置优化器...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    log_dir = os.path.join(BASE_DIR, "文档", "训练记录")
    logger = TrainingLogger(TRAINING_LOG_DIR)
    
    # ==================== 训练器 ====================
    weights_file = os.path.join(WEIGHT_SAVE_DIR, "best_lesion.pth")

    trainer = LesionClassifierTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=weights_file,
        scheduler=scheduler,
        patience=EARLY_STOP_PATIENCE,
        grad_clip=GRAD_CLIP,
        use_amp=USE_AMP,
        show_realtime_viz=True,
        logger=logger  # 需要在 Trainer 类中添加该参数
    )

    # ==================== 开始训练 ====================
    history = trainer.Train(train_loader, val_loader)

    print("\n🎉 全部训练完成！")
    print(f"模型保存于: {weights_file}")


if __name__ == "__main__":
    main()
