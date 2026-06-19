import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from .Conf.Config import *
from .Datasets.Datasets import CTSliceDataset
from .Model.AttentionUNet2D import UNet2D
from .Loss.Losses import DiceFocalLoss, DeepSupervisionLoss
from .Trainer.UNet2DTrainer import Trainer
from .Vision.MeticsVisualizer import Visualizer

def main():
    print("=" * 50)
    print("2D UNet 脑部CT伪影分割训练（增强版：SE + Lovász + 弹性变形）")
    print(f"设备：{DEVICE}")
    print("=" * 50)

    # 数据加载
    dataset = CTSliceDataset(CT_PATH, MASK_PATH)
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_ds.dataset.is_train = True
    val_ds.dataset.is_train = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 模型（传入 SE 参数）
    model = UNet2D(
        use_auxiliary=USE_AUXILIARY,
        use_se=USE_SE_ATTENTION,
        se_reduction=SE_REDUCTION
    ).to(DEVICE)

    # 损失函数（包含 Lovász 和边界损失）
    base_loss = DiceFocalLoss(
        use_focal=USE_FOCAL_LOSS,
        dice_weight=DICE_WEIGHT,
        ce_weight=CE_WEIGHT,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        use_boundary=USE_BOUNDARY_LOSS,
        boundary_weight=BOUNDARY_WEIGHT,
        boundary_sigma=BOUNDARY_SIGMA,
        use_lovasz=USE_LOVASZ_LOSS,
        lovasz_weight=LOVASZ_WEIGHT
    )

    if USE_AUXILIARY:
        loss_fn = DeepSupervisionLoss(base_loss)
        print("✅ 启用深度监督训练（含多尺度融合监督）")
    else:
        loss_fn = base_loss

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    weights_file = os.path.join(WEIGHT_SAVE_DIR, WEIGHT_FILE)

    trainer = Trainer(
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
        show_realtime_viz=True
    )
    history = trainer.Train(train_loader, val_loader)

    print("\n📊 开始生成可视化图表...")
    vision = Visualizer(save_dir=SAVE_DIR)
    vision.plot_all_curves(history)
    vision.plot_confusion(model, val_loader, DEVICE)
    vision.plot_sample_pred(model, dataset, DEVICE)

    print("\n🎉 全部训练 + 可视化完成！")

if __name__ == "__main__":
    main()