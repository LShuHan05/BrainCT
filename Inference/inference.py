import torch
import numpy as np
import os
from scipy.ndimage import binary_fill_holes, label
from torch.utils.data import DataLoader
from ..Datasets.Datasets import CTSliceDataset
from ..Model.AttentionUNet2D import UNet2D
from ..Conf.Config import DEVICE, CT_PATH, MASK_PATH, WEIGHT_SAVE_DIR

def tta_predict(model, x):
    """测试时增强：水平翻转 + 原始，取平均"""
    model.eval()
    with torch.no_grad():
        pred_orig = torch.sigmoid(model(x))
        x_flip = torch.flip(x, dims=[-1])
        pred_flip = torch.sigmoid(model(x_flip))
        pred_flip = torch.flip(pred_flip, dims=[-1])
        pred = (pred_orig + pred_flip) / 2
    return pred

def postprocess_mask(mask, min_area=50):
    """后处理：小孔填充 + 移除小连通域"""
    mask = binary_fill_holes(mask)
    labeled, num = label(mask)
    for i in range(1, num+1):
        if (labeled == i).sum() < min_area:
            mask[labeled == i] = 0
    return mask

def ensemble_predict(models, x):
    """多模型集成：平均概率"""
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = tta_predict(model, x)   # 每个模型自带 TTA
            preds.append(pred)
    return torch.stack(preds).mean(dim=0)

def inference_on_dataset(model_paths=None, use_tta=True, use_ensemble=True, use_postprocess=True, min_area=50):
    """
    对测试集（或验证集）进行推理并计算 Dice
    :param model_paths: 模型权重路径列表，若 None 则使用最佳模型路径
    :param use_tta: 是否使用 TTA
    :param use_ensemble: 是否使用集成（需提供多个模型路径）
    :param use_postprocess: 是否使用后处理
    """
    # 加载测试数据集
    dataset = CTSliceDataset(CT_PATH, MASK_PATH)
    dataset.is_train = False
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 加载模型
    if model_paths is None:
        model_paths = [os.path.join(WEIGHT_SAVE_DIR, "best.pth")]

    models = []
    for path in model_paths:
        model = UNet2D(use_auxiliary=False).to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    total_dice = 0
    total_iou = 0
    total_recall = 0

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img = img.to(DEVICE)
            mask_np = mask.squeeze().cpu().numpy()

            if use_ensemble and len(models) > 1:
                pred_prob = ensemble_predict(models, img)
            else:
                pred_prob = tta_predict(models[0], img) if use_tta else torch.sigmoid(models[0](img))

            pred_bin = (pred_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            if use_postprocess:
                pred_bin = postprocess_mask(pred_bin, min_area)

            # 计算指标
            intersection = (pred_bin * mask_np).sum()
            union = pred_bin.sum() + mask_np.sum()
            dice = (2. * intersection + 1e-7) / (union + 1e-7)
            iou = (intersection + 1e-7) / (union - intersection + 1e-7)
            tp = intersection
            fn = mask_np.sum() - tp
            recall = tp / (tp + fn + 1e-7)

            total_dice += dice
            total_iou += iou
            total_recall += recall

            if i % 20 == 0:
                print(f"Sample {i}: Dice={dice:.4f}, IoU={iou:.4f}, Recall={recall:.4f}")

    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    avg_recall = total_recall / len(loader)
    print(f"\n🎯 最终结果: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, Recall={avg_recall:.4f}")
    return avg_dice

if __name__ == "__main__":
    # 示例：使用集成 3 个最佳模型
    model_list = [
        os.path.join(WEIGHT_SAVE_DIR, "best_epoch55.pth"),
        os.path.join(WEIGHT_SAVE_DIR, "best_epoch58.pth"),
        os.path.join(WEIGHT_SAVE_DIR, "best.pth")   # 最终模型
    ]
    # 如果只有单个模型，传入列表只含一个路径，并设置 use_ensemble=False
    inference_on_dataset(model_paths=model_list, use_tta=True, use_ensemble=True, use_postprocess=True, min_area=50)