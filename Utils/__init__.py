import sys
import os

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Utils.Preprocessing import normalize_ct_slice
from Utils.EvalMetrics import MetricCalculator

# ⚠️ 【新增】如果需要 compute_dice 等函数，可以创建包装函数
def compute_dice(pred, target, threshold=0.5):
    """计算 Dice 系数"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.item()

def compute_iou(pred, target, threshold=0.5):
    """计算 IoU"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()

__all__ = [
    'normalize_ct_slice',
    'MetricCalculator',
    'compute_dice',
    'compute_iou'
]
