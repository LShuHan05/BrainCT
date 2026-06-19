"""
统一的数据预处理工具
- 训练和推理共享相同的归一化逻辑
- 避免代码重复和不一致
"""

import numpy as np
import torch


def z_score_normalize(image):
    """
    Z-score归一化（单通道）

    Args:
        image: numpy array 或 torch tensor

    Returns:
        normalized image (相同类型)
    """
    if isinstance(image, np.ndarray):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-7)

    elif isinstance(image, torch.Tensor):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-7)

    else:
        raise TypeError(f"不支持的类型: {type(image)}")


def normalize_ct_slice(ct_arr):
    """
    CT切片标准化预处理流程

    Args:
        ct_arr: 2D numpy array (CT像素值)

    Returns:
        normalized 2D numpy array
    """
    # Z-score归一化
    normalized = z_score_normalize(ct_arr)
    return normalized.astype(np.float32)


def mask_to_binary(mask_arr, threshold=0):
    """
    将mask转换为二值格式

    Args:
        mask_arr: numpy array
        threshold: 二值化阈值

    Returns:
        binary mask (float32, 0或1)
    """
    return (mask_arr > threshold).astype(np.float32)
