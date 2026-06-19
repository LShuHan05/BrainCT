"""
多标签分类损失函数
支持：
1. Binary Cross Entropy with class weights
2. Focal Loss for imbalanced classes
3. Asymmetric Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    多标签 Focal Loss
    解决类别不平衡问题
    """

    def __init__(
            self,
            alpha: float = 0.75,
            gamma: float = 2.0,
            reduction: str = 'mean',
            pos_weight: torch.Tensor = None
    ):
        """
        Args:
            alpha: 平衡正负样本权重
            gamma: 调节难易样本权重
            reduction: 'mean', 'sum', or 'none'
            pos_weight: 各类别正样本权重 (num_classes,)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (B, num_classes)
            targets: binary labels (B, num_classes)

        Returns:
            loss: scalar loss
        """
        # Sigmoid activation
        probs = torch.sigmoid(inputs)

        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Focal weighting
        probs_gt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - probs_gt) ** self.gamma

        # Alpha balancing
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Final loss
        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    非对称损失函数
    专门处理多标签分类中的类别不平衡
    """

    def __init__(
            self,
            gamma_neg: float = 4.0,
            gamma_pos: float = 1.0,
            clip: float = 0.05,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (B, num_classes)
            targets: binary labels (B, num_classes)
        """
        probs = torch.sigmoid(inputs)

        # Basic BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Asymmetric focusing
        probs_gt = probs * targets + (1 - probs) * (1 - targets)

        # Negative samples get more focusing
        weight_neg = (1 - probs_gt) ** self.gamma_neg
        weight_pos = (1 - probs_gt) ** self.gamma_pos

        weights = targets * weight_pos + (1 - targets) * weight_neg

        # Clipping to prevent instability
        weights = torch.clamp(weights, min=self.clip)

        loss = weights * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedMultiLabelLoss(nn.Module):
    """
    组合损失函数
    BCE + Focal + Label Smoothing
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            focal_weight: float = 0.5,
            alpha: float = 0.75,
            gamma: float = 2.0,
            label_smoothing: float = 0.1,
            pos_weight: torch.Tensor = None
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean'
        )

        self.focal_loss = MultiLabelFocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean',
            pos_weight=pos_weight
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (B, num_classes)
            targets: binary labels (B, num_classes)
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss
        bce = self.bce_loss(inputs, targets)

        # Focal loss
        focal = self.focal_loss(inputs, targets)

        # Combined
        loss = self.bce_weight * bce + self.focal_weight * focal

        return loss
