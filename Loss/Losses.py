import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# ====================== Focal Loss ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.55, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()

# ====================== 边界损失 ======================
class BoundaryLoss(nn.Module):
    def __init__(self, sigma=2):
        super().__init__()
        self.sigma = sigma

    def _distance_transform(self, mask):
        mask_float = mask.float()
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = F.conv2d(mask_float, kernel, padding=1)
        eroded = F.conv2d(mask_float, -kernel, padding=1)
        boundary = (dilated > 0.5) & (eroded < 0.5)
        weight = boundary.float()
        weight = gaussian_blur(weight, kernel_size=5, sigma=self.sigma)
        return weight

    def forward(self, logits, target):
        with torch.no_grad():
            weight = self._distance_transform(target)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = (loss * weight).sum() / (weight.sum() + 1e-7)
        return loss

# ====================== 【新增】Lovász Loss（直接优化 IoU） ======================
# 参考 https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:p-1]
    return jaccard

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W) 二值
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        signs = 2. * targets - 1.
        errors = 1. - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = targets[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

# ====================== 混合损失（Dice + Focal + Boundary + Lovász） ======================
class DiceFocalLoss(nn.Module):
    def __init__(self, use_focal=True, dice_weight=0.5, ce_weight=0.3,
                 focal_alpha=0.55, focal_gamma=1.5,
                 use_boundary=False, boundary_weight=0.1, boundary_sigma=2,
                 use_lovasz=False, lovasz_weight=0.2):
        super().__init__()
        self.use_focal = use_focal
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.use_lovasz = use_lovasz
        self.lovasz_weight = lovasz_weight

        if use_focal:
            self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            pos_weight = torch.tensor([10.0])
            self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if use_boundary:
            self.boundary_loss = BoundaryLoss(sigma=boundary_sigma)

        if use_lovasz:
            self.lovasz = LovaszHingeLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        # 1. Dice Loss
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + 1e-7) / (pred_sigmoid.sum() + target.sum() + 1e-7)

        # 2. Focal / CE Loss
        if self.use_focal:
            ce_loss = self.focal(pred, target)
        else:
            ce_loss = self.ce(pred, target)

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        # 3. Boundary Loss
        if self.use_boundary:
            boundary_loss = self.boundary_loss(pred, target)
            total_loss += self.boundary_weight * boundary_loss

        # 4. Lovász Loss
        if self.use_lovasz:
            lovasz_loss = self.lovasz(pred, target)
            total_loss += self.lovasz_weight * lovasz_loss

        return total_loss

# ====================== 深度监督损失包装器 ======================
class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss_fn, aux_weights=None):
        super().__init__()
        self.base_loss = base_loss_fn
        self.aux_weights = aux_weights or {'aux_d4': 0.3, 'aux_d3': 0.2, 'aux_fusion': 0.2}

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            main_loss = self.base_loss(outputs['main'], targets)
            total_aux_loss = 0
            for aux_name, weight in self.aux_weights.items():
                if aux_name in outputs:
                    aux_pred = nn.functional.interpolate(
                        outputs[aux_name],
                        size=targets.shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                    aux_loss = self.base_loss(aux_pred, targets)
                    total_aux_loss += weight * aux_loss
            return main_loss + total_aux_loss
        else:
            return self.base_loss(outputs, targets)

DiceCELoss = DiceFocalLoss