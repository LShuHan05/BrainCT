import torch


# ====================== 评价指标计算类（静态方法） ======================
class MetricCalculator:
    """
	分割评价指标计算器
	【优化】新增IoU指标
	"""

    @staticmethod
    def calculate_metrics(pred, target):
        """
		计算精确率、召回率、Dice、准确率、IoU
		【优化】返回5个指标（新增IoU）
		"""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        tp = intersection
        fp = pred.sum() - tp
        fn = target.sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        accuracy = (pred == target).sum() / torch.numel(pred)

        # 【新增】IoU（交并比）
        iou = (intersection + 1e-7) / (union - intersection + 1e-7)

        return precision.item(), recall.item(), dice.item(), accuracy.item(), iou.item()

# ====================== 评价指标计算类（静态方法） ======================
class MetricCalculator:
    """
    分割评价指标计算器
    【优化】新增IoU指标和自定义阈值支持
    """

    @staticmethod
    def calculate_metrics(pred, target, threshold=0.5):
        """
        计算精确率、召回率、Dice、准确率、IoU

        :param pred: 模型输出（logits）
        :param target: 真实标签
        :param threshold: 二值化阈值，默认0.5
        """
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        tp = intersection
        fp = pred.sum() - tp
        fn = target.sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        accuracy = (pred == target).sum() / torch.numel(pred)

        # 【新增】IoU（交并比）
        iou = (intersection + 1e-7) / (union - intersection + 1e-7)

        return precision.item(), recall.item(), dice.item(), accuracy.item(), iou.item()

    @staticmethod
    def calculate_metrics_with_threshold(pred, target, threshold=0.5):
        """别名方法，保持兼容性"""
        return MetricCalculator.calculate_metrics(pred, target, threshold)
