import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def find_optimal_threshold(predictions, targets):
    """
    在验证集上找到最优阈值

    :param predictions: 模型输出的概率值（0-1）
    :param targets: 真实标签
    :return: 最优阈值
    """
    precisions, recalls, thresholds = precision_recall_curve(targets.flatten(), predictions.flatten())

    # 计算每个阈值的F1分数
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0)

    # 找到F1最大的阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"✅ 最优阈值: {best_threshold:.3f}")
    print(f"   对应Precision: {precisions[best_idx]:.3f}")
    print(f"   对应Recall: {recalls[best_idx]:.3f}")
    print(f"   对应F1: {f1_scores[best_idx]:.3f}")

    return best_threshold

# 在训练器中使用这个函数
