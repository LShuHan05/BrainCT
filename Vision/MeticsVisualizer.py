import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import confusion_matrix

plt.rcParams['backend'] = 'Agg'


class Visualizer:
    def __init__(self, save_dir):
        """
        初始化可视化工具
        :param save_dir: 图片保存路径
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_all_curves(self, history):
        """绘制所有训练曲线"""
        # Loss 曲线
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

        # 准确率曲线
        plt.figure(figsize=(8, 5))
        plt.plot(history['acc'], color='g', label='Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'acc_curve.png'), dpi=300)
        plt.close()

        # 精确率 & 召回率
        plt.figure(figsize=(8, 5))
        plt.plot(history['precision'], label='Precision')
        plt.plot(history['recall'], label='Recall')
        plt.title('Precision & Recall')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'pr_curve.png'), dpi=300)
        plt.close()

        # Dice & F1
        plt.figure(figsize=(8, 5))
        plt.plot(history['dice'], label='Dice')
        f1 = [2 * p * r / (p + r + 1e-7) for p, r in zip(history['precision'], history['recall'])]
        plt.plot(f1, label='F1')
        plt.title('Dice & F1')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'dice_f1_curve.png'), dpi=300)
        plt.close()

        # 【新增】IoU 曲线
        if 'iou' in history and any(history['iou']):
            plt.figure(figsize=(8, 5))
            plt.plot(history['iou'], color='orange', label='IoU')
            plt.title('Intersection over Union (IoU)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'iou_curve.png'), dpi=300)
            plt.close()

    def plot_confusion(self, model, loader, device):
        """绘制混淆矩阵"""
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for img, gt in loader:
                img = img.to(device)
                pred = model(img)
                # 【优化】处理多输出情况
                if isinstance(pred, dict):
                    pred = pred['main']
                pred = torch.sigmoid(pred) > 0.5
                all_pred.extend(pred.cpu().numpy().flatten().astype(int))
                all_gt.extend(gt.numpy().flatten().astype(int))

        cm = confusion_matrix(all_gt, all_pred, normalize='true')
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]:.2f}', ha="center", va="center")

        plt.savefig(os.path.join(self.save_dir, 'confusion.png'), dpi=300)
        plt.close()

    def plot_sample_pred(self, model, dataset, device):
        """绘制单张预测结果图"""
        model.eval()
        ct, mask = dataset[0]
        with torch.no_grad():
            pred = model(ct.unsqueeze(0).to(device))
            # 【优化】处理多输出情况
            if isinstance(pred, dict):
                pred = pred['main']
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(ct.squeeze(), cmap='gray')
        plt.title("CT")

        plt.subplot(132)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title("GT Mask")

        plt.subplot(133)
        plt.imshow(pred > 0.5, cmap='gray')
        plt.title("Prediction")

        plt.savefig(os.path.join(self.save_dir, 'pred_sample.png'), dpi=300)
        plt.close()
