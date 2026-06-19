import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import os


class RealTimeVisualizer:
    """
    实时训练可视化工具
    - 在终端中显示动态更新的训练曲线
    - 支持多指标同时展示
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_metrics(self, history, epoch, save_path=None):
        """
        绘制当前所有指标的动态图

        :param history: 训练历史记录字典
        :param epoch: 当前epoch数
        :param save_path: 保存路径
        :return: PIL Image对象
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')

        epochs_range = range(1, epoch + 1)

        # 1. Loss 曲线
        ax = axes[0, 0]
        if history['train_loss']:
            ax.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
        if history['val_loss']:
            ax.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss', markersize=4)
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Dice & F1 曲线
        ax = axes[0, 1]
        if history['dice']:
            ax.plot(epochs_range, history['dice'], 'g-o', label='Dice', markersize=4)
        if history['precision'] and history['recall']:
            f1 = [2 * p * r / (p + r + 1e-7) for p, r in zip(history['precision'], history['recall'])]
            ax.plot(epochs_range, f1, 'm-o', label='F1', markersize=4)
        ax.set_title('Dice & F1 Score')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Precision & Recall 曲线
        ax = axes[0, 2]
        if history['precision']:
            ax.plot(epochs_range, history['precision'], 'b-o', label='Precision', markersize=4)
        if history['recall']:
            ax.plot(epochs_range, history['recall'], 'r-o', label='Recall', markersize=4)
        ax.set_title('Precision & Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Accuracy 曲线
        ax = axes[1, 0]
        if history['acc']:
            ax.plot(epochs_range, history['acc'], 'c-o', label='Accuracy', markersize=4)
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. IoU 曲线
        ax = axes[1, 1]
        if history.get('iou') and any(history['iou']):
            ax.plot(epochs_range, history['iou'], 'orange', marker='o', label='IoU', markersize=4)
            ax.set_title('Intersection over Union (IoU)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No IoU Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('IoU')

        # 6. 综合雷达图（最新epoch）
        ax = axes[1, 2]
        if history['dice'] and history['precision'] and history['recall'] and history['acc']:
            latest_dice = history['dice'][-1]
            latest_pre = history['precision'][-1]
            latest_rec = history['recall'][-1]
            latest_acc = history['acc'][-1]
            latest_iou = history.get('iou', [0])[-1] if history.get('iou') else 0

            categories = ['Dice', 'Precision', 'Recall', 'Accuracy', 'IoU']
            values = [latest_dice, latest_pre, latest_rec, latest_acc, latest_iou]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            ax = plt.subplot(2, 3, 6, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2, color='red')
            ax.fill(angles, values, alpha=0.25, color='red')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(f'Latest Metrics (Epoch {epoch})', pad=20, fontweight='bold')
            ax.grid(True)

        plt.tight_layout()

        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)

        # 同时保存到文件
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return image

    def print_epoch_summary(self, epoch, train_loss, val_loss, metrics, best_dice):
        """
        打印美观的epoch总结

        :param epoch: 当前epoch
        :param train_loss: 训练损失
        :param val_loss: 验证损失
        :param metrics: 指标字典 {'acc', 'precision', 'recall', 'dice', 'iou'}
        :param best_dice: 最佳Dice分数
        """
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # 创建指标表格
        table = Table(title=f"📊 Epoch {epoch} Training Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="green", width=12)
        table.add_column("Status", style="yellow", width=10)

        # 添加各项指标
        table.add_row("Train Loss", f"{train_loss:.4f}", "✅")
        table.add_row("Val Loss", f"{val_loss:.4f}", "✅")
        table.add_row("Accuracy", f"{metrics['acc']:.4f}", "📈")
        table.add_row("Precision", f"{metrics['precision']:.4f}", "🎯")
        table.add_row("Recall", f"{metrics['recall']:.4f}", "🔍")
        table.add_row("Dice", f"{metrics['dice']:.4f}", "⭐")
        table.add_row("IoU", f"{metrics['iou']:.4f}", "📐")

        # 最佳Dice高亮
        if metrics['dice'] >= best_dice:
            best_row_text = Text(f"{best_dice:.4f} (NEW!)", style="bold green")
            status_text = Text("🏆", style="bold red")
        else:
            best_row_text = Text(f"{best_dice:.4f}", style="dim")
            status_text = Text("-", style="dim")

        table.add_row("Best Dice", best_row_text, status_text)

        console.print(table)
        console.print()
