# 文件路径：BrainCT/Utils/TrainingLogger.py
import os
import csv
import pandas as pd
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self.txt_path = os.path.join(log_dir, f"{experiment_name}.txt")
        self.headers = ['epoch', 'train_loss', 'val_loss', 'macro_f1', 'sample_f1', 'lr', 'best_sample_f1']
        # 初始化CSV（若不存在）
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_epoch(self, epoch, train_loss, val_loss, macro_f1, sample_f1, lr, best_sample_f1):
        # 写入CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, macro_f1, sample_f1, lr, best_sample_f1])
        # 追加到文本文件（便于阅读）
        with open(self.txt_path, 'a') as f:
            f.write(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
                    f"MacroF1={macro_f1:.4f}, SampleF1={sample_f1:.4f}, LR={lr:.6f}, BestF1={best_sample_f1:.4f}\n")

    def save_final_summary(self, best_epoch, best_f1, total_time):
        with open(self.txt_path, 'a') as f:
            f.write("\n========== Final Summary ==========\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Sample F1: {best_f1:.4f}\n")
            f.write(f"Total Time: {total_time:.2f} minutes\n")