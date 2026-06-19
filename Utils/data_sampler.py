"""
CQ500 数据集分层抽样脚本
功能：
1. 读取 reads.csv 标注文件
2. 按病灶类型分层抽样
3. 保证阳性病例占比60%
4. 复制选中的 DICOM 文件到训练目录
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class CQ500DataSampler:
    """CQ500 数据集分层抽样器"""

    def __init__(
            self,
            data_root: str,
            annotation_file: str,
            output_dir: str,
            positive_ratio: float = 0.6,
            total_samples: int = 100
    ):
        """
        Args:
            data_root: CQ500 原始数据根目录
            annotation_file: reads.csv 或 prediction_probabilities.csv 路径
            output_dir: 抽样后数据输出目录
            positive_ratio: 阳性样本比例（默认60%）
            total_samples: 总抽样数量
        """
        self.data_root = Path(data_root)
        self.annotation_file = Path(annotation_file)
        self.output_dir = Path(output_dir)
        self.positive_ratio = positive_ratio
        self.total_samples = total_samples

        # 计算正负样本数量
        self.n_positive = int(total_samples * positive_ratio)
        self.n_negative = total_samples - self.n_positive

        # 创建输出目录结构
        self._create_output_dirs()

        # 加载标注数据
        self.annotations = None
        self.load_annotations()

    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir / "CT",
            self.output_dir / "MASK",
            self.output_dir / "splits",
            self.output_dir / "metadata"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录已创建: {self.output_dir}")

    def load_annotations(self):
        """加载标注文件"""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")

        # 读取 CSV
        ext = self.annotation_file.suffix.lower()
        if ext == '.csv':
            self.annotations = pd.read_csv(self.annotation_file)
        else:
            raise ValueError(f"不支持的标注文件格式: {ext}")

        print(f"📊 加载标注数据: {len(self.annotations)} 条记录")
        print(f"   列名: {list(self.annotations.columns)[:10]}...")

    def identify_positive_cases(self) -> Tuple[List[str], List[str]]:
        """
        识别阳性和阴性病例

        Returns:
            (positive_ids, negative_ids): 阳性ID列表和阴性ID列表
        """
        if self.annotations is None:
            raise ValueError("请先加载标注数据")

        # 定义出血相关字段（根据 reads.csv 格式）
        bleed_columns = [
            'R1:ICH', 'R1:IPH', 'R1:IVH', 'R1:SDH', 'R1:EDH', 'R1:SAH',
            'R2:ICH', 'R2:IPH', 'R2:IVH', 'R2:SDH', 'R2:EDH', 'R2:SAH',
            'R3:ICH', 'R3:IPH', 'R3:IVH', 'R3:SDH', 'R3:EDH', 'R3:SAH',
            'Fracture', 'CalvarialFracture', 'OtherFracture'
        ]

        # 检查哪些列存在
        available_cols = [col for col in bleed_columns if col in self.annotations.columns]

        if not available_cols:
            # 尝试使用 prediction_probabilities.csv 格式
            prob_columns = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH', 'CalvarialFracture']
            available_cols = [col for col in prob_columns if col in self.annotations.columns]

            if not available_cols:
                raise ValueError("未找到任何病灶标注列")

        # 判断阳性：至少一位医生标注为阳性 或 概率 > 0.5
        if 'R1:' in available_cols[0]:
            # reads.csv 格式：三位医生中至少一位标注为1
            positive_mask = self.annotations[available_cols].max(axis=1) > 0
        else:
            # prediction_probabilities.csv 格式：概率 > 0.5
            positive_mask = self.annotations[available_cols].max(axis=1) > 0.5

        positive_ids = self.annotations.loc[positive_mask, 'name'].tolist()
        negative_ids = self.annotations.loc[~positive_mask, 'name'].tolist()

        print(f"🔍 数据统计:")
        print(f"   阳性病例: {len(positive_ids)}")
        print(f"   阴性病例: {len(negative_ids)}")

        return positive_ids, negative_ids

    def stratified_sample(
            self,
            positive_ids: List[str],
            negative_ids: List[str]
    ) -> List[str]:
        """
        分层抽样

        Args:
            positive_ids: 阳性病例ID列表
            negative_ids: 阴性病例ID列表

        Returns:
            sampled_ids: 抽样后的病例ID列表
        """
        # 随机抽样
        np.random.seed(42)  # 固定随机种子

        if len(positive_ids) < self.n_positive:
            print(f"⚠️  警告: 阳性样本不足 ({len(positive_ids)} < {self.n_positive})")
            sampled_positive = positive_ids
        else:
            sampled_positive = np.random.choice(
                positive_ids,
                size=self.n_positive,
                replace=False
            ).tolist()

        if len(negative_ids) < self.n_negative:
            print(f"⚠️  警告: 阴性样本不足 ({len(negative_ids)} < {self.n_negative})")
            sampled_negative = negative_ids
        else:
            sampled_negative = np.random.choice(
                negative_ids,
                size=self.n_negative,
                replace=False
            ).tolist()

        sampled_ids = sampled_positive + sampled_negative
        np.random.shuffle(sampled_ids)

        print(f"✅ 抽样完成: {len(sampled_ids)} 个病例")
        print(f"   阳性: {len(sampled_positive)}, 阴性: {len(sampled_negative)}")

        return sampled_ids

    def find_dicom_files(self, case_id: str) -> List[Path]:
        """
        查找指定病例的 DICOM 文件

        Args:
            case_id: 病例ID (如 CQ500-CT-208 或 CQ500CT208)

        Returns:
            dicom_files: DICOM 文件路径列表
        """
        # ⚠️ 【修复】处理不同的ID格式
        # 标注文件中: CQ500-CT-208
        # 文件夹名称: CQ500CT208 CQ500CT208

        # 标准化 case_id：移除连字符和空格
        normalized_id = case_id.replace('-', '').replace(' ', '')
        # 例如: CQ500-CT-208 -> CQ500CT208

        print(f"\n   [DEBUG] 原始ID: {case_id}, 标准化: {normalized_id}")

        # 在 data_root 下搜索该病例文件夹
        # 使用更宽松的匹配：只要包含标准化ID即可
        all_folders = list(self.data_root.glob("*"))

        matching_folders = []
        for folder in all_folders:
            if not folder.is_dir():
                continue

            # 标准化文件夹名
            folder_name_normalized = folder.name.replace('-', '').replace(' ', '')

            # 检查是否包含目标ID
            if normalized_id in folder_name_normalized or folder_name_normalized in normalized_id:
                matching_folders.append(folder)
                print(f"   [DEBUG] 匹配到文件夹: {folder.name}")

        if not matching_folders:
            print(f"   [DEBUG] 未找到匹配的文件夹")
            # 列出前10个文件夹供调试
            sample_folders = list(self.data_root.glob("*"))[:10]
            print(f"   [DEBUG] 示例文件夹: {[f.name for f in sample_folders]}")
            return []

        dicom_files = []
        for folder in matching_folders:
            # 优先选择 CT Plain 序列
            plain_folders = [
                folder / "Unknown Study" / "CT Plain",
                folder / "Unknown Study" / "CT Plain 3mm",
                folder / "Unknown Study" / "CT PLAIN THIN"
            ]

            target_folder = None
            for pf in plain_folders:
                if pf.exists():
                    target_folder = pf
                    print(f"   [DEBUG] 找到平扫序列: {pf}")
                    break

            if target_folder is None:
                # 如果没有平扫，使用第一个存在的序列
                study_folder = folder / "Unknown Study"
                if study_folder.exists():
                    subfolders = [f for f in study_folder.iterdir() if f.is_dir()]
                    if subfolders:
                        target_folder = subfolders[0]
                        print(f"   [DEBUG] 使用替代序列: {target_folder}")

            if target_folder:
                # 获取所有 .dcm 文件
                files = list(target_folder.glob("*.dcm"))
                dicom_files.extend(files)
                print(f"   [DEBUG] 找到 {len(files)} 个DICOM文件")

        return dicom_files

    def copy_selected_data(self, sampled_ids: List[str]):
        """
        复制选中的数据到输出目录

        Args:
            sampled_ids: 选中的病例ID列表
        """
        ct_output = self.output_dir / "CT"
        mask_output = self.output_dir / "MASK"

        success_count = 0
        fail_count = 0

        print(f"\n📦 开始复制数据...")

        for i, case_id in enumerate(sampled_ids, 1):
            print(f"   [{i}/{len(sampled_ids)}] 处理 {case_id}...", end=" ")

            dicom_files = self.find_dicom_files(case_id)

            if not dicom_files:
                print("❌ 未找到DICOM文件")
                fail_count += 1
                continue

            # 复制 DICOM 文件
            case_success = 0
            for dcm_file in dicom_files:
                dest_file = ct_output / dcm_file.name

                # 避免文件名冲突，添加病例前缀
                new_name = f"{case_id}_{dcm_file.name}"
                dest_file = ct_output / new_name

                try:
                    shutil.copy2(dcm_file, dest_file)
                    case_success += 1
                except Exception as e:
                    print(f"⚠️  复制失败: {e}")

            if case_success > 0:
                print(f"✅ 复制 {case_success} 个文件")
                success_count += 1
            else:
                print("❌ 复制失败")
                fail_count += 1

        print(f"\n📊 复制统计:")
        print(f"   成功: {success_count} 个病例")
        print(f"   失败: {fail_count} 个病例")
        print(f"   CT文件总数: {len(list(ct_output.glob('*.dcm')))}")

    def save_metadata(self, sampled_ids: List[str]):
        """保存抽样元数据"""
        metadata_dir = self.output_dir / "metadata"

        # 保存抽样列表
        sample_df = pd.DataFrame({
            'case_id': sampled_ids,
            'is_positive': [
                1 if self._is_positive(cid) else 0
                for cid in sampled_ids
            ]
        })
        sample_df.to_csv(metadata_dir / "sample_list.csv", index=False)

        # 保存抽样配置
        config = {
            'total_samples': self.total_samples,
            'positive_ratio': self.positive_ratio,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
            'random_seed': 42
        }
        pd.DataFrame([config]).to_csv(metadata_dir / "sampling_config.csv", index=False)

        print(f"✅ 元数据已保存到: {metadata_dir}")

    def _is_positive(self, case_id: str) -> bool:
        """判断病例是否为阳性"""
        if self.annotations is None:
            return False

        row = self.annotations[self.annotations['name'] == case_id]
        if row.empty:
            return False

        # 简化判断：检查是否有任何出血标注
        bleed_cols = [col for col in row.columns if 'ICH' in col or 'IPH' in col or 'SDH' in col]
        if bleed_cols:
            return row[bleed_cols].values.max() > 0

        return False

    def run(self):
        """执行完整的抽样流程"""
        print("=" * 60)
        print("CQ500 数据集分层抽样")
        print("=" * 60)

        # 1. 识别正负样本
        positive_ids, negative_ids = self.identify_positive_cases()

        # 2. 分层抽样
        sampled_ids = self.stratified_sample(positive_ids, negative_ids)

        # 3. 复制数据
        self.copy_selected_data(sampled_ids)

        # 4. 保存元数据
        self.save_metadata(sampled_ids)

        print("\n" + "=" * 60)
        print("✅ 抽样完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='CQ500 数据集分层抽样')
    parser.add_argument('--data_root', type=str,
                        default='D:/Code/Python/Code/neuSoft/BrainCT/data/CQ500_orig',
                        help='CQ500 原始数据根目录')
    parser.add_argument('--annotation', type=str,
                        default='D:/Code/Python/Code/neuSoft/BrainCT/md/reads.csv',
                        help='标注文件路径 (reads.csv 或 prediction_probabilities.csv)')
    parser.add_argument('--output', type=str,
                        default='D:/Code/Python/Code/neuSoft/BrainCT/datasets_filtered',
                        help='输出目录')
    parser.add_argument('--samples', type=int, default=100,
                        help='总抽样数量')
    parser.add_argument('--positive_ratio', type=float, default=0.6,
                        help='阳性样本比例 (0-1)')

    args = parser.parse_args()

    sampler = CQ500DataSampler(
        data_root=args.data_root,
        annotation_file=args.annotation,
        output_dir=args.output,
        positive_ratio=args.positive_ratio,
        total_samples=args.samples
    )

    sampler.run()


if __name__ == "__main__":
    main()
