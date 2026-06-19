"""
CQ500 多标签分类数据集加载器
支持：
1. 从 DICOM 文件加载 CT 图像
2. 读取多标签标注（9种病灶类型）
3. 数据增强（3D旋转、翻转、噪声等）
4. 返回 (volume, labels) 对
"""
"""
CQ500 多标签分类数据集加载器
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from scipy.ndimage import rotate, zoom
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Conf.Config import USE_THREE_VIEWS

class CQ500ClassificationDataset(Dataset):
    """CQ500 多标签分类数据集"""

    # 病灶类型定义
    LABEL_COLUMNS = [
        'ICH',  # 脑内出血
        'IPH',  # 脑实质出血
        'IVH',  # 脑室内出血
        'SDH',  # 硬膜下血肿
        'EDH',  # 硬膜外血肿
        'SAH',  # 蛛网膜下腔出血
        'Fracture',  # 颅骨骨折
        'MassEffect',  # 占位效应
        'MidlineShift'  # 中线移位
    ]

    def __init__(
            self,
            data_root: str,
            annotation_file: str,
            transform: bool = True,
            target_size: tuple = (128, 128, 64),  # (H, W, D)
            use_3_slices: bool = False  # 是否使用3个中心切片而非全体积
    ):
        """
        Args:
            data_root: 数据根目录（包含 CT 文件夹）
            annotation_file: 标注文件路径
            transform: 是否应用数据增强
            target_size: 目标体积大小
            use_3_slices: 如果True，只提取3个中心切片（轴位、冠状、矢状）
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size
        self.use_3_slices = use_3_slices

        # 加载标注
        self.annotations = pd.read_csv(annotation_file)

        # 构建病例ID到标注的映射
        self.case_labels = {}
        self.case_files = {}

        # 扫描数据目录
        self._scan_data()

        # 病例列表
        self.case_ids = list(self.case_files.keys())

        print(f"✅ 数据集加载完成: {len(self.case_ids)} 个病例")
        print(f"   目标尺寸: {target_size}")
        print(f"   标签维度: {len(self.LABEL_COLUMNS)}")

    def _scan_data(self):
        """扫描数据目录，建立病例ID到文件和标签的映射"""
        ct_dir = self.data_root / "CT"

        if not ct_dir.exists():
            raise FileNotFoundError(f"CT目录不存在: {ct_dir}")

        # 按病例ID分组文件
        case_groups = {}
        for dcm_file in ct_dir.glob("*.dcm"):
            # 从文件名提取病例ID (格式: CQ500CT0_xxx.dcm)
            case_id = '_'.join(dcm_file.stem.split('_')[:-1]) if '_' in dcm_file.stem else dcm_file.stem

            if case_id not in case_groups:
                case_groups[case_id] = []
            case_groups[case_id].append(dcm_file)

        # 匹配标注
        for case_id, files in case_groups.items():
            # 在标注文件中查找该病例
            label_row = self.annotations[
                self.annotations['name'].str.contains(case_id, na=False)
            ]

            if label_row.empty:
                continue

            # 提取标签
            labels = self._extract_labels(label_row.iloc[0])

            self.case_files[case_id] = sorted(files)
            self.case_labels[case_id] = labels

    def _extract_labels(self, row: pd.Series) -> np.ndarray:
        """
        从标注行提取多标签向量

        Args:
            row: 标注数据的一行

        Returns:
            labels: 形状为 (9,) 的二进制标签向量
        """
        labels = np.zeros(len(self.LABEL_COLUMNS), dtype=np.float32)

        # 尝试从不同格式的标注中提取
        for i, col in enumerate(self.LABEL_COLUMNS):
            # 方式1: 直接列名 (prediction_probabilities.csv)
            if col in row.index:
                value = row[col]
                labels[i] = 1.0 if value > 0.5 else 0.0

            # 方式2: R1/R2/R3 前缀 (reads.csv) - 取最大值
            r_cols = [f'R1:{col}', f'R2:{col}', f'R3:{col}']
            available = [c for c in r_cols if c in row.index]
            if available:
                max_val = row[available].max()
                labels[i] = 1.0 if max_val > 0 else 0.0

        return labels

    def load_volume(self, case_id: str) -> np.ndarray:
        """
        加载病例的3D体积数据

        Args:
            case_id: 病例ID

        Returns:
            volume: 3D numpy数组 (D, H, W)
        """
        files = self.case_files[case_id]

        # 读取所有 DICOM 切片
        slices = []
        for dcm_file in files:
            ds = pydicom.dcmread(str(dcm_file))
            pixel_array = ds.pixel_array.astype(np.float32)

            # 转换为 HU 单位
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

            slices.append(pixel_array)

        if not slices:
            raise ValueError(f"无法读取病例 {case_id} 的DICOM文件")

        # 按位置排序
        # 注意：这里简化处理，实际应该根据 ImagePositionPatient 排序
        volume = np.stack(slices, axis=0)  # (D, H, W)

        return volume

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        预处理体积数据

        Args:
            volume: 原始体积 (D, H, W)

        Returns:
            processed: 预处理后的体积
        """
        # 1. 裁剪到脑部区域（简单阈值法）
        # 脑组织 HU 范围: -100 到 100
        volume = np.clip(volume, -100, 100)

        # 2. 归一化到 [0, 1]
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max - volume_min > 0:
            volume = (volume - volume_min) / (volume_max - volume_min)

        # 3. 调整大小到目标尺寸
        current_shape = volume.shape
        target_shape = self.target_size

        if current_shape != target_shape:
            zoom_factors = [
                target_shape[0] / current_shape[0],
                target_shape[1] / current_shape[1],
                target_shape[2] / current_shape[2]
            ]
            volume = zoom(volume, zoom_factors, order=1)

        return volume

    def extract_three_views(self, volume: np.ndarray) -> np.ndarray:
        """
        从 3D 体积中提取轴位、冠状、矢状中心切片，并堆叠为三通道图像
        用于训练数据增强（与推理端保持一致）
        """
        d, h, w = volume.shape
        axial = volume[d//2, :, :]
        coronal = volume[:, h//2, :]
        sagittal = volume[:, :, w//2]

        # 旋转使方向一致（与推理端相同的旋转参数）
        coronal = np.rot90(coronal, k=1)
        sagittal = np.rot90(sagittal, k=-1)

        # 可选：统一尺寸（如果训练时输入尺寸固定，此处可 resize）
        # 但通常数据集本身已经预处理过，此处仅提取视图
        return np.stack([axial, coronal, sagittal], axis=0)

    def augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        应用数据增强

        Args:
            volume: 体积数据

        Returns:
            augmented: 增强后的体积
        """
        if not self.transform:
            return volume

        # 1. 随机翻转
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()  # 左右翻转
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()  # 上下翻转

        # 2. 随机旋转（小角度）
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            # 对每个切片应用旋转
            augmented = np.zeros_like(volume)
            for i in range(volume.shape[0]):
                augmented[i] = rotate(volume[i], angle, reshape=False, order=1)
            volume = augmented

        # 3. 添加高斯噪声
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = np.clip(volume + noise, 0, 1)

        return volume

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            dict: {
                'image': 图像张量,
                'labels': 标签张量,
                'case_id': 病例ID
            }
        """
        case_id = self.case_ids[idx]
        labels = self.case_labels[case_id]

        # 加载体积
        volume = self.load_volume(case_id)

        # 预处理
        volume = self.preprocess_volume(volume)

        # 数据增强
        volume = self.augment_volume(volume)

        # 如果使用三视图模式
        if self.use_3_slices:
            volume = self.extract_three_views(volume)
            # 形状: (3, H, W)
            image = torch.from_numpy(volume).float()
        else:
            # 全3D体积
            # 形状: (1, D, H, W) - 添加通道维度
            image = torch.from_numpy(volume).unsqueeze(0).float()

        labels_tensor = torch.from_numpy(labels).float()

        return {
            'image': image,
            'labels': labels_tensor,
            'case_id': case_id
        }


class CQ500SliceDataset(Dataset):
    """
    CQ500 切片级数据集（用于2D模型）
    将3D体积分解为2D切片进行训练
    """

    def __init__(
            self,
            data_root: str,
            annotation_file: str,
            transform: bool = True,
            target_size: tuple = (256, 256)
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size

        # 加载标注
        self.annotations = pd.read_csv(annotation_file)

        # 构建切片列表
        self.slice_list = []  # [(case_id, slice_idx, labels), ...]
        self._build_slice_list()

        print(f"✅ 切片数据集加载完成: {len(self.slice_list)} 个切片")

    def _build_slice_list(self):
        """构建切片列表"""
        ct_dir = self.data_root / "CT"

        if not ct_dir.exists():
            raise FileNotFoundError(f"CT目录不存在: {ct_dir}")

        # 按病例分组
        case_groups = {}
        for dcm_file in ct_dir.glob("*.dcm"):
            case_id = '_'.join(dcm_file.stem.split('_')[:-1]) if '_' in dcm_file.stem else dcm_file.stem

            if case_id not in case_groups:
                case_groups[case_id] = []
            case_groups[case_id].append(dcm_file)

        # 为每个病例的每个切片创建条目
        for case_id, files in case_groups.items():
            # 获取标签
            label_row = self.annotations[
                self.annotations['name'].str.contains(case_id, na=False)
            ]

            if label_row.empty:
                continue

            labels = self._extract_labels(label_row.iloc[0])

            # 为每个切片创建条目
            for slice_idx, dcm_file in enumerate(sorted(files)):
                self.slice_list.append((case_id, slice_idx, str(dcm_file), labels))

    def _extract_labels(self, row: pd.Series) -> np.ndarray:
        """提取标签（同CQ500ClassificationDataset）"""
        label_columns = [
            'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
            'Fracture', 'MassEffect', 'MidlineShift'
        ]
        labels = np.zeros(len(label_columns), dtype=np.float32)

        for i, col in enumerate(label_columns):
            if col in row.index:
                value = row[col]
                labels[i] = 1.0 if value > 0.5 else 0.0

            r_cols = [f'R1:{col}', f'R2:{col}', f'R3:{col}']
            available = [c for c in r_cols if c in row.index]
            if available:
                max_val = row[available].max()
                labels[i] = 1.0 if max_val > 0 else 0.0

        return labels

    def load_and_preprocess_slice(self, file_path: str) -> np.ndarray:
        """加载并预处理单个切片"""
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)

        # 转换为 HU
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        # 裁剪和归一化
        pixel_array = np.clip(pixel_array, -100, 100)
        vmin, vmax = pixel_array.min(), pixel_array.max()
        if vmax - vmin > 0:
            pixel_array = (pixel_array - vmin) / (vmax - vmin)

        # 调整大小
        from scipy.ndimage import zoom
        h, w = pixel_array.shape
        target_h, target_w = self.target_size
        zoom_factors = (target_h / h, target_w / w)
        pixel_array = zoom(pixel_array, zoom_factors, order=1)

        return pixel_array

    def augment_slice(self, image: np.ndarray) -> np.ndarray:
        """2D数据增强"""
        if not self.transform:
            return image

        # 随机翻转
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()

        # 随机旋转
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            from scipy.ndimage import rotate
            image = rotate(image, angle, reshape=False, order=1)

        return image

    def __len__(self) -> int:
        return len(self.slice_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, slice_idx, file_path, labels = self.slice_list[idx]

        # 加载切片
        image = self.load_and_preprocess_slice(file_path)

        # 增强
        image = self.augment_slice(image)

        # ⚠️ 【修改】如果使用三视图模式，复制单通道为3通道
        if USE_THREE_VIEWS:
            # 将 (H, W) 复制为 (3, H, W)
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
            image_tensor = image_tensor.repeat(3, 1, 1)  # (3, H, W)
        else:
            # 单通道
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)

        labels_tensor = torch.from_numpy(labels).float()

        return {
            'image': image_tensor,
            'labels': labels_tensor,
            'case_id': case_id,
            'slice_idx': slice_idx
        }


# 测试代码
if __name__ == "__main__":
    # 测试3D数据集
    print("测试 3D 分类数据集...")
    dataset_3d = CQ500ClassificationDataset(
        data_root="D:/Code/Python/Code/neuSoft/BrainCT/datasets_filtered",
        annotation_file="D:/Code/Python/Code/neuSoft/BrainCT/md/reads.csv",
        use_3_slices=False
    )

    if len(dataset_3d) > 0:
        sample = dataset_3d[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签: {sample['labels']}")
        print(f"病例ID: {sample['case_id']}")

    # 测试2D切片数据集
    print("\n测试 2D 切片数据集...")
    dataset_2d = CQ500SliceDataset(
        data_root="D:/Code/Python/Code/neuSoft/BrainCT/datasets_filtered",
        annotation_file="D:/Code/Python/Code/neuSoft/BrainCT/md/reads.csv"
    )

    if len(dataset_2d) > 0:
        sample = dataset_2d[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签: {sample['labels']}")
        print(f"病例ID: {sample['case_id']}, 切片: {sample['slice_idx']}")
