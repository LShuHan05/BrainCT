"""
CT伪影分割推理引擎
- 支持.nii/.nii.gz文件直接推理
- 保持DICOM/SimpleITK空间元数据对齐
- 生产级批量推理支持
"""

import os
import sys
# 将项目根目录添加到 sys.path，确保绝对导入有效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

# 绝对导入
from Model.AttentionUNet2D import UNet2D
from Conf.Config import DEVICE

class CTArtifactInfer:
    """
    CT伪影分割推理类

    用法示例：
        infer = CTArtifactInfer('run/weights/best.pth')

        # 方式1：从文件路径推理
        mask_img = infer.predict_from_nii('input.nii.gz', 'output_mask.nii.gz')

        # 方式2：从SimpleITK图像推理
        ct_img = sitk.ReadImage('input.nii.gz')
        mask_img = infer.predict_from_sitk(ct_img, 'output_mask.nii.gz')
    """

    def __init__(self, model_weight_path, device=None, use_auxiliary=True, threshold=0.5):
        """
        初始化推理类

        Args:
            model_weight_path: 模型权重路径 (best.pth)
            device: 推理设备，默认使用配置中的DEVICE
            use_auxiliary: 是否启用辅助输出（训练时的配置）
            threshold: 分割阈值，默认0.5
        """
        self.device = device if device else DEVICE
        self.model_weight_path = model_weight_path
        self.use_auxiliary = use_auxiliary
        self.threshold = threshold

        # 加载模型
        self.model = self._load_model()

        print(f"✅ 模型加载成功: {model_weight_path}")
        print(f"📍 设备: {self.device}")
        print(f"🎯 阈值: {self.threshold}")

    def _load_model(self):
        """加载训练好的UNet模型"""
        model = UNet2D(in_ch=1, out_ch=1, use_auxiliary=self.use_auxiliary).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.model_weight_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _normalize_slice(self, img_slice):
        """
        单切片Z-score归一化（与训练保持一致）

        Args:
            img_slice: 2D numpy array

        Returns:
            normalized 2D numpy array
        """
        mean = img_slice.mean()
        std = img_slice.std()
        normalized = (img_slice - mean) / (std + 1e-7)
        return normalized.astype(np.float32)

    def predict_slice(self, img_slice):
        """
        单张切片推理（内部使用）

        Args:
            img_slice: 2D numpy array (已归一化)

        Returns:
            pred_mask: 2D numpy array (0或1)
        """
        # 构造模型输入 [B, C, H, W]
        tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            # 处理深度监督输出的字典格式
            if isinstance(output, dict):
                output = output['main']
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred > self.threshold).astype(np.int16)

        return pred_mask

    def predict_from_nii(self, nii_path, save_mask_path=None, threshold=None):
        """
        核心接口：从.nii文件推理

        Args:
            nii_path: 输入的CT .nii 或 .nii.gz 路径
            save_mask_path: 可选，保存掩码的路径
            threshold: 可选，覆盖初始化时的阈值

        Returns:
            sitk_mask: SimpleITK图像对象（保持空间元数据）
        """
        if threshold is not None:
            self.threshold = threshold

        # 1. 读取CT体积
        print(f"📖 读取CT文件: {nii_path}")
        sitk_ct = sitk.ReadImage(nii_path)
        ct_vol = sitk.GetArrayFromImage(sitk_ct)  # [D, H, W]
        D, H, W = ct_vol.shape
        print(f"📊 体积尺寸: {D} x {H} x {W}")

        # 2. 初始化掩码体积
        mask_vol = np.zeros((D, H, W), dtype=np.int16)

        # 3. 逐切片推理
        print("🔮 开始推理...")
        for z in tqdm(range(D), desc="推理切片"):
            slice_img = ct_vol[z]

            # 归一化
            normalized = self._normalize_slice(slice_img)

            # 推理
            mask_slice = self.predict_slice(normalized)
            mask_vol[z] = mask_slice

        # 4. 转为SimpleITK并对齐空间信息
        print("🔄 重建空间元数据...")
        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)  # ✅ 关键：复制所有空间信息

        # 5. 可选保存
        if save_mask_path is not None:
            output_dir = os.path.dirname(save_mask_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)
            print(f"💾 掩码已保存: {save_mask_path}")

        return sitk_mask

    def predict_from_sitk(self, sitk_ct, save_mask_path=None, threshold=None):
        """
        进阶接口：直接处理SimpleITK图像（适合GUI流程）

        Args:
            sitk_ct: SimpleITK图像对象
            save_mask_path: 可选，保存掩码的路径
            threshold: 可选，覆盖初始化时的阈值

        Returns:
            sitk_mask: SimpleITK图像对象
        """
        if threshold is not None:
            self.threshold = threshold

        # 1. 获取numpy数组
        ct_vol = sitk.GetArrayFromImage(sitk_ct)
        D, H, W = ct_vol.shape

        # 2. 初始化掩码
        mask_vol = np.zeros((D, H, W), dtype=np.int16)

        # 3. 逐切片推理
        for z in tqdm(range(D), desc="推理切片"):
            slice_img = ct_vol[z]
            normalized = self._normalize_slice(slice_img)
            mask_vol[z] = self.predict_slice(normalized)

        # 4. 重建SimpleITK
        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)

        # 5. 可选保存
        if save_mask_path:
            output_dir = os.path.dirname(save_mask_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)

        return sitk_mask

    def predict_batch(self, nii_dir, output_dir, threshold=None):
        """
        批量推理整个目录的.nii文件

        Args:
            nii_dir: 输入目录（包含多个.nii文件）
            output_dir: 输出目录
            threshold: 可选，覆盖阈值
        """
        if threshold is not None:
            self.threshold = threshold

        os.makedirs(output_dir, exist_ok=True)

        nii_files = [f for f in os.listdir(nii_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        print(f"📁 找到 {len(nii_files)} 个文件")

        for nii_file in tqdm(nii_files, desc="批量推理"):
            input_path = os.path.join(nii_dir, nii_file)
            output_path = os.path.join(output_dir, nii_file.replace('.gz', '').replace('.nii', '_mask.nii'))

            try:
                self.predict_from_nii(input_path, output_path)
            except Exception as e:
                print(f"❌ 处理失败 {nii_file}: {e}")
                continue

        print(f"✅ 批量推理完成！结果保存在: {output_dir}")


# ====================== 使用示例 ======================
if __name__ == '__main__':
    # 初始化推理器
    infer = CTArtifactInfer(
        model_weight_path='/run/weights/best.pth',
        threshold=0.5
    )

    # 单文件推理
    mask = infer.predict_from_nii(
        nii_path='test_sample.nii.gz',
        save_mask_path='test_sample_mask.nii.gz'
    )

    print(f"✅ 推理完成！掩码形状: {mask.GetSize()}")
