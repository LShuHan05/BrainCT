import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import os
import torch

# 改为相对导入
from ..Conf.Config import *
from ..Model.UNet2D import UNet2D


class CTArtifactInfer:

    def __init__(self, model_weight_path, device=None):
        """
        初始化推理类
        :param model_weight_path: 模型权重路径 best.pth
        :param device: 推理设备，默认使用配置中的 DEVICE
        """
        self.device = device if device is not None else DEVICE
        self.model_weight_path = model_weight_path
        self.model = self._load_model()

    def _load_model(self):
        """加载训练好的UNet模型"""
        model = UNet2D().to(self.device)
        model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
        model.eval()
        return model

    def predict_slice(self, img_slice):
        """单张切片推理（内部使用）"""
        img_slice = img_slice.astype(np.float32)

        # 训练一致的 Z-score 归一化
        mean = img_slice.mean()
        std = img_slice.std()
        img_slice = (img_slice - mean) / (std + 1e-7)

        # 构造模型输入
        tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred > 0.5).astype(np.int16)

        return pred_mask

    def predict_from_nii(self, nii_path, save_mask_path=None):
        """
        外部调用核心接口：输入 nii 路径 → 返回掩码 SimpleITK 图像
        :param nii_path: 输入的CT .nii / .nii.gz 路径
        :param save_mask_path: 可选，保存掩码路径
        :return: sitk_mask (SimpleITK 图像)
        """
        # 1. 读取CT
        sitk_ct = sitk.ReadImage(nii_path)
        ct_vol = sitk.GetArrayFromImage(sitk_ct)
        D, H, W = ct_vol.shape

        # 2. 初始化掩码
        mask_vol = np.zeros((D, H, W), dtype=np.int16)

        # 3. 逐切片推理
        for z in tqdm(range(D), desc="推理切片"):
            slice_img = ct_vol[z]
            mask_slice = self.predict_slice(slice_img)
            mask_vol[z] = mask_slice

        # 4. 转为SimpleITK并对齐空间信息
        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)

        # 5. 可选保存 ✅ 在这里修复
        if save_mask_path is not None:
            output_dir = os.path.dirname(save_mask_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)

        return sitk_mask  # ✅ 返回结果给外部

    def predict_from_sitk(self, sitk_ct, save_mask_path=None):
        """
        进阶：直接传入 SimpleITK 图像 → 返回掩码
        适合在GUI/流程中直接调用，无需读写文件
        """
        ct_vol = sitk.GetArrayFromImage(sitk_ct)
        D, H, W = ct_vol.shape
        mask_vol = np.zeros((D, H, W), dtype=np.int16)

        for z in range(D):
            slice_img = ct_vol[z]
            mask_vol[z] = self.predict_slice(slice_img)

        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)

        if save_mask_path:
            output_dir = os.path.dirname(save_mask_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)

        return sitk_mask