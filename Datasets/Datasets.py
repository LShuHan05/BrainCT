import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter

# ⚠️ 【修改】改为绝对导入
from Utils.Preprocessing import normalize_ct_slice
from Conf.Config import (
    USE_ELASTIC_TRANSFORM, ELASTIC_ALPHA, ELASTIC_SIGMA,
    USE_RANDOM_BRIGHTNESS_CONTRAST, BRIGHTNESS_RANGE, CONTRAST_RANGE
)
class CTSliceDataset(Dataset):
    def __init__(self, ct_root, mask_root):
        self.ct_root = ct_root
        self.mask_root = mask_root
        self.is_train = False

        print("⏳ 正在预加载数据到内存...")
        self.ct_list = []
        self.mask_list = []

        all_dcm = sorted([f for f in os.listdir(ct_root) if f.endswith(".dcm")])
        for fname in all_dcm:
            ct_path = os.path.join(ct_root, fname)
            mask_path = os.path.join(mask_root, fname)
            if os.path.exists(mask_path):
                ct_arr = pydicom.dcmread(ct_path).pixel_array.astype(np.float32)
                mask_arr = pydicom.dcmread(mask_path).pixel_array.astype(np.float32)
                ct_arr = normalize_ct_slice(ct_arr)
                mask_arr = (mask_arr > 0).astype(np.float32)
                self.ct_list.append(ct_arr)
                self.mask_list.append(mask_arr)

        print(f"✅ 数据集预加载完成！总切片数量：{len(self.ct_list)}")

    def __len__(self):
        return len(self.ct_list)

    # 【新增】弹性变形
    def elastic_transform(self, image, mask, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA):
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = (x + dx, y + dy)
        image_deformed = map_coordinates(image, indices, order=1, mode='reflect')
        mask_deformed = map_coordinates(mask, indices, order=0, mode='reflect')
        return image_deformed, mask_deformed

    # 【新增】随机亮度对比度调整
    def adjust_brightness_contrast(self, img, brightness_range=0.1, contrast_range=0.1):
        brightness = 1 + np.random.uniform(-brightness_range, brightness_range)
        contrast = 1 + np.random.uniform(-contrast_range, contrast_range)
        img = img * contrast + brightness
        return np.clip(img, 0, 1)   # CT 已归一化到 [0,1]

    def __getitem__(self, idx):
        ct_arr = self.ct_list[idx].copy()
        mask_arr = self.mask_list[idx].copy()

        if self.is_train:
            # 1. 弹性变形
            if USE_ELASTIC_TRANSFORM and np.random.rand() > 0.5:
                ct_arr, mask_arr = self.elastic_transform(ct_arr, mask_arr)

            # 2. 随机亮度对比度
            if USE_RANDOM_BRIGHTNESS_CONTRAST and np.random.rand() > 0.5:
                ct_arr = self.adjust_brightness_contrast(ct_arr, BRIGHTNESS_RANGE, CONTRAST_RANGE)

            # 3. 翻转
            if np.random.rand() > 0.5:
                ct_arr = np.fliplr(ct_arr).copy()
                mask_arr = np.fliplr(mask_arr).copy()
            if np.random.rand() > 0.5:
                ct_arr = np.flipud(ct_arr).copy()
                mask_arr = np.flipud(mask_arr).copy()

            # 4. 随机旋转
            if np.random.rand() > 0.5:
                rot_angle = np.random.randint(-8, 9)
                from scipy.ndimage import rotate
                ct_arr = rotate(ct_arr, rot_angle, reshape=False, order=1)
                mask_arr = rotate(mask_arr, rot_angle, reshape=False, order=0)

            # 5. Cutout
            if np.random.rand() > 0.85:
                h, w = ct_arr.shape
                cutout_h, cutout_w = h // 8, w // 8
                start_y = np.random.randint(0, h - cutout_h)
                start_x = np.random.randint(0, w - cutout_w)
                ct_arr[start_y:start_y+cutout_h, start_x:start_x+cutout_w] = 0

        ct = torch.from_numpy(ct_arr).unsqueeze(0).float()
        mask = torch.from_numpy(mask_arr).unsqueeze(0).float()
        return ct, mask