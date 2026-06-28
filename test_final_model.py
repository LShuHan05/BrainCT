# test_final_model.py
"""
快速验证最终模型是否加载成功并能正常推理
"""

import os
import sys
import torch
import pydicom
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Conf.Config import *
from Model.LesionClassifier import LesionClassifier2D

def test_model():
    print("=" * 60)
    print("🧪 测试最终模型加载与推理")
    print("=" * 60)

    # 1. 加载模型
    weight_path = os.path.join(WEIGHT_SAVE_DIR, "final_best_lesion.pth")
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        return

    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    print(f"✅ 加载权重: {weight_path}")
    print(f"   Best F1: {checkpoint.get('best_f1', 0):.4f}")
    print(f"   Best Thresholds: {checkpoint.get('best_thresholds', [])}")

    model = LesionClassifier2D(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        use_three_views=USE_THREE_VIEWS
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("✅ 模型结构加载完成")

    # 2. 找一张测试 DICOM 文件
    test_dcm_path = os.path.join(CQ500_DATA_ROOT, "CT", "CQ500CT0_slice_000.dcm")
    if not os.path.exists(test_dcm_path):
        # 尝试另一个路径
        test_dcm_path = "/mnt/workspace/BrainCT/datasets_filtered/CT/CQ500CT0_slice_000.dcm"
        if not os.path.exists(test_dcm_path):
            print("⚠️  未找到测试 DICOM，使用随机张量模拟")
            dummy_input = torch.randn(1, 3, 512, 512).to(DEVICE)
        else:
            ds = pydicom.dcmread(test_dcm_path)
            arr = ds.pixel_array.astype(np.float32)
            arr = np.clip(arr, -100, 100)
            vmin, vmax = arr.min(), arr.max()
            if vmax - vmin > 0:
                arr = (arr - vmin) / (vmax - vmin)
            h, w = arr.shape
            arr = zoom(arr, (512/h, 512/w), order=1)
            dummy_input = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            dummy_input = dummy_input.repeat(1, 3, 1, 1).to(DEVICE)
            print(f"✅ 加载真实 DICOM: {test_dcm_path}")
    else:
        ds = pydicom.dcmread(test_dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        arr = np.clip(arr, -100, 100)
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 0:
            arr = (arr - vmin) / (vmax - vmin)
        h, w = arr.shape
        arr = zoom(arr, (512/h, 512/w), order=1)
        dummy_input = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        dummy_input = dummy_input.repeat(1, 3, 1, 1).to(DEVICE)
        print(f"✅ 加载真实 DICOM: {test_dcm_path}")

    # 3. 推理
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    print("\n📊 预测结果（概率）:")
    for label, prob in zip(LESION_LABELS, probs):
        print(f"   {label:12s}: {prob:.4f}")

    # 4. 应用阈值
    thresholds = checkpoint.get('best_thresholds', [0.5] * NUM_CLASSES)
    print("\n🎯 应用最佳阈值后的阳性判断:")
    for i, (label, prob) in enumerate(zip(LESION_LABELS, probs)):
        pos = prob >= thresholds[i]
        status = "✅ 阳性" if pos else "❌ 阴性"
        print(f"   {label:12s}: {prob:.4f} >= {thresholds[i]:.3f} -> {status}")

    print("\n🎉 模型测试通过！")

if __name__ == "__main__":
    test_model()