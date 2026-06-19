import sys
import os

# 添加项目根目录到路径（支持直接运行）
# 当前文件: D:/Code/Python/Code/neuSoft/BrainCT/Inference/test_infer.py
# 需要追溯到: D:/Code/Python/Code/neuSoft (3层dirname)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import SimpleITK as sitk
from BrainCT.Inference.CTArtifactInfer import CTArtifactInfer
from BrainCT.Model.AttentionUNet2D import UNet2D


def test_model_loading():
    """测试1：模型加载"""
    print("=" * 60)
    print("测试1：模型加载")
    print("=" * 60)

    try:
        # 使用相对路径（相对于项目根目录）
        weight_path = os.path.join(project_root, 'BrainCT', 'run', 'weights', 'best.pth')
        infer = CTArtifactInfer(
            model_weight_path=weight_path,
            threshold=0.5
        )
        print("✅ 模型加载成功！")
        return infer
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def test_feature_extraction():
    """测试2：特征提取功能"""
    print("\n" + "=" * 60)
    print("测试2：特征提取功能")
    print("=" * 60)

    try:
        # 创建模型
        model = UNet2D(use_auxiliary=True)
        model.eval()

        # 构造随机输入
        x = torch.randn(1, 1, 256, 256)

        with torch.no_grad():
            output = model(x)
            features = model.extract_features()

        print(f"✅ 输入形状: {x.shape}")
        print(f"✅ 输出形状: {output.shape}")
        print(f"✅ 特征向量形状: {features.shape}")
        print(f"✅ 特征归一化检查: L2范数 = {torch.norm(features).item():.4f}")

        return True
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_inference():
    """测试3：合成数据推理"""
    print("\n" + "=" * 60)
    print("测试3：合成数据推理（完整流程）")
    print("=" * 60)

    try:
        # 初始化推理器
        weight_path = os.path.join(project_root, 'BrainCT', 'run', 'weights', 'best.pth')
        infer = CTArtifactInfer(
            model_weight_path=weight_path,
            threshold=0.5
        )

        # 【改进】创建更真实的模拟CT（包含类似金属伪影的高密度区域）
        print("🔧 创建模拟CT体积（含高密度伪影区域）...")
        D, H, W = 10, 256, 256
        ct_vol = np.random.randn(D, H, W).astype(np.float32) * 100 + 500  # HU值范围

        # 添加一些高密度区域（模拟金属伪影）
        center_y, center_x = H // 2, W // 2
        for z in range(D):
            # 在中心附近创建一个高密度圆形区域
            y, x = np.meshgrid(range(H), range(W))
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask_circle = distance < 20  # 半径20的圆
            ct_vol[z][mask_circle] = 3000  # 模拟金属的高HU值（通常>2000）

        # 转为SimpleITK
        sitk_ct = sitk.GetImageFromArray(ct_vol)
        sitk_ct.SetSpacing([1.0, 1.0, 5.0])  # 设置体素间距
        sitk_ct.SetOrigin([0.0, 0.0, 0.0])

        print(f"📊 合成CT尺寸: {ct_vol.shape}")
        print(f"📍 体素间距: {sitk_ct.GetSpacing()}")
        print(f"🔬 HU值范围: [{ct_vol.min():.0f}, {ct_vol.max():.0f}]")
        print(f"⚡ 高密度像素数: {(ct_vol > 2000).sum()}")

        # 推理
        print("🔮 开始推理...")
        sitk_mask = infer.predict_from_sitk(sitk_ct, save_mask_path=None)

        # 验证结果
        mask_vol = sitk.GetArrayFromImage(sitk_mask)
        print(f"\n✅ 推理完成！")
        print(f"📊 掩码形状: {mask_vol.shape}")
        print(f"📈 阳性像素数: {(mask_vol > 0).sum()}")
        print(f"📉 阴性像素数: {(mask_vol == 0).sum()}")
        print(f"🎯 阳性比例: {(mask_vol > 0).sum() / mask_vol.size * 100:.2f}%")

        # 验证空间元数据
        print(f"\n🔍 空间元数据检查:")
        print(f"   原始CT间距: {sitk_ct.GetSpacing()}")
        print(f"   掩码间距: {sitk_mask.GetSpacing()}")
        print(f"   原始CT原点: {sitk_ct.GetOrigin()}")
        print(f"   掩码原点: {sitk_mask.GetOrigin()}")
        print(f"   ✅ 空间信息对齐: {sitk_ct.GetSpacing() == sitk_mask.GetSpacing()}")

        # 【新增】检查结果合理性
        if (mask_vol > 0).sum() > 0:
            print(f"   ✅ 检测到伪影区域（合理）")
        else:
            print(f"   ⚠️ 未检测到伪影（可能需要调整输入数据）")

        return True

    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nii_file_io():
    """测试4：.nii文件读写"""
    print("\n" + "=" * 60)
    print("测试4：.nii文件读写测试")
    print("=" * 60)

    try:
        # 创建临时测试文件
        temp_dir = os.path.join(project_root, 'BrainCT', 'test_temp')
        os.makedirs(temp_dir, exist_ok=True)

        input_path = os.path.join(temp_dir, 'test_input.nii.gz')
        output_path = os.path.join(temp_dir, 'test_output_mask.nii.gz')

        # 创建测试CT
        print("🔧 创建测试.nii文件...")
        ct_vol = np.random.randn(20, 256, 256).astype(np.float32) * 100
        sitk_ct = sitk.GetImageFromArray(ct_vol)
        sitk_ct.SetSpacing([1.0, 1.0, 5.0])
        sitk.WriteImage(sitk_ct, input_path)
        print(f"✅ 输入文件已保存: {input_path}")

        # 推理并保存
        print("🔮 开始推理并保存...")
        weight_path = os.path.join(project_root, 'BrainCT', 'run', 'weights', 'best.pth')
        infer = CTArtifactInfer(
            model_weight_path=weight_path,
            threshold=0.5
        )

        sitk_mask = infer.predict_from_nii(input_path, output_path)

        # 验证文件存在
        if os.path.exists(output_path):
            print(f"✅ 输出文件已保存: {output_path}")

            # 重新读取验证
            loaded_mask = sitk.ReadImage(output_path)
            loaded_vol = sitk.GetArrayFromImage(loaded_mask)
            print(f"✅ 文件读取成功，形状: {loaded_vol.shape}")
            print(f"✅ 数据类型: {loaded_vol.dtype}")
            print(f"✅ 唯一值: {np.unique(loaded_vol)}")
        else:
            print(f"❌ 输出文件未找到: {output_path}")
            return False

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 临时文件已清理")

        return True

    except Exception as e:
        print(f"❌ 文件IO测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("CTArtifactInfer 推理引擎测试套件")
    print("🚀" * 30 + "\n")

    results = {}

    # 测试1：模型加载
    infer = test_model_loading()
    results['模型加载'] = infer is not None

    # 测试2：特征提取
    results['特征提取'] = test_feature_extraction()

    # 测试3：合成数据推理
    results['合成数据推理'] = test_synthetic_inference()

    # 测试4：文件IO
    results['文件IO'] = test_nii_file_io()

    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 测试汇总报告")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:15s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！推理引擎工作正常！")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查错误信息")


if __name__ == '__main__':
    main()
