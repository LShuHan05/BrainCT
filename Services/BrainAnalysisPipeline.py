"""
脑部CT综合分析流水线
整合：
1. 图像质量评估（伪影检测）
2. 病灶识别
3. 报告生成
"""

import torch
import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.UNet2D import UNet2D as ArtifactUNet
from Model.LesionClassifier import LesionClassifier2D
from Services.LLMService import MedicalReportGenerator


class BrainAnalysisPipeline:
    """脑部CT分析流水线"""

    def __init__(
            self,
            artifact_model_path: str = "run/weights/best.pth",
            lesion_model_path: str = "run/weights/best_lesion.pth"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载伪影分割模型
        print("🔄 加载伪影分割模型...")
        self.artifact_model = self._load_artifact_model(artifact_model_path)

        # 加载病灶分类模型
        print("🔄 加载病灶分类模型...")
        self.lesion_model = self._load_lesion_model(lesion_model_path)

        # 初始化报告生成器
        self.report_generator = MedicalReportGenerator()

        print("✅ 流水线初始化完成")

    def _load_artifact_model(self, path: str):
        """加载伪影分割模型"""
        model = ArtifactUNet(in_ch=1, out_ch=1)

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✅ 伪影模型加载成功: {path}")
        else:
            print(f"   ⚠️  伪影模型文件不存在: {path}，将跳过伪影检测")
            return None

        model.to(self.device)
        model.eval()
        return model

    def _load_lesion_model(self, path: str):
        """加载病灶分类模型"""
        model = LesionClassifier2D(
            num_classes=9,
            input_channels=1,
            use_three_views=False
        )

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ 病灶模型加载成功: {path}")
        else:
            print(f"   ⚠️  病灶模型文件不存在: {path}")
            return None

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def detect_artifact(self, image: torch.Tensor) -> Dict:
        """
        检测伪影

        Args:
            image: CT图像 (1, 1, H, W)

        Returns:
            result: 伪影检测结果
        """
        if self.artifact_model is None:
            return {
                'has_artifact': False,
                'artifact_ratio': 0.0,
                'mask': None,
                'quality': 'unknown'
            }

        # 推理
        image = image.to(self.device)
        mask = self.artifact_model(image)
        mask_prob = torch.sigmoid(mask)

        # 计算伪影比例
        artifact_ratio = mask_prob.mean().item()

        # 二值化
        mask_binary = (mask_prob > 0.5).float()

        # 质量评估
        if artifact_ratio > 0.15:
            quality = 'poor'
            has_artifact = True
        elif artifact_ratio > 0.05:
            quality = 'fair'
            has_artifact = True
        else:
            quality = 'good'
            has_artifact = False

        return {
            'has_artifact': has_artifact,
            'artifact_ratio': artifact_ratio,
            'mask': mask_binary.cpu(),
            'quality': quality
        }

    @torch.no_grad()
    def classify_lesion(self, image: torch.Tensor) -> Dict:
        """
        病灶分类

        Args:
            image: CT图像 (1, 1, H, W)

        Returns:
            result: 病灶分类结果
        """
        if self.lesion_model is None:
            return {
                'predictions': {},
                'positive_findings': [],
                'emergency_level': 'unknown'
            }

        # 推理
        image = image.to(self.device)
        logits = self.lesion_model(image)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 标签映射
        label_names = [
            'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
            'Fracture', 'MassEffect', 'MidlineShift'
        ]

        predictions = {}
        positive_findings = []

        for label, prob in zip(label_names, probs):
            predictions[label] = float(prob)

            if prob >= 0.5:
                positive_findings.append({
                    'label': label,
                    'probability': float(prob),
                    'severity': 'high' if prob >= 0.8 else 'medium' if prob >= 0.6 else 'low'
                })

        # 紧急程度评估
        emergency_level = self._assess_emergency(positive_findings)

        return {
            'predictions': predictions,
            'positive_findings': positive_findings,
            'emergency_level': emergency_level
        }

    def _assess_emergency(self, findings: list) -> str:
        """评估紧急程度"""
        if not findings:
            return "B1"

        high_risk = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']

        for finding in findings:
            if finding['label'] in high_risk and finding['probability'] >= 0.7:
                return "B2"

        return "B1"

    def analyze(self, image: torch.Tensor, case_id: str = None) -> Dict:
        """
        完整分析流程

        Args:
            image: CT图像
            case_id: 病例ID

        Returns:
            full_result: 完整分析结果
        """
        print("\n" + "=" * 60)
        print("开始脑部CT综合分析")
        print("=" * 60)

        # 步骤1：伪影检测
        print("\n📊 步骤1: 图像质量评估...")
        artifact_result = self.detect_artifact(image)

        print(f"   图像质量: {artifact_result['quality']}")
        print(f"   伪影比例: {artifact_result['artifact_ratio']:.2%}")

        if artifact_result['has_artifact']:
            print(f"   ⚠️  检测到伪影")
        else:
            print(f"   ✅ 图像质量良好")

        # 步骤2：病灶识别
        print("\n🔍 步骤2: 病灶识别...")
        lesion_result = self.classify_lesion(image)

        print(f"   紧急程度: {lesion_result['emergency_level']}")
        print(f"   阳性发现: {len(lesion_result['positive_findings'])} 个")

        for finding in lesion_result['positive_findings']:
            print(f"     - {finding['label']}: {finding['probability']:.2%} ({finding['severity']})")

        # 步骤3：生成报告
        print("\n📝 步骤3: 生成报告...")
        report = self.report_generator.generate_report(
            case_id=case_id or "Unknown",
            ai_predictions=lesion_result['predictions'],
            exam_type="头部CT平扫"
        )

        # 组合结果
        full_result = {
            'case_id': case_id,
            'quality_assessment': artifact_result,
            'lesion_analysis': lesion_result,
            'structured_report': report,
            'timestamp': str(np.datetime64('now'))
        }

        print("\n" + "=" * 60)
        print("✅ 分析完成")
        print("=" * 60)

        return full_result


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import pydicom
    from scipy.ndimage import zoom

    # 初始化流水线
    pipeline = BrainAnalysisPipeline(
        artifact_model_path="run/weights/best.pth",
        lesion_model_path="run/weights/best_lesion.pth"
    )

    # 加载测试图像
    test_file = "datasets_filtered/CT/CQ500CT0_slice_000.dcm"

    if os.path.exists(test_file):
        ds = pydicom.dcmread(test_file)
        pixel_array = ds.pixel_array.astype(np.float32)

        # 预处理
        pixel_array = np.clip(pixel_array, -100, 100)
        vmin, vmax = pixel_array.min(), pixel_array.max()
        if vmax - vmin > 0:
            pixel_array = (pixel_array - vmin) / (vmax - vmin)

        # 调整大小
        h, w = pixel_array.shape
        zoom_factors = (256 / h, 256 / w)
        pixel_array = zoom(pixel_array, zoom_factors, order=1)

        # 转换为张量
        tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()

        # 分析
        result = pipeline.analyze(tensor, case_id="CQ500CT0")

        # 打印报告
        print("\n" + "=" * 60)
        print("诊断报告:")
        print("=" * 60)
        print(result['structured_report'])
    else:
        print(f"⚠️  测试文件不存在: {test_file}")
