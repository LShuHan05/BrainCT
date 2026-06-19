"""
综合推理服务
整合：
1. 病灶分类模型
2. 报告生成
3. 数据库记录
4. 对外 API
"""

import torch
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
import json

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Conf.Config import *
from Model.LesionClassifier import LesionClassifier2D
from Services.LLMService import MedicalReportGenerator, LLMService
import pydicom
import numpy as np
from scipy.ndimage import zoom


class ComprehensiveInferenceService:
    """综合推理服务"""

    def __init__(
            self,
            model_path: str = None,
            api_key: str = None
    ):
        """
        Args:
            model_path: 模型权重路径
            api_key: LLM API Key
        """
        self.device = DEVICE
        self.label_names = LESION_LABELS

        # 加载模型
        if model_path is None:
            model_path = os.path.join("run", "weights", "best_lesion.pth")

        self.model = self._load_model(model_path)

        # 初始化 LLM 服务
        self.llm_service = LLMService(api_key=api_key)
        self.report_generator = MedicalReportGenerator(self.llm_service)

        print("✅ 综合推理服务初始化完成")

    def _load_model(self, model_path: str):
        """加载模型"""
        model = LesionClassifier2D(
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS,
            use_three_views=USE_THREE_VIEWS
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"✅ 模型加载成功: {model_path}")
        print(f"   Best F1: {checkpoint.get('best_f1', 0):.4f}")

        return model

    def preprocess_dicom(self, dcm_path: str) -> torch.Tensor:
        """预处理 DICOM 文件"""
        ds = pydicom.dcmread(dcm_path)
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
        h, w = pixel_array.shape
        target_h, target_w = TARGET_SIZE_2D
        zoom_factors = (target_h / h, target_w / w)
        pixel_array = zoom(pixel_array, zoom_factors, order=1)

        # 转换为张量
        tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()

        return tensor.to(self.device)

    @torch.no_grad()
    def predict_from_file(self, dcm_path: str) -> Dict:
        """
        从 DICOM 文件预测

        Args:
            dcm_path: DICOM 文件路径

        Returns:
            result: 预测结果字典
        """
        # 预处理
        tensor = self.preprocess_dicom(dcm_path)

        # 推理
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 构建结果
        predictions = {}
        positive_findings = []

        for label, prob in zip(self.label_names, probs):
            predictions[label] = float(prob)

            if prob >= POSITIVE_THRESHOLD:
                positive_findings.append({
                    "label": label,
                    "probability": float(prob),
                    "severity": "high" if prob >= 0.8 else "medium" if prob >= 0.6 else "low"
                })

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "positive_findings": positive_findings,
            "emergency_level": self._assess_emergency_level(positive_findings)
        }

        return result

    def _assess_emergency_level(self, findings: List[Dict]) -> str:
        """评估紧急程度"""
        if not findings:
            return "B1"  # 轻症

        # 高危病变
        high_risk_labels = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']

        for finding in findings:
            if finding['label'] in high_risk_labels and finding['probability'] >= 0.7:
                return "B2"  # 中重度

        return "B1"

    def generate_full_report(
            self,
            dcm_path: str,
            case_id: str = None,
            patient_info: Dict = None
    ) -> Dict:
        """
        生成完整报告（AI预测 + LLM润色）

        Args:
            dcm_path: DICOM 文件路径
            case_id: 病例ID
            patient_info: 患者信息

        Returns:
            full_report: 完整报告
        """
        # 1. AI 预测
        prediction_result = self.predict_from_file(dcm_path)

        # 2. 生成结构化报告
        report_text = self.report_generator.generate_report(
            case_id=case_id or "Unknown",
            ai_predictions=prediction_result['predictions'],
            exam_type="头部CT平扫"
        )

        # 3. 组合完整报告
        full_report = {
            "case_id": case_id,
            "patient_info": patient_info,
            "exam_time": datetime.now().isoformat(),
            "ai_analysis": prediction_result,
            "structured_report": report_text,
            "recommendations": self._generate_recommendations(prediction_result)
        }

        return full_report

    def _generate_recommendations(self, prediction: Dict) -> List[str]:
        """生成诊疗建议"""
        recommendations = []

        findings = prediction.get('positive_findings', [])

        # 根据检测结果生成建议
        for finding in findings:
            label = finding['label']

            if label in ['ICH', 'IPH']:
                recommendations.append("🔴 立即神经外科会诊，评估手术指征")
                recommendations.append("🔴 严密监测生命体征和意识状态")
                recommendations.append("🟡 控制血压在安全范围")

            elif label == 'SDH':
                recommendations.append("🔴 评估血肿量和占位效应")
                recommendations.append("🟡 准备可能的钻孔引流术")

            elif label == 'Fracture':
                recommendations.append("🟡 评估骨折类型和位移程度")
                recommendations.append("🟢 预防性使用抗生素")

        if not recommendations:
            recommendations.append("🟢 未见明显异常，建议定期体检")

        return recommendations

    def save_report_to_file(self, report: Dict, output_path: str):
        """保存报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 报告已保存: {output_path}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化服务
    service = ComprehensiveInferenceService(
        model_path="run/weights/best_lesion.pth"
    )

    # 测试预测
    test_dcm = "datasets_filtered/CT/CQ500CT0_slice_000.dcm"

    if os.path.exists(test_dcm):
        result = service.predict_from_file(test_dcm)

        print("\n" + "=" * 60)
        print("预测结果:")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 生成完整报告
        full_report = service.generate_full_report(
            dcm_path=test_dcm,
            case_id="CQ500CT0",
            patient_info={"name": "测试患者", "age": 45}
        )

        print("\n" + "=" * 60)
        print("完整报告:")
        print("=" * 60)
        print(full_report['structured_report'])

        # 保存报告
        service.save_report_to_file(full_report, "report_output.json")
    else:
        print(f"⚠️  测试文件不存在: {test_dcm}")
