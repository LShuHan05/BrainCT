from typing import Dict, List, Optional
from .LLMService import LLMService
from Conf.Config import LESION_LABELS
from datetime import datetime

class MedicalReportGenerator:
    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService()

    def _build_prompt(
        self,
        case_id: str,
        predictions: Dict[str, float],
        exam_type: str = "头部CT平扫",
        patient_info: Optional[Dict] = None
    ) -> str:
        """构建提示词模板"""
        findings = []
        for label, prob in predictions.items():
            if prob >= 0.5:
                level = "高度可能" if prob >= 0.8 else "中度可能" if prob >= 0.6 else "低度可能"
                findings.append(f"- {label}: {level}（概率 {prob:.1%}）")
        if not findings:
            findings.append("- 未发现明显异常征象")

        patient_str = ""
        if patient_info:
            patient_str = f"- 患者姓名：{patient_info.get('name', '未知')}\n- 年龄：{patient_info.get('age', '未知')}\n- 性别：{patient_info.get('gender', '未知')}"

        return f"""
你是一位经验丰富的放射科医生。请根据以下 AI 影像分析结果，生成一份专业的医学影像报告。

**患者信息：**
{patient_str}
- 检查类型：{exam_type}
- 检查时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

**AI 分析结果：**
{chr(10).join(findings)}

**要求：**
1. 使用专业医学术语，结构清晰（检查方法、影像表现、诊断意见、建议）
2. 语气客观、准确
3. 长度控制在 300-500 字
4. 用中文输出

请生成报告：
"""

    def generate_report(
        self,
        case_id: str,
        predictions: Dict[str, float],
        exam_type: str = "头部CT平扫",
        patient_info: Optional[Dict] = None,
        use_cache: bool = True
    ) -> str:
        """异步生成报告"""
        prompt = self._build_prompt(case_id, predictions, exam_type, patient_info)
        messages = [
            {"role": "system", "content": "你是一位专业的放射科医生。"},
            {"role": "user", "content": prompt}
        ]
        response =  self.llm.chat_completion(
            messages=messages,
            temperature=0.5,
            max_tokens=1000,
            use_cache=use_cache
        )
        return response['choices'][0]['message']['content']