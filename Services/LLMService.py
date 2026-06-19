"""
大语言模型服务
集成 DeepSeek、OpenAI 等 API
提供：
1. 医学报告生成
2. 医患对话
3. 诊断建议优化
"""

import os
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime
import time
from Conf.Config import DEEPSEEK_MODEL

class LLMService:
    """大语言模型服务基类"""

    def __init__(self, api_key: str = None, model: str = DEEPSEEK_MODEL):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1"

        if not self.api_key:
            print("⚠️  警告: 未设置 API Key，将使用模拟模式")
            self.mock_mode = True
        else:
            self.mock_mode = False

    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 2000,
            **kwargs
    ) -> Dict:
        """
        聊天完成

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            response: API 响应
        """
        if self.mock_mode:
            return self._mock_chat_response(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            # 打印更详细的错误信息
            print(f"❌ API 调用失败: {e}")
            print(f"响应内容: {e.response.text}")  # 这里会包含具体的错误码和原因
            return self._mock_chat_response(messages)

    def _mock_chat_response(self, messages: List[Dict]) -> Dict:
        """模拟响应（用于测试）"""
        last_message = messages[-1]['content'] if messages else ""

        mock_response = {
            "id": "mock_" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self._generate_mock_content(last_message)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(last_message) // 4,
                "completion_tokens": 100,
                "total_tokens": 100 + len(last_message) // 4
            }
        }

        return mock_response

    def _generate_mock_content(self, user_message: str) -> str:
        """生成模拟回复内容"""
        if "出血" in user_message or "ICH" in user_message:
            return """根据影像分析结果，患者存在脑内出血（ICH）。

**临床建议：**
1. 立即进行神经外科会诊
2. 监测生命体征和意识状态
3. 控制血压，避免继续出血
4. 必要时考虑手术治疗

**预后评估：**
出血量较小，位置不在关键功能区，预后相对良好。但需密切观察24-48小时。"""

        elif "骨折" in user_message or "Fracture" in user_message:
            return """影像显示存在颅骨骨折。

**处理建议：**
1. 评估骨折类型和位移程度
2. 检查是否有脑脊液漏
3. 预防性使用抗生素
4. 定期复查CT观察愈合情况

大多数线性骨折可保守治疗，凹陷性骨折可能需要手术复位。"""

        else:
            return """根据目前的检查结果，我建议：

1. **进一步检查**：建议完善相关实验室检查
2. **对症治疗**：根据症状给予相应处理
3. **随访观察**：定期复查，监测病情变化

如有任何不适，请及时就医。"""


class MedicalReportGenerator:
    """医学报告生成器"""

    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService()

        # 报告模板
        self.report_template = """
你是一位经验丰富的放射科医生。请根据以下 AI 影像分析结果，生成一份专业的医学影像报告。

**患者信息：**
- 病历号：{case_id}
- 检查时间：{exam_time}
- 检查类型：{exam_type}

**AI 分析结果：**
{ai_findings}

**要求：**
1. 使用专业医学术语
2. 结构清晰，包含：检查方法、影像表现、诊断意见、建议
3. 语气客观、准确
4. 长度控制在 300-500 字
5. 用中文输出

请生成报告：
"""

    def generate_report(
            self,
            case_id: str,
            ai_predictions: Dict,
            exam_type: str = "头部CT平扫",
            exam_time: str = None
    ) -> str:
        """
        生成医学报告

        Args:
            case_id: 病例ID
            ai_predictions: AI预测结果
            exam_type: 检查类型
            exam_time: 检查时间

        Returns:
            report: 生成的报告文本
        """
        if exam_time is None:
            exam_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 格式化 AI 发现
        ai_findings = self._format_ai_findings(ai_predictions)

        # 构建提示词
        prompt = self.report_template.format(
            case_id=case_id,
            exam_time=exam_time,
            exam_type=exam_type,
            ai_findings=ai_findings
        )

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的放射科医生，擅长撰写医学影像报告。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # 调用 LLM
        response = self.llm.chat_completion(
            messages=messages,
            temperature=0.5,  # 较低温度，保证专业性
            max_tokens=1000
        )

        # 提取报告内容
        report = response['choices'][0]['message']['content']

        return report

    def _format_ai_findings(self, predictions: Dict) -> str:
        """格式化 AI 预测结果为文本"""
        findings = []

        for label, prob in predictions.items():
            if prob >= 0.5:
                severity = "高度可能" if prob >= 0.8 else "中度可能" if prob >= 0.6 else "低度可能"
                findings.append(f"- {label}: {severity} (概率: {prob:.2%})")

        if not findings:
            findings.append("- 未发现明显异常征象")

        return "\n".join(findings)


class MedicalChatbot:
    """医疗对话机器人"""

    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService()

        # 系统提示词
        self.system_prompt = """你是一位专业的医疗助手，具有以下职责：

1. **解答患者疑问**：用通俗易懂的语言解释医学术语和检查结果
2. **提供健康建议**：基于循证医学给出合理建议
3. **心理支持**：给予患者适当的安慰和鼓励
4. **引导就医**：在必要时建议患者及时就医

**重要原则：**
- 不提供确诊，仅作为参考
- 遇到紧急情况立即建议就医
- 尊重患者隐私
- 语气亲切、专业、有同理心
- 避免使用过于专业的术语，必要时进行解释

**免责声明：**
我的回答仅供参考，不能替代专业医生的诊断和治疗建议。如有不适，请及时就医。"""

        # 对话历史管理
        self.conversation_history = {}

    def chat(
            self,
            user_id: str,
            message: str,
            context: Dict = None,
            conversation_id: str = None
    ) -> Dict:
        """
        对话接口

        Args:
            user_id: 用户ID
            message: 用户消息
            context: 上下文信息（如患者病历、检查结果）
            conversation_id: 会话ID

        Returns:
            response: 对话响应
        """
        # 初始化或获取会话历史
        if conversation_id is None:
            conversation_id = f"{user_id}_{int(time.time())}"

        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        history = self.conversation_history[conversation_id]

        # 添加上下文信息
        context_info = ""
        if context:
            if 'diagnosis' in context:
                context_info += f"\n**患者诊断信息：**\n{context['diagnosis']}\n"
            if 'lab_results' in context:
                context_info += f"\n**检查结果：**\n{context['lab_results']}\n"

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt + context_info}
        ]

        # 添加历史对话（最近10轮）
        for msg in history[-10:]:
            messages.append(msg)

        # 添加当前消息
        messages.append({"role": "user", "content": message})

        # 调用 LLM
        start_time = time.time()
        response = self.llm.chat_completion(
            messages=messages,
            temperature=0.8,
            max_tokens=500
        )
        response_time = int((time.time() - start_time) * 1000)

        # 提取回复
        assistant_message = response['choices'][0]['message']
        reply = assistant_message['content']

        # 更新历史
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        # 限制历史长度
        if len(history) > 20:
            self.conversation_history[conversation_id] = history[-20:]

        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "tokens_used": response.get('usage', {}).get('total_tokens', 0),
            "response_time_ms": response_time
        }

    def clear_history(self, conversation_id: str):
        """清除会话历史"""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 测试报告生成
    print("=" * 60)
    print("测试医学报告生成")
    print("=" * 60)

    llm = LLMService()
    generator = MedicalReportGenerator(llm)

    ai_results = {
        "ICH": 0.85,
        "IPH": 0.78,
        "SDH": 0.12,
        "Fracture": 0.05,
        "MassEffect": 0.45
    }

    report = generator.generate_report(
        case_id="CQ500CT100",
        ai_predictions=ai_results,
        exam_type="头部CT平扫"
    )

    print("\n生成的报告：")
    print(report)

    # 测试对话
    print("\n" + "=" * 60)
    print("测试医疗对话")
    print("=" * 60)

    chatbot = MedicalChatbot(llm)

    response = chatbot.chat(
        user_id="patient_001",
        message="医生说我脑出血了，严重吗？需要注意什么？",
        context={
            "diagnosis": "脑实质出血（右侧基底节区），出血量约15ml"
        }
    )

    print(f"\nAI 回复：")
    print(response['reply'])
    print(f"\n耗时: {response['response_time_ms']}ms")
