"""
Agent 1: 风险分类器 (RiskClassifier)

职责：
- 分析用户输入的产品功能描述
- 识别涉及的数据类型、处理行为、业务场景
- 给出初步风险等级判断
- 输出结构化JSON，供下游Agent使用

这个Agent不检索法条，只做"理解输入"这一件事。
"""
import json
from openai import OpenAI
from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """你是一名数据合规分类专家。你的任务是将产品功能描述分解为结构化的合规要素。

请分析用户描述的产品功能，识别以下内容：

1. **data_types**: 涉及的数据类型
   可选值: 个人身份信息、位置信息、生物识别信息、健康数据、金融信息、
          通信内容、设备信息、行为日志、未成年人信息、其他

2. **processing_actions**: 数据处理行为
   可选值: 收集、存储、使用、加工、传输、提供、公开、跨境传输、删除

3. **scenario_tags**: 业务场景标签
   可选值: 社交网络、电商交易、金融支付、医疗健康、在线教育、
          未成年人相关、位置服务、AI/算法推荐、物联网、企业服务

4. **risk_areas**: 可能涉及的合规风险领域
   可选值: 告知同意缺陷、最小必要超标、敏感信息违规、数据跨境风险、
          未成年人保护不足、数据安全措施不足、算法合规风险、第三方共享风险

5. **initial_risk_level**: 初步风险等级 (high / medium / low)

6. **risk_summary**: 30字以内的风险概况描述

请以JSON格式输出，不要输出任何其他内容。"""


class RiskClassifier:
    """风险分类Agent"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def classify(self, user_input: str) -> dict:
        """
        分析用户输入，返回结构化的分类结果。

        返回示例:
        {
            "data_types": ["位置信息", "个人身份信息"],
            "processing_actions": ["收集", "存储", "使用"],
            "scenario_tags": ["社交网络", "位置服务"],
            "risk_areas": ["告知同意缺陷", "敏感信息违规"],
            "initial_risk_level": "high",
            "risk_summary": "实时位置共享涉及敏感个人信息，需严格合规"
        }
        """
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,  # 极低温度，确保分类稳定一致
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # 容错：如果JSON解析失败，返回默认值
            result = {
                "data_types": [],
                "processing_actions": [],
                "scenario_tags": [],
                "risk_areas": [],
                "initial_risk_level": "medium",
                "risk_summary": "分类解析异常，使用默认配置",
            }

        return result
