"""
Agent 3: 报告生成器 (ReportGenerator)

职责：
- 接收分类结果 + 精选法条
- 生成结构化合规风险报告
- 输出Markdown格式（高风险项、中风险项、合规项、修改建议、总结）

这个Agent只负责"写报告"，不自己做检索或分类。
"""
from openai import OpenAI
from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

SYSTEM_PROMPT = """你是一名资深法律合规专家，精通中国个人信息保护、数据安全、网络安全法律法规。

你的任务是基于已有的风险分类结果和相关法条，生成一份结构化的合规风险报告。

## 报告格式要求

### 🔴 高风险项
使用表格：| 风险点 | 违反法规 | 具体条款 | 风险说明 |

### 🟡 中风险项
使用表格：| 风险点 | 违反法规 | 具体条款 | 风险说明 |

### 🟢 合规项
列出已符合法规要求的设计点（如有）

### 📝 修改建议
每条建议包含：
- 问题描述
- 改进方案
- 法规依据

### ⚠️ 总结
2-3句话总结核心风险和最重要的改进方向

## 注意事项
- 每项风险必须有明确的法规依据
- 不可编造法条，只能引用提供的条款
- 修改建议要具体可操作
- 使用Markdown格式
- 如果某条款不适用当前场景，不要强行引用"""


class ReportGenerator:
    """报告生成Agent"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def generate(
        self,
        user_input: str,
        classification: dict,
        articles: list[dict],
    ) -> str:
        """
        生成最终的合规审查报告。

        参数:
            user_input: 用户原始输入
            classification: Agent 1 的分类结果
            articles: Agent 2 精选的法条列表
        """
        # 格式化分类结果
        class_text = f"""- 数据类型: {', '.join(classification.get('data_types', []))}
- 处理行为: {', '.join(classification.get('processing_actions', []))}
- 业务场景: {', '.join(classification.get('scenario_tags', []))}
- 风险领域: {', '.join(classification.get('risk_areas', []))}
- 初步风险等级: {classification.get('initial_risk_level', 'unknown')}
- 风险概况: {classification.get('risk_summary', '')}"""

        # 格式化法条
        articles_text = "\n\n".join([
            f"### 法条{i+1}\n"
            f"**法规**: {a['law_name']}\n"
            f"**条款**: {a['article_id']}\n"
            f"**内容**: {a['text'][:800]}\n"
            f"**适用原因**: {a.get('relevance_reason', '与当前场景相关')}"
            for i, a in enumerate(articles)
        ])

        prompt = f"""## 用户输入的产品功能描述
{user_input}

## 风险分类结果
{class_text}

## 相关法律法规条款
{articles_text}

请根据以上信息，生成合规审查报告。"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=4096,
        )

        report = response.choices[0].message.content

        # 统计Token用量
        usage = response.usage
        if usage:
            cost = (usage.prompt_tokens + usage.completion_tokens * 2) / 1_000_000
            print(f"  Agent 3: 生成报告完成 (费用约 ¥{cost:.4f})")

        return report
