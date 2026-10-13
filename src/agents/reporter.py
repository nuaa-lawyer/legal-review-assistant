"""
Agent 3: 报告生成器 (ReportGenerator)

职责：
- 接收分类结果 + 精选法条
- 生成结构化合规风险报告
- 输出Markdown格式（高风险项、中风险项、合规项、修改建议、总结）

这个Agent只负责"写报告"，不自己做检索或分类。
"""
from datetime import date
from openai import OpenAI
from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from src.utils.usage import record

SYSTEM_PROMPT = """你是一名资深法律合规专家，精通中国个人信息保护、数据安全、网络安全法律法规，以及各行业的专门监管法规。

你的任务是基于已有的风险分类结果和相关法条，生成一份结构化合规风险报告。
你不是"检索工具"，你输出的是一份可以直接支撑产品决策的审查结论。

## 报告格式要求
报告最顶部必须先用一行标注「审查日期」，该日期必须使用用户消息中提供的系统当前日期，严禁自行编造其他日期。

### ⚖️ 合规结论
必须输出三行：
- **风险等级**：高 / 中 / 低（与下方风险项一致）
- **上线建议**：三选一 —— 「可直接上线」/「整改后可上线」/「不建议上线」
- **结论一句话**：说明最关键的一个原因

### 🔴 高风险项
使用表格：| 风险点 | 违反法规 | 具体条款 | 风险说明 |

### 🟡 中风险项
使用表格：| 风险点 | 违反法规 | 具体条款 | 风险说明 |

### 🟢 合规项
列出已符合法规要求的设计点（如有）

### 🛠️ 整改清单
使用表格，按优先级排序：| 优先级 | 整改项 | 具体操作 | 法规依据 |
- 优先级取值：P0（上线前必须改） / P1（上线后尽快改） / P2（优化项）
- 「具体操作」必须是可以直接交给产品/研发执行的动宾短语，例如「将实时位置改为模糊距离显示」
- 每一条整改项都必须对应上面的风险项和法条，不得凭空新增

### ⚠️ 总结
2-3句话总结核心风险和最重要的改进方向

## 行业路由规则（重要）
- 先识别产品所属行业，以该行业的专门监管法规作为主要依据（主依据）。
- 《个人信息保护法》《消费者权益保护法》等通用法只能作为兜底或补充，不得作为唯一依据。
- 示例：教育类App → 以教育App备案、双减政策、《未成年人网络保护条例》、《互联网信息服务算法推荐管理规定》为主；化妆品 → 以《化妆品监督管理条例》《广告法》为主。
- 若检索到的法条中缺少行业专门法规，必须在报告中明确指出「该行业专门法规尚未覆盖，建议补充审查」，不得用通用法强行替代。

## 注意事项
- 每项风险必须有明确的法规依据
- 不可编造法条，只能引用提供的条款
- **禁止用「立法目的」「适用范围」类条款（如第一条、第二条）作为风险依据或整改依据**，
  这类条款只讲立法目的，不能说明具体行为规范；请改用具体行为条款（例如最小必要对应第六条、
  单独同意对应第二十九条、告知对应第十七条）
- 每条风险点引用的条款必须与风险行为语义对应，不得把"沾边"的条款堆上去充数
- 修改建议要具体可操作
- 使用Markdown格式
- 如果某条款不适用当前场景，不要强行引用"""


def _fewshot_block() -> str:
    """
    读取已被用户采纳的历史审查案例，作为风格与判断口径的参考（人机协同的持续学习环节）。
    反馈数据缺失时静默跳过，不影响主流程。
    """
    try:
        from src.feedback.fewshots import get_fewshots
        shots = get_fewshots(limit=3)
    except Exception:
        return ""
    if not shots:
        return ""
    parts = []
    for s in shots:
        parts.append(
            f"- 历史场景：{s['input']}\n"
            f"  已采纳的关键法条：{s.get('articles', '')}\n"
            f"  已采纳的整改要点：{s.get('advice', '')}"
        )
    return "\n## 历史已采纳案例（同类场景的判断口径参考，不要直接抄写）\n" + "\n".join(parts) + "\n"


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
        today = date.today().strftime("%Y年%m月%d日")

        # 格式化分类结果
        class_text = f"""- 所属行业: {classification.get('industry', '未识别')}
- 数据类型: {', '.join(classification.get('data_types', []))}
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

        prompt = f"""## 审查日期（必须使用此日期）
{today}

## 用户输入的产品功能描述
{user_input}

## 风险分类结果
{class_text}

## 相关法律法规条款
{articles_text}
{_fewshot_block()}
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

        # 统计Token用量（全局累加器，供成本核算使用）
        record(response)
        usage = response.usage
        if usage:
            cost = (usage.prompt_tokens + usage.completion_tokens * 2) / 1_000_000
            print(f"  Agent 3: 生成报告完成 (费用约 ¥{cost:.4f})")

        return report
