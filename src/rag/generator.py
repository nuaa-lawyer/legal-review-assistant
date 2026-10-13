"""
LLM生成模块：调用DeepSeek API，基于检索到的法条生成合规审查报告。

核心功能：
1. 将检索结果格式化为Prompt
2. 调用LLM生成结构化合规报告
3. 支持连续追问（带对话历史）
"""
from openai import OpenAI

from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from src.utils.prompts import COMPLIANCE_REVIEW_PROMPT, FOLLOWUP_PROMPT
from src.utils.usage import record
from src.rag.vector_store import search


def _get_llm_client() -> OpenAI:
    """获取DeepSeek API客户端（兼容OpenAI接口）"""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "请先配置 DEEPSEEK_API_KEY！\n"
            "1. 访问 https://platform.deepseek.com 注册\n"
            "2. 复制 .env.template 为 .env\n"
            "3. 填入你的API Key"
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _format_articles(articles: list[dict]) -> str:
    """将检索到的条款格式化为Prompt可用的文本"""
    formatted = []
    for i, art in enumerate(articles, 1):
        formatted.append(
            f"### 相关法条{i}\n"
            f"**法规**: {art['law_name']}\n"
            f"**条款**: {art['article_id']}\n"
            f"**内容**: {art['text']}\n"
        )
    return "\n".join(formatted)


def generate_report(user_input: str) -> str:
    """
    生成合规审查报告。

    完整流程:
    1. 语义检索相关法条（从ChromaDB）
    2. 将法条 + 用户输入 填入Prompt
    3. 调用DeepSeek生成结构化报告

    参数:
        user_input: 用户输入的产品功能描述

    返回:
        结构化的合规审查报告（Markdown格式）
    """
    print(f"\n{'='*60}")
    print(f"收到审查请求: {user_input}")

    # Step 1: 检索
    print("正在检索相关法规条款...")
    articles = search(user_input, top_k=5)
    print(f"检索到 {len(articles)} 条相关条款:")
    for a in articles:
        print(f"  - {a['law_name']} {a['article_id']} (相关性: {a['relevance_score']})")

    # Step 2: 构建Prompt
    formatted_articles = _format_articles(articles)
    prompt = COMPLIANCE_REVIEW_PROMPT.format(
        user_input=user_input,
        retrieved_articles=formatted_articles,
    )

    # Step 3: 调用LLM
    print("正在生成合规报告...")
    client = _get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名资深法律合规专家，擅长分析产品功能的法律合规风险。请使用Markdown格式输出分析报告。"
            },
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=4096,
    )

    report = response.choices[0].message.content
    record(response)

    # 统计Token用量
    usage = response.usage
    if usage:
        cost = (usage.prompt_tokens * 1 + usage.completion_tokens * 2) / 1_000_000
        print(f"Token用量: 输入{usage.prompt_tokens} + 输出{usage.completion_tokens}")
        print(f"预估费用: ¥{cost:.4f} (DeepSeek价格极低)")

    return report


def generate_followup(
    original_input: str,
    followup_question: str,
    chat_history: list[dict],
) -> str:
    """
    处理追问：基于原始问题和对话历史，重新检索并生成回答。

    参数:
        original_input: 最初的产品功能描述
        followup_question: 用户的追问
        chat_history: 之前的对话记录 [{"role": "user/assistant", "content": "..."}]
    """
    # 将对话历史转为文本
    history_text = ""
    for msg in chat_history[-6:]:  # 只保留最近3轮
        role = "用户" if msg["role"] == "user" else "专家"
        history_text += f"{role}: {msg['content']}\n"

    # 用追问 + 原始描述一起检索
    search_query = f"{original_input} {followup_question}"
    articles = search(search_query, top_k=5)
    formatted_articles = _format_articles(articles)

    prompt = FOLLOWUP_PROMPT.format(
        chat_history=history_text,
        original_input=original_input,
        followup_question=followup_question,
        retrieved_articles=formatted_articles,
    )

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名资深法律合规专家。针对用户对产品方案的追问给出专业分析。"
            },
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=4096,
    )

    record(response)
    return response.choices[0].message.content


if __name__ == "__main__":    # 命令行快速测试
    test_input = "我们的社交App允许用户查看附近100米内的其他用户位置信息"
    report = generate_report(test_input)
    print(f"\n{'='*60}")
    print(report)
