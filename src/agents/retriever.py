"""
Agent 2: 法规检索器 (LegalRetriever)

职责：
- 接收分类Agent的输出（数据类型、风险领域等）
- 根据分类结果生成多角度的检索查询
- 每个查询独立检索向量库
- 合并去重，选出最相关的Top-5法条
- 对每条法条给出"适用性分析"（为什么这条法条适用于当前场景）
"""
from openai import OpenAI
from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from src.rag.vector_store import search

SYSTEM_PROMPT = """你是一名法律检索专家。根据风险分类结果，你需要：

1. 生成2-3个不同角度的检索查询语句（用于语义搜索法条库）
2. 对检索到的每条法条，判断是否真正适用于当前场景
3. 对适用的法条，用一句话说明"为什么适用"

请以JSON格式输出检索查询列表：
{"queries": ["查询角度1", "查询角度2", "查询角度3"]}

查询应该从不同角度覆盖风险领域，用语要接近法律条文的表述风格。
优先覆盖该产品所属行业的专门监管法规名称，确保检索到行业主依据法条。"""


class LegalRetriever:
    """法规检索Agent"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def retrieve(self, user_input: str, classification: dict) -> list[dict]:
        """
        基于分类结果，多角度检索并精选法条。

        流程:
        1. 让LLM生成2-3个检索查询
        2. 每个查询独立检索向量库
        3. 合并去重
        4. 让LLM精选Top-5 + 写适用性分析

        返回:
        [
            {
                "law_name": "...",
                "article_id": "第X条",
                "text": "条款原文...",
                "relevance_reason": "该条款要求...，而当前功能...",
                "relevance_score": 0.87
            },
            ...
        ]
        """
        # Step 1: 生成多角度检索查询
        query_prompt = f"""用户输入: {user_input}

分类结果:
- 所属行业: {classification.get('industry', '未识别')}
- 行业专门法规: {', '.join(classification.get('industry_regulations', []))}
- 数据类型: {', '.join(classification.get('data_types', []))}
- 处理行为: {', '.join(classification.get('processing_actions', []))}
- 风险领域: {', '.join(classification.get('risk_areas', []))}

请生成2-3个检索查询，其中至少一个查询要包含上述行业专门法规名称。"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        # 解析查询列表
        import json
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            queries_data = json.loads(raw)
            search_queries = queries_data.get("queries", [user_input])
        except json.JSONDecodeError:
            search_queries = [user_input]

        if not search_queries:
            search_queries = [user_input]

        # 行业路由：优先用行业专门法规名称检索，强制匹配行业主依据法条
        industry_regs = classification.get('industry_regulations', [])
        if isinstance(industry_regs, list):
            for reg in industry_regs:
                if reg and str(reg) not in search_queries:
                    search_queries.insert(0, str(reg))

        print(f"  Agent 2: 生成 {len(search_queries)} 个检索查询")
        for q in search_queries:
            print(f"    - {q}")

        # Step 2: 多路检索
        all_articles = {}  # 用 key 去重
        for query in search_queries:
            results = search(query, top_k=5)
            for r in results:
                key = f"{r['law_name']}_{r['article_id']}"
                if key not in all_articles or r['relevance_score'] > all_articles[key]['relevance_score']:
                    all_articles[key] = r

        # 按相关性排序，取 Top-10 候选
        candidates = sorted(
            all_articles.values(),
            key=lambda x: x['relevance_score'],
            reverse=True
        )[:10]

        print(f"  Agent 2: 检索到 {len(candidates)} 条候选法条")

        # Step 3: LLM 精选 + 适用性分析
        candidates_text = "\n\n".join([
            f"[{i+1}] {c['law_name']} {c['article_id']}\n{c['text'][:500]}"
            for i, c in enumerate(candidates)
        ])

        selection_prompt = f"""用户功能: {user_input}

风险领域: {', '.join(classification.get('risk_areas', []))}

候选法条:
{candidates_text}

请从候选法条中选出最适用的3-5条，对每条写一句"适用原因"。

输出JSON:
{{"selected": [
    {{"index": 编号, "reason": "该条款要求...而当前功能...因此适用"}},
    ...
]}}"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是法律检索专家，精确匹配法条与场景。"},
                {"role": "user", "content": selection_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            selected = json.loads(raw).get("selected", [])
        except json.JSONDecodeError:
            selected = [{"index": i+1, "reason": "适用于当前场景"} for i in range(min(5, len(candidates)))]

        # 组装最终结果
        final_articles = []
        for sel in selected[:5]:
            idx = sel["index"] - 1
            if 0 <= idx < len(candidates):
                art = candidates[idx].copy()
                art["relevance_reason"] = sel.get("reason", "")
                final_articles.append(art)

        print(f"  Agent 2: 精选 {len(final_articles)} 条法条")
        return final_articles
