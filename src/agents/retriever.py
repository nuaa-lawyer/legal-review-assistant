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
from src.utils.usage import record
from src.rag.vector_store import search, get_articles_by_keys
from src.agents.knowledge_rules import mandatory_pairs, is_low_value

# 每路检索召回条数 / 候选上限 / 规则强制召回上限 / 最终依据条款上限 / 至少保留的语义条款数
TOP_K_PER_QUERY = 8
MAX_CANDIDATES = 15
MAX_MANDATORY = 12
MAX_FINAL_ARTICLES = 14
MIN_SEMANTIC_ARTICLES = 3

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
        record(response)
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

        # Step 2: 多路语义检索
        all_articles = {}  # 用 key 去重
        for query in search_queries:
            results = search(query, top_k=TOP_K_PER_QUERY)
            for r in results:
                # 过滤立法目的/适用范围类低价值条款，避免挤占名额
                if is_low_value(r["law_name"], r["article_id"]):
                    continue
                key = f"{r['law_name']}_{r['article_id']}"
                if key not in all_articles or r['relevance_score'] > all_articles[key]['relevance_score']:
                    all_articles[key] = r

        # 按相关性排序，取 Top-15 候选
        candidates = sorted(
            all_articles.values(),
            key=lambda x: x['relevance_score'],
            reverse=True
        )[:MAX_CANDIDATES]

        print(f"  Agent 2: 语义检索到 {len(candidates)} 条候选法条")

        # Step 2b: 规则强制召回（行业知识注入，与语义检索构成混合召回）
        mandatory = mandatory_pairs(classification, user_input)
        rule_articles = []
        if mandatory:
            fetched = get_articles_by_keys(mandatory)
            for key in mandatory:
                art = fetched.get(key)
                if not art:
                    continue
                art = dict(art)
                art["relevance_score"] = 1.0
                art["relevance_reason"] = "按风险领域/行业规则强制召回的核心条款"
                art["source"] = "rule"
                rule_articles.append(art)
                all_articles.pop(f"{art['law_name']}_{art['article_id']}", None)
            rule_articles = rule_articles[:MAX_MANDATORY]
            print(f"  Agent 2: 规则强制召回 {len(rule_articles)} 条核心条款"
                  f"（风险领域: {', '.join(classification.get('risk_areas') or [])}）")

        # Step 3: LLM 精选 + 适用性分析（候选 = 语义候选 + 规则强制召回条款）
        candidates_text = "\n\n".join([
            f"[{i+1}] {c['law_name']} {c['article_id']}\n{c['text'][:500]}"
            for i, c in enumerate(candidates)
        ])

        rule_text = "\n\n".join([
            f"[R{i+1}] {r['law_name']} {r['article_id']}\n{r['text'][:500]}"
            for i, r in enumerate(rule_articles)
        ])

        selection_prompt = f"""用户功能: {user_input}

风险领域: {', '.join(classification.get('risk_areas', []) or [])}
所属行业: {classification.get('industry', '未识别')}

【语义检索候选法条】
{candidates_text}

【规则强制召回的核心条款（这些是法律判断上必须纳入的条款，不得剔除）】
{rule_text if rule_text else '（无）'}

任务：
1. 从【语义检索候选法条】中选出最适用的 2-4 条（index 用数字编号）
2. 为【规则强制召回的核心条款】逐条写一句"适用原因"（说明该条款要求什么、而当前功能违反/满足什么）

输出JSON:
{{"selected": [{{"index": 编号, "reason": "该条款要求...而当前功能...因此适用"}}],
 "rule_reasons": [{{"law": "法规全名", "article": "第X条", "reason": "..."}}]}}

注意：只引用确实与当前功能相关的条款，不得编造条文内容。"""

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
        record(response)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        selected = parsed.get("selected") or [
            {"index": i + 1, "reason": "适用于当前场景"} for i in range(min(5, len(candidates)))
        ]
        rule_reasons = {}
        for item in parsed.get("rule_reasons") or []:
            if isinstance(item, dict) and item.get("article"):
                rule_reasons[(item.get("law", ""), item["article"])] = item.get("reason", "")

        # 组装最终结果：规则强制召回条款优先纳入，再补语义精选条款
        # 预留语义条款名额，避免规则条款把"场景特异性"条款全部挤掉
        final_articles, seen = [], set()

        rule_budget = max(1, MAX_FINAL_ARTICLES - MIN_SEMANTIC_ARTICLES)
        for art in rule_articles[:rule_budget]:
            reason = rule_reasons.get((art["law_name"], art["article_id"])) \
                or "该条款属于本场景的核心监管要求，需重点核对"
            item = dict(art)
            item["relevance_reason"] = reason
            seen.add((item["law_name"], item["article_id"]))
            final_articles.append(item)

        semantic_added = 0
        for sel in selected[:6]:
            idx = sel.get("index", 0) - 1
            if not (0 <= idx < len(candidates)):
                continue
            art = candidates[idx].copy()
            key = (art["law_name"], art["article_id"])
            if key in seen:
                continue
            seen.add(key)
            art["relevance_reason"] = sel.get("reason", "")
            art["source"] = "semantic"
            final_articles.append(art)
            semantic_added += 1
            if len(final_articles) >= MAX_FINAL_ARTICLES:
                break

        n_rule = len(final_articles) - semantic_added
        print(f"  Agent 2: 最终法条 {len(final_articles)} 条（规则召回 {n_rule} + 语义精选 {semantic_added}）")
        return final_articles
