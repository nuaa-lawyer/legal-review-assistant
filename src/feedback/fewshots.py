"""
反馈闭环 · few-shot 提供层

把"用户认为准确且被采纳"的历史案例作为报告 Agent 的判断口径参考。
数据来源：feedback/feedback.jsonl（rating=up 且有采纳法条）+ feedback/reviews.jsonl（原始输入）

设计原则：
- 只取高质量样本（rating=up），最多 N 条，避免 prompt 膨胀
- 任何异常都不影响主流程（反馈缺失时返回空列表）
"""
import json
import os

from src.feedback.store import FEEDBACK_PATH, REVIEWS_PATH, FEWSHOTS_PATH


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def get_fewshots(limit: int = 3) -> list[dict]:
    """
    返回可直接注入 prompt 的 few-shot 案例：
    [{"input": ..., "articles": "法规 条款", "advice": "用户认可的整改要点"}, ...]
    """
    # 优先读取人工/自动提炼好的 few_shots.json
    if os.path.exists(FEWSHOTS_PATH):
        try:
            with open(FEWSHOTS_PATH, "r", encoding="utf-8") as f:
                prepared = json.load(f)
            if prepared:
                return prepared[:limit]
        except (json.JSONDecodeError, OSError):
            pass

    reviews = {r["review_id"]: r for r in _load_jsonl(REVIEWS_PATH)}
    candidates = []
    for fb in reversed(_load_jsonl(FEEDBACK_PATH)):
        if fb.get("rating") != "up":
            continue
        rev = reviews.get(fb.get("review_id"))
        if not rev:
            continue
        articles = fb.get("adopted_articles") or rev.get("articles") or []
        comment = (fb.get("comment") or "").strip()
        candidates.append({
            "input": rev.get("input", ""),
            "articles": "、".join(articles[:5]),
            "advice": comment or "（用户未填写具体意见，但确认该份报告准确可用）",
        })
        if len(candidates) >= limit:
            break
    return candidates
