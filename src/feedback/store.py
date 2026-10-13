"""
反馈闭环 · 数据存储层

持续学习机制的"数据入口"：每一次审查与每一条反馈都落盘，
再由 promote.py 转化为 few-shot 示例 / 新评测用例，形成"越用越准"的闭环。

数据文件（feedback/ 目录）：
  reviews.jsonl   每次审查的完整快照（输入、分类、命中法条、报告、耗时、成本）
  feedback.jsonl  用户反馈事件（rating、采纳法条、文字意见）
  few_shots.json  由高价值反馈提炼的 few-shot 示例（供报告 Agent 参考）
  update_log.jsonl 知识库变更记录（法规新增/重建索引）
"""
import json
import os
import threading
import uuid
from datetime import datetime

FEEDBACK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "feedback"
)
REVIEWS_PATH = os.path.join(FEEDBACK_DIR, "reviews.jsonl")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
FEWSHOTS_PATH = os.path.join(FEEDBACK_DIR, "few_shots.json")
UPDATE_LOG_PATH = os.path.join(FEEDBACK_DIR, "update_log.jsonl")

_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _append(path: str, record: dict) -> None:
    _ensure_dir()
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_all(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ==================== 写入 ====================

def log_review(user_input: str, result: dict, mode: str = "multi_agent") -> str:
    """记录一次审查（review_id 用于后续反馈关联）"""
    review_id = uuid.uuid4().hex[:12]
    articles = result.get("articles") or []
    _append(REVIEWS_PATH, {
        "review_id": review_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "input": user_input,
        "classification": result.get("classification") or {},
        "articles": [
            f"{a.get('law_name')} {a.get('article_id')}" for a in articles
        ],
        "report": result.get("report", ""),
        "total_elapsed": result.get("total_elapsed"),
        "usage": result.get("usage") or {},
    })
    return review_id


def log_feedback(review_id: str, rating: str, adopted_articles=None,
                 comment: str = "", source: str = "local_web") -> dict:
    """
    记录一条用户反馈。

    rating: "up"（准确有用）/ "down"（不准确）/ "neutral"
    adopted_articles: 用户实际采纳的法条列表（体现"哪些法条真正有用"）
    """
    record = {
        "feedback_id": uuid.uuid4().hex[:12],
        "review_id": review_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "adopted_articles": adopted_articles or [],
        "comment": comment,
        "source": source,
    }
    _append(FEEDBACK_PATH, record)
    return record


def log_kb_update(event: dict) -> None:
    """记录知识库变更（新增法规、重建索引），作为"持续学习"的可核查证据"""
    event = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    _append(UPDATE_LOG_PATH, event)


# ==================== 读取 / 统计 ====================

def list_reviews(limit: int = 0) -> list:
    rows = _read_all(REVIEWS_PATH)
    return rows[-limit:] if limit else rows


def list_feedback(limit: int = 0) -> list:
    rows = _read_all(FEEDBACK_PATH)
    return rows[-limit:] if limit else rows


def get_review(review_id: str) -> dict:
    for r in _read_all(REVIEWS_PATH):
        if r["review_id"] == review_id:
            return r
    return {}


def stats() -> dict:
    """反馈统计：用于首页展示"越用越准"的运营指标"""
    reviews = _read_all(REVIEWS_PATH)
    feedback = _read_all(FEEDBACK_PATH)
    up = sum(1 for f in feedback if f.get("rating") == "up")
    down = sum(1 for f in feedback if f.get("rating") == "down")
    rated = up + down
    adopted = sum(len(f.get("adopted_articles") or []) for f in feedback)

    # 采纳率：按周统计，用于展示趋势
    by_week = {}
    for f in feedback:
        if f.get("rating") not in ("up", "down"):
            continue
        week = f["ts"][:10]
        w = by_week.setdefault(week, {"up": 0, "down": 0})
        w[f["rating"]] += 1

    return {
        "reviews": len(reviews),
        "feedback_count": len(feedback),
        "rated": rated,
        "up": up,
        "down": down,
        "satisfaction": round(up / rated, 3) if rated else None,
        "adopted_articles": adopted,
        "by_date": by_week,
    }
