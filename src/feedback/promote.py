"""
反馈闭环 · 价值转化层（promote）

把收集到的反馈变成两类可复用资产，这是"越用越聪明"的实际动作：

1. 👎 不准确的报告 → 自动追加为新评测用例（eval/cases_feedback.jsonl）
   下次改 prompt / 换模型时，这批真实 bad case 会立刻把回归问题暴露出来。

2. 👍 准确且被采纳的报告 → 提炼为 few-shot 示例（feedback/few_shots.json）
   作为后续审查的判断口径参考。

用法：
    python -m src.feedback.promote --dry-run     # 只看会生成什么
    python -m src.feedback.promote               # 实际写入
"""
import argparse
import json
import os
from datetime import datetime

from src.feedback.store import FEEDBACK_PATH, REVIEWS_PATH, FEWSHOTS_PATH, FEEDBACK_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_FEEDBACK_CASES = os.path.join(BASE_DIR, "eval", "cases_feedback.jsonl")


def _load(path):
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


def build():
    reviews = {r["review_id"]: r for r in _load(REVIEWS_PATH)}
    feedback = _load(FEEDBACK_PATH)

    new_cases, few_shots = [], []
    existing_bad_ids = {c.get("from_review") for c in _load(EVAL_FEEDBACK_CASES)}

    for fb in feedback:
        rev = reviews.get(fb.get("review_id"))
        if not rev:
            continue

        if fb.get("rating") == "down" and rev["review_id"] not in existing_bad_ids:
            new_cases.append({
                "id": f"FB{rev['review_id'][:6]}",
                "from_review": rev["review_id"],
                "industry": (rev.get("classification") or {}).get("industry", "其他"),
                "scenario": "用户反馈不准确",
                "risk_level": "unknown",
                "input": rev["input"],
                "must_hit": [],
                "key_points": [],
                "user_comment": fb.get("comment", ""),
                "system_cited": rev.get("articles", []),
                "status": "待人工补标注 ground truth",
            })

        if fb.get("rating") == "up":
            few_shots.append({
                "from_review": rev["review_id"],
                "input": rev["input"],
                "articles": "、".join((fb.get("adopted_articles") or rev.get("articles", []))[:5]),
                "advice": (fb.get("comment") or "").strip() or "（用户确认报告准确可用）",
                "added_at": datetime.now().isoformat(timespec="seconds"),
            })

    return new_cases, few_shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_cases, few_shots = build()
    print(f"👎 新增待标注 bad case: {len(new_cases)} 条")
    for c in new_cases:
        print(f"   - {c['id']} | {c['input'][:40]}")
    print(f"👍 可提炼 few-shot 示例: {len(few_shots)} 条")

    if args.dry_run:
        print("\n（dry-run，未写入）")
        return

    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if new_cases:
        with open(EVAL_FEEDBACK_CASES, "a", encoding="utf-8") as f:
            for c in new_cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if few_shots:
        # 去重后覆盖写入（保留最近 20 条）
        seen, merged = set(), []
        for s in few_shots:
            if s["input"] in seen:
                continue
            seen.add(s["input"])
            merged.append(s)
        with open(FEWSHOTS_PATH, "w", encoding="utf-8") as f:
            json.dump(merged[:20], f, ensure_ascii=False, indent=2)

    print(f"\n已写入：\n  {EVAL_FEEDBACK_CASES}\n  {FEWSHOTS_PATH}")


if __name__ == "__main__":
    main()
