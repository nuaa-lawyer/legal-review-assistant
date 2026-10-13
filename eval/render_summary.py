"""
评测汇总重算工具（不调用任何 API）

用途：口径调整或需要重新排版时，直接基于已保存的 metrics.json 重算汇总表，
不必重跑几十分钟的评测。

用法：
    python eval/render_summary.py                       # 用最新一次结果
    python eval/render_summary.py eval/results/20261013_115747
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def latest_result_dir() -> str:
    root = os.path.join(BASE_DIR, "eval", "results")
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    if not dirs:
        raise SystemExit("没有找到任何评测结果目录")
    return max(dirs, key=os.path.getmtime)


def render(result_dir: str) -> str:
    with open(os.path.join(result_dir, "metrics.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    details = data.get("details", {})
    kb_note = data.get("summary", {})

    rows = []
    for name, records in details.items():
        valid = [r for r in records if "error" not in r]
        if not valid:
            continue
        # 统一按当前口径重算：可信 = 核心条款全覆盖 且 零编造
        # （提及库外法规属于「建议补充审查」，不影响可信度）
        for r in valid:
            r["fabricated"] = len(r.get("hallucinated") or [])
            r["unknown_law_count"] = len(r.get("unknown_laws") or [])
            r["trusted"] = (r.get("core_recall") == 1.0) and r["fabricated"] == 0
        label = kb_note.get(name, {}).get("label", name)
        rows.append({
            "label": label,
            "cases": len(valid),
            "core_recall": mean([r["core_recall"] for r in valid]),
            "full_recall": mean([r["recall"] for r in valid]),
            "avg_cited": round(mean([r["cited_count"] for r in valid]) or 0, 1),
            "precision": mean([r["citation_precision"] for r in valid]),
            "fabricated": round(mean([r["fabricated"] for r in valid]) or 0, 3),
            "no_fab_rate": mean([1 if not r["fabricated"] else 0 for r in valid]),
            "unknown_rate": mean([1 if r["unknown_law_count"] else 0 for r in valid]),
            "trusted_rate": mean([1 if r["trusted"] else 0 for r in valid]),
            "keypoint": mean([r["keypoint_coverage"] for r in valid]),
            "elapsed": mean([r["elapsed"] for r in valid]),
            "cost": round(mean([r["cost_cny"] for r in valid]) or 0, 5),
        })

    lines = ["# 法审助手 · 定量评测报告", "",
             "**指标口径**",
             "- 核心条款召回率：人工标注的每场景 2 条必备依据被正确引用的比例（准确性主线）",
             "- 全量条款召回率：标注的全部依据（含扩展条款）的命中比例",
             "- 平均引用条款数：报告实际引用的条款数量，与召回率同看，用于识别「堆条款充数」",
             "- 引用精准率：引用条款中属于标注依据的比例（越低说明冗余越多）",
             "- 编造条款/份：引用了知识库中不存在的条款号（真幻觉）",
             "- 无编造报告率：完全不存在编造条款的报告占比",
             "- 提及库外法规率：报告点名了知识库尚未收录的法规（属于「建议补充审查」，非幻觉，单独统计）",
             "- 可信报告率：核心条款全覆盖且零编造的报告占比（面向业务可用性的综合指标）", "",
             "| 配置 | 核心条款召回率 | 全量条款召回率 | 平均引用数 | 引用精准率 | 编造条款/份 | 无编造报告率 | 可信报告率 | 关键点覆盖率 | 平均耗时 | 平均成本 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['core_recall']} | {r['full_recall']} | {r['avg_cited']} | "
                     f"{r['precision']} | {r['fabricated']} | {r['no_fab_rate']} | "
                     f"{r['trusted_rate']} | {r['keypoint']} | {r['elapsed']}s | ¥{r['cost']} |")

    if cases:
        lines += ["", f"- 用例数：{len(cases)}，覆盖行业："
                      f"{', '.join(sorted({c['industry'] for c in cases}))}"]

    out = "\n".join(lines) + "\n"
    with open(os.path.join(result_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(out)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else latest_result_dir()
    print(f"重算目录: {target}\n")
    print(render(target))
