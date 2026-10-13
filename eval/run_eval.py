"""
法审助手 - 定量评测脚本

对比三种配置，量化「精准决策提升水平」：
  baseline : 裸 DeepSeek-V3（无行业知识库、无检索、无 Agent 编排）
  single   : 单流水线 RAG（检索 Top-5 法条 → 一次性生成报告）
  multi    : 多 Agent 协作（分类 → 多路检索 → 精选 → 决策报告）

评测指标：
  1. 法条召回率      ground-truth 法条中被报告正确引用的比例（核心指标）
  2. 完整命中率      全部 ground-truth 法条都被引用的用例比例
  3. 幻觉引用        引用了知识库中不存在的条款 / 法规（越少越好，直接对应"杜绝幻觉"）
  4. 关键风险点覆盖率 报告是否覆盖标注的关键风险要点
  5. 高风险召回率     high 风险用例中，报告是否给出高风险结论
  6. 平均耗时 / 平均成本

用法：
    python eval/run_eval.py --limit 3                     # 冒烟测试
    python eval/run_eval.py --configs multi               # 只跑多Agent
    python eval/run_eval.py                               # 全量三种配置
    python eval/run_eval.py --ids C01,C04 --configs baseline,multi
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from eval.citation import load_kb, extract_citations, parse_cn_number  # noqa: E402
from src.utils.usage import measure  # noqa: E402


# ==================== 三种配置 ====================

BASELINE_SYSTEM = """你是一名资深法律合规专家。请对用户描述的产品功能进行合规风险审查，输出 Markdown 报告，包含：
### 🔴 高风险项（表格：| 风险点 | 违反法规 | 具体条款 | 风险说明 |）
### 🟡 中风险项（同上表格）
### 🛠️ 整改清单（表格：| 优先级 | 整改项 | 具体操作 | 法规依据 |）
### ⚠️ 总结
每项风险必须注明具体的法规名称和条款号。"""


def run_baseline(user_input: str) -> str:
    """裸大模型：不使用任何行业知识库与检索"""
    from openai import OpenAI
    from src.utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
    from src.utils.usage import record

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    record(resp)
    return resp.choices[0].message.content


def run_single(user_input: str) -> str:
    from src.rag.generator import generate_report
    return generate_report(user_input)


def run_multi(user_input: str) -> str:
    from src.agents.orchestrator import multi_agent_review
    return multi_agent_review(user_input, verbose=False)["report"]


CONFIGS = {
    "baseline": ("裸大模型（无知识库）", run_baseline),
    "single": ("单流水线 RAG", run_single),
    "multi": ("多 Agent 协作", run_multi),
}


# ==================== 指标计算 ====================

def _section(report: str, keyword: str) -> str:
    """取出报告中包含 keyword 的那一节内容"""
    lines = report.split("\n")
    out, capturing = [], False
    for line in lines:
        if line.strip().startswith("#") and keyword in line:
            capturing = True
            continue
        if capturing and line.strip().startswith("#"):
            break
        if capturing:
            out.append(line)
    return "\n".join(out)


def _table_rows(section_text: str) -> int:
    rows = 0
    for line in section_text.split("\n"):
        s = line.strip()
        if s.startswith("|") and not set(s) <= set("|-: "):
            rows += 1
    return max(0, rows - 1)  # 去掉表头


def score_case(case: dict, report: str, kb: dict, elapsed: float, usage: dict) -> dict:
    cite = extract_citations(report, kb)
    cited = set(cite["citations"])

    # ground truth：must_hit 前 2 条 = 核心依据（人工标注的必备条款），其余为扩展依据
    must = {(law, parse_cn_number(a.replace("第", "").replace("条", "")))
            for law, a in case["must_hit"]}
    core = {(law, parse_cn_number(a.replace("第", "").replace("条", "")))
            for law, a in case["must_hit"][:2]}

    hit = must & cited
    miss = must - cited
    core_hit = core & cited

    # 引用精准率：报告引用的条款里，有多少条属于人工标注的依据（衡量"堆条款充数"）
    citation_precision = round(len(hit) / len(cited), 3) if cited else None

    key_hit, key_total = 0, len(case["key_points"])
    for alternatives in case["key_points"]:
        if any(kw in report for kw in alternatives):
            key_hit += 1

    high_section = _section(report, "高风险")
    mid_section = _section(report, "中风险")
    high_rows = _table_rows(high_section) or high_section.count("\n- ")
    conclusion = _section(report, "合规结论") or report[:400]
    has_level_high = ("高" in conclusion.split("\n")[0] if conclusion else False)
    level_correct = None
    if case["risk_level"] == "high":
        level_correct = bool(high_rows > 0 or "高风险" in report and "无" not in high_section[:20])

    return {
        "id": case["id"],
        "industry": case["industry"],
        "scenario": case["scenario"],
        "core_recall": round(len(core_hit) / len(core), 3) if core else None,
        "recall": round(len(hit) / len(must), 3) if must else None,
        "complete_hit": len(miss) == 0,
        "cited_count": len(cited),
        "citation_precision": citation_precision,
        "fabricated": len(cite["hallucinated"]),          # 编造条款号（真幻觉）
        "unknown_law_count": len(cite["unknown_laws"]),   # 提及库外法规（提示补充审查，非幻觉）
        "trusted": (len(core_hit) == len(core)) and not cite["hallucinated"],
        "hits": [f"{l}{a}条" for l, a in sorted(hit)],
        "missing": [f"{l}{a}条" for l, a in sorted(miss)],
        "hallucinated": [f"{l}第{a}条" for l, a in cite["hallucinated"]],
        "unknown_laws": cite["unknown_laws"],
        "keypoint_coverage": round(key_hit / key_total, 3) if key_total else None,
        "high_risk_recall": int(bool(level_correct)) if level_correct is not None else None,
        "elapsed": round(elapsed, 2),
        "cost_cny": usage.get("cost_cny", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ==================== 主流程 ====================

def load_cases(path: str, ids: list, limit: int) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if limit:
        cases = cases[:limit]
    return cases


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(BASE_DIR, "eval", "cases.jsonl"))
    ap.add_argument("--configs", default="baseline,single,multi")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    kb = load_kb()
    cases = load_cases(args.cases, [i.strip() for i in args.ids.split(",") if i.strip()], args.limit)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in config_names:
        if c not in CONFIGS:
            raise SystemExit(f"未知配置: {c}，可选 {list(CONFIGS)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(BASE_DIR, "eval", "results", stamp)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)

    print(f"知识库: {kb['total_articles']} 条条款 / {len(kb['law_articles'])} 部法规")
    print(f"用例数: {len(cases)} | 配置: {config_names}")
    print(f"输出目录: {out_dir}\n")

    summary = {}
    all_rows = {}
    for name in config_names:
        label, fn = CONFIGS[name]
        rows = []
        print(f"{'='*60}\n[{name}] {label}\n{'='*60}")
        for case in cases:
            t0 = time.time()
            with measure() as usage:
                try:
                    report = fn(case["input"])
                except Exception as e:
                    print(f"  {case['id']} 失败: {e}")
                    rows.append({"id": case["id"], "error": str(e)})
                    continue
            elapsed = time.time() - t0
            row = score_case(case, report, kb, elapsed, usage)
            row["usage"] = usage
            rows.append(row)

            rdir = os.path.join(out_dir, "reports", name)
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, f"{case['id']}.md"), "w", encoding="utf-8") as f:
                f.write(f"# [{name}] {case['id']} {case['scenario']}\n\n"
                        f"**输入**: {case['input']}\n\n"
                        f"**ground truth**: {', '.join(l + a for l, a in case['must_hit'])}\n\n"
                        f"---\n\n{report}")

            print(f"  {case['id']} 核心召回={row['core_recall']} 全量召回={row['recall']} "
                  f"关键点={row['keypoint_coverage']} 幻觉={len(row['hallucinated'])} "
                  f"耗时={row['elapsed']}s ¥{row['cost_cny']}")

        valid = [r for r in rows if "error" not in r]
        all_rows[name] = valid
        summary[name] = {
            "label": label,
            "cases": len(valid),
            "core_recall": mean([r["core_recall"] for r in valid]),
            "avg_recall": mean([r["recall"] for r in valid]),
            "complete_hit_rate": mean([1 if r["complete_hit"] else 0 for r in valid]),
            "avg_cited": round(mean([r["cited_count"] for r in valid]) or 0, 1),
            "citation_precision": mean([r["citation_precision"] for r in valid]),
            "no_fabrication_rate": mean([1 if not r["fabricated"] else 0 for r in valid]),
            "unknown_law_rate": mean([1 if r["unknown_law_count"] else 0 for r in valid]),
            "trusted_rate": mean([1 if r["trusted"] else 0 for r in valid]),
            "avg_hallucinations": round(mean([r["fabricated"] for r in valid]) or 0, 3),
            "avg_keypoint_coverage": mean([r["keypoint_coverage"] for r in valid]),
            "high_risk_recall": mean([r["high_risk_recall"] for r in valid]),
            "avg_elapsed_s": mean([r["elapsed"] for r in valid]),
            "avg_cost_cny": round(mean([r["cost_cny"] for r in valid]) or 0, 5),
            "avg_tokens": round(mean([r["total_tokens"] for r in valid]) or 0, 1),
        }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "summary": summary, "details": all_rows},
                  f, ensure_ascii=False, indent=2)

    # 明细 CSV（合并三种配置）
    with open(os.path.join(out_dir, "detail.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "id", "scenario", "core_recall", "full_recall", "complete_hit",
                    "hallucinations", "keypoint_coverage", "high_risk_recall", "cited_count",
                    "elapsed_s", "cost_cny", "missing"])
        for name in config_names:
            for r in all_rows.get(name, []):
                w.writerow([name, r["id"], r["scenario"], r["core_recall"], r["recall"],
                            r["complete_hit"], len(r["hallucinated"]), r["keypoint_coverage"],
                            r["high_risk_recall"], r["cited_count"], r["elapsed"],
                            r["cost_cny"], ";".join(r["missing"])])

    # Markdown 汇总
    lines = ["# 法审助手 · 定量评测报告", "",
             f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"- 用例数：{len(cases)}（覆盖行业：{', '.join(sorted({c['industry'] for c in cases}))}）",
             f"- 知识库：{kb['total_articles']} 条条款 / {len(kb['law_articles'])} 部法规", "",
             "**指标口径**",
             "- 核心条款召回率：人工标注的每场景 2 条必备依据被正确引用的比例（准确性主线）",
             "- 全量条款召回率：标注的全部依据（含扩展条款）命中比例",
             "- 平均引用条款数：报告实际引用的条款数量 —— 与召回率同看，用于识别「堆条款充数」",
             "- 引用精准率：引用条款中属于标注依据的比例（越低说明冗余越多）",
             "- 幻觉引用/份：引用了知识库中不存在的条款或法规（编造法条）",
             "- 可信报告率：核心条款全覆盖且零幻觉的报告占比（面向业务可用性的综合指标）", "",
             "| 配置 | 核心条款召回率 | 全量条款召回率 | 平均引用数 | 引用精准率 | 幻觉引用/份 | 零幻觉报告率 | 可信报告率 | 关键点覆盖率 | 平均耗时 | 平均成本 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {s['label']} | {s['core_recall']} | {s['avg_recall']} | {s['avg_cited']} | "
                     f"{s['citation_precision']} | {s['avg_hallucinations']} | "
                     f"{s['zero_hallucination_rate']} | {s['trusted_rate']} | "
                     f"{s['avg_keypoint_coverage']} | {s['avg_elapsed_s']}s | ¥{s['avg_cost_cny']} |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[6:]))
    print(f"\n明细与原始报告已保存至: {out_dir}")


if __name__ == "__main__":
    main()
