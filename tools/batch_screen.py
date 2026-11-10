# -*- coding: utf-8 -*-
"""批量筛查：辖区商户线上物料合规风险台账

用途（基层治理场景）：
    街道（乡镇）市场监管所把辖区内一批商户的线上物料清单交给系统，
    一次跑完，输出可统计的风险台账，用于确定下一轮重点检查对象。

用法：
    python tools/batch_screen.py --input 清单.csv --out 台账.csv
    python tools/batch_screen.py --input 清单.csv --out 台账.csv --limit 3

输入 CSV（首行为表头，UTF-8 编码，可带 BOM）：
    商户,渠道,物料类型,物料内容
    城南早点铺,美团外卖,店铺公告,本店所有粥品无糖更健康，糖尿病人可放心食用

输出：
    <台账>.csv        逐户审查结果
    终端汇总          风险分布、高频条款、耗时与成本
"""
import argparse
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import multi_agent_review  # noqa: E402

LEVEL_ORDER = {"高": 3, "中": 2, "低": 1, "unknown": 0}


def parse_report(report: str, articles: list) -> dict:
    """从 Markdown 报告中抽取台账字段。"""
    level = "unknown"
    m = re.search(r"风险等级[*\s】]*[：:]?\s*\**\s*(高|中|低)", report)
    if m:
        level = m.group(1)

    advice = ""
    m = re.search(r"上线建议[*\s】]*[：:]?\s*\**\s*(可直接上线|整改后可上线|不建议上线)", report)
    if m:
        advice = m.group(1)

    # 整改清单中的 P0/P1/P2 条目数
    counts = {}
    for p in ("P0", "P1", "P2"):
        # 兼容两种整改清单格式：Markdown 表格行（| P0 | ...）与列表项（- P0 ...）
        in_table = re.findall(rf"(?m)^\s*\|\s*\**\s*{p}\b", report)
        in_list = re.findall(rf"(?m)^\s*(?:[-*]|\d+[.、])\s*\**{p}\b", report)
        counts[p] = max(len(in_table), len(in_list))

    # 命中条款（来自检索结果，取前 4 条）
    hits = []
    for a in articles[:4]:
        law = str(a.get("law_name", "")).replace("中华人民共和国", "")
        aid = str(a.get("article_id", ""))
        hits.append(f"{law}{aid}")

    return {
        "风险等级": level,
        "上线建议": advice,
        "P0项数": counts["P0"],
        "P1项数": counts["P1"],
        "P2项数": counts["P2"],
        "核心条款": " / ".join(hits),
    }


def main():
    ap = argparse.ArgumentParser(description="辖区商户线上物料批量合规筛查")
    ap.add_argument("--input", required=True, help="商户清单 CSV")
    ap.add_argument("--out", required=True, help="台账输出 CSV")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 户（调试用）")
    ap.add_argument("--verbose", action="store_true", help="打印每个智能体的执行细节")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        rows = rows[: args.limit]
    print(f"\n辖区商户清单：{len(rows)} 户\n" + "=" * 62)

    results, t_all, t0 = [], 0.0, time.time()
    for i, row in enumerate(rows, 1):
        name = (row.get("商户") or "").strip() or f"商户{i}"
        channel = (row.get("渠道") or "").strip()
        kind = (row.get("物料类型") or "").strip()
        content = (row.get("物料内容") or "").strip()

        print(f"[{i}/{len(rows)}] {name}（{channel} · {kind}）… ", end="", flush=True)
        if not content:
            print("跳过：物料内容为空")
            continue

        start = time.time()
        try:
            res = multi_agent_review(content, verbose=args.verbose)
        except Exception as e:  # 单户失败不影响整批
            print(f"失败：{type(e).__name__}: {e}")
            results.append({
                "商户": name, "渠道": channel, "物料类型": kind,
                "风险等级": "审查失败", "上线建议": "", "P0项数": "", "P1项数": "",
                "P2项数": "", "核心条款": "", "耗时(秒)": "",
            })
            continue

        elapsed = round(time.time() - start, 1)
        t_all += elapsed
        info = parse_report(res["report"], res.get("articles", []))
        info.update({
            "商户": name, "渠道": channel, "物料类型": kind,
            "耗时(秒)": elapsed,
        })
        results.append(info)
        print(f"{info['风险等级']}风险 / {info['上线建议'] or '—'} / "
              f"P0×{info['P0项数']} / {elapsed}s")

    fields = ["商户", "渠道", "物料类型", "风险等级", "上线建议",
              "P0项数", "P1项数", "P2项数", "核心条款", "耗时(秒)"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})

    # ===== 汇总 =====
    ok = [r for r in results if r.get("风险等级") in LEVEL_ORDER and r["风险等级"] != "unknown"]
    dist = {k: sum(1 for r in ok if r.get("风险等级") == k) for k in ("高", "中", "低")}

    print("=" * 62)
    print(f"台账已写入：{args.out}")
    print(f"有效审查 {len(ok)} 户 | 高风险 {dist['高']} 户 | 中风险 {dist['中']} 户 | 低风险 {dist['低']} 户")
    print(f"需上线前整改（P0>0）{sum(1 for r in ok if str(r.get('P0项数')).isdigit() and int(r['P0项数']) > 0)} 户")
    print(f"总耗时 {round(time.time() - t0, 1)} 秒 | 单户平均 {round(t_all / max(len(ok), 1), 1)} 秒")

    # 高频条款
    freq = {}
    for r in ok:
        for a in str(r.get("核心条款", "")).split(" / "):
            a = a.strip()
            if a:
                freq[a] = freq.get(a, 0) + 1
    if freq:
        top = sorted(freq.items(), key=lambda x: -x[1])[:5]
        print("本辖区出现频次最高的条款：" + "、".join(f"{k}（{v}户）" for k, v in top))
    print("=" * 62)


if __name__ == "__main__":
    main()
