"""
Agent 编排器：串联三个 Agent 的协作流程。

流程:
  Agent 1 (分类器) → Agent 2 (检索器) → Agent 3 (报告器) → 最终报告

同时也保留原始的单流水线作为 fallback。
"""
import time
from src.agents.classifier import RiskClassifier
from src.agents.retriever import LegalRetriever
from src.agents.reporter import ReportGenerator
from src.rag.generator import generate_report as single_pipeline_review


def multi_agent_review(user_input: str, verbose: bool = True) -> dict:
    """
    多Agent协作的合规审查。

    返回:
    {
        "report": "最终Markdown报告",
        "trace": [
            {"agent": "分类器", "summary": "...", "elapsed": 1.2},
            {"agent": "检索器", "summary": "...", "elapsed": 2.5},
            {"agent": "报告器", "summary": "...", "elapsed": 4.0},
        ],
        "classification": {...},
        "articles": [...],
    }
    """
    trace = []
    total_start = time.time()

    # ===== Agent 1: 分类 =====
    if verbose:
        print(f"\n{'='*50}")
        print("  Agent 1/3: 风险分类器")
        print(f"{'='*50}")

    t1 = time.time()
    classifier = RiskClassifier()
    classification = classifier.classify(user_input)
    elapsed1 = round(time.time() - t1, 2)

    trace.append({
        "agent": "🔍 风险分类器",
        "summary": f"识别数据类型: {', '.join(classification.get('data_types', []))} | "
                   f"风险等级: {classification.get('initial_risk_level', 'N/A')} | "
                   f"风险领域: {', '.join(classification.get('risk_areas', []))}",
        "elapsed": elapsed1,
    })

    if verbose:
        print(f"  风险等级: {classification.get('initial_risk_level')}")
        print(f"  数据类型: {classification.get('data_types')}")
        print(f"  风险领域: {classification.get('risk_areas')}")
        print(f"  耗时: {elapsed1}s")

    # ===== Agent 2: 检索 =====
    if verbose:
        print(f"\n{'='*50}")
        print("  Agent 2/3: 法规检索器")
        print(f"{'='*50}")

    t2 = time.time()
    retriever = LegalRetriever()
    articles = retriever.retrieve(user_input, classification)
    elapsed2 = round(time.time() - t2, 2)

    article_summary = " | ".join([
        f"{a['law_name']} {a['article_id']}"
        for a in articles[:5]
    ])

    trace.append({
        "agent": "📚 法规检索器",
        "summary": f"检索到 {len(articles)} 条相关法条: {article_summary}",
        "elapsed": elapsed2,
    })

    if verbose:
        print(f"  精选法条数: {len(articles)}")
        print(f"  耗时: {elapsed2}s")

    # ===== Agent 3: 报告生成 =====
    if verbose:
        print(f"\n{'='*50}")
        print("  Agent 3/3: 报告生成器")
        print(f"{'='*50}")

    t3 = time.time()
    reporter = ReportGenerator()
    report = reporter.generate(user_input, classification, articles)
    elapsed3 = round(time.time() - t3, 2)

    trace.append({
        "agent": "📝 报告生成器",
        "summary": f"生成合规审查报告",
        "elapsed": elapsed3,
    })

    total_elapsed = round(time.time() - total_start, 2)

    if verbose:
        print(f"\n  总耗时: {total_elapsed}s")

    return {
        "report": report,
        "trace": trace,
        "classification": classification,
        "articles": articles,
        "total_elapsed": total_elapsed,
    }
