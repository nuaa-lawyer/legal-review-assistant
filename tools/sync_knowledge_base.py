"""
知识库增量更新管道（持续学习 · 数据侧）

作用：让法审助手在"新法规发布"后能靠一条命令完成知识库迭代，而不是人工重跑全量索引。

流程：
  1. 扫描 data/raw/ 下的 docx / pdf / txt / md 法规原文
  2. 解析为条款级结构
  3. 与现有索引比对，只对"新增条款"生成向量（增量，不重复付费）
  4. 合并写回 processed/articles.json + processed/vectors.npy
  5. 写入 processed/kb_manifest.json 与 feedback/update_log.jsonl（可核查的更新记录）

用法（在 legal-review-assistant 目录下运行）：
    python tools/sync_knowledge_base.py            # 扫描并预览（不写库）
    python tools/sync_knowledge_base.py --apply    # 增量入库
    python tools/sync_knowledge_base.py --apply --raw-dir "E:/法条"
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from src.rag.embedding import get_embeddings  # noqa: E402
from src.feedback.store import log_kb_update  # noqa: E402

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
ARTICLES_PATH = os.path.join(PROCESSED_DIR, "articles.json")
VECTORS_PATH = os.path.join(PROCESSED_DIR, "vectors.npy")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "kb_manifest.json")

DEFAULT_RAW_DIRS = [
    os.path.join(os.path.dirname(BASE_DIR), "data", "raw"),   # 父级 data/raw
    os.path.join(BASE_DIR, "data", "raw"),
]

MAX_CHARS_FOR_EMBEDDING = 450
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千万零]+条")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万零]+章")
SUPPORTED_EXT = (".docx", ".pdf", ".txt", ".md")


def law_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.split(r"[_＿]", stem)[0]


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """归一化正文（忽略空白差异），用于判断条款正文是否被污染"""
    return _WS_RE.sub("", text or "")


def _collect_paragraphs(docx_path: str) -> list:
    from docx import Document
    return [p.text.strip() for p in Document(docx_path).paragraphs]


def _collect_pdf_lines(pdf_path: str) -> list:
    import fitz
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        lines.extend(t.strip() for t in page.get_text("text").split("\n"))
    doc.close()
    return lines


def _collect_text_lines(txt_path: str) -> list:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f.read().split("\n")]


def parse_to_articles(filepath: str) -> list:
    """把任意格式的法规原文解析为条款列表（与现有索引结构完全一致）"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        lines = _collect_paragraphs(filepath)
    elif ext == ".pdf":
        lines = _collect_pdf_lines(filepath)
    else:
        lines = _collect_text_lines(filepath)

    filename = os.path.basename(filepath)
    law_name = law_name_from_filename(filename)

    articles, chapter, cur_id, buf, in_body = [], "总则", "", [], False
    for text in lines:
        if not text:
            continue
        if CHAPTER_RE.match(text):
            chapter, in_body = text, True
            continue
        m = ARTICLE_RE.match(text)
        if m:
            # 公报 PDF 中"裸条号"单独成行 → 跳过
            if len(text) == len(m.group()) and ext == ".pdf":
                continue
            if cur_id and buf:
                articles.append({"law_name": law_name, "article_id": cur_id,
                                 "chapter": chapter, "text": "\n".join(buf),
                                 "source_file": filename})
            cur_id, buf, in_body = m.group(), [text], True
            continue
        if in_body and cur_id:
            buf.append(text)
    if cur_id and buf:
        articles.append({"law_name": law_name, "article_id": cur_id,
                         "chapter": chapter, "text": "\n".join(buf),
                         "source_file": filename})
    return articles


def find_raw_dirs(raw_dir_arg: str) -> list:
    if raw_dir_arg:
        return [raw_dir_arg]
    return [d for d in DEFAULT_RAW_DIRS if os.path.isdir(d)]


def scan_files(raw_dirs: list) -> list:
    files = []
    for d in raw_dirs:
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(SUPPORTED_EXT):
                files.append(os.path.join(d, name))
    return files


def load_index():
    if not os.path.exists(ARTICLES_PATH):
        raise SystemExit(f"找不到现有索引: {ARTICLES_PATH}")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    vectors = np.load(VECTORS_PATH) if os.path.exists(VECTORS_PATH) else None
    return articles, vectors


def build_manifest(articles: list) -> dict:
    cnt = Counter(a["law_name"] for a in articles)
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "total_articles": len(articles),
        "law_count": len(cnt),
        "laws": [{"law_name": k, "articles": v} for k, v in sorted(cnt.items())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入索引（默认仅扫描预览）")
    ap.add_argument("--raw-dir", default="", help="指定法规原文目录")
    args = ap.parse_args()

    raw_dirs = find_raw_dirs(args.raw_dir)
    if not raw_dirs:
        raise SystemExit("未找到法规原文目录，请用 --raw-dir 指定")
    print(f"扫描目录: {', '.join(raw_dirs)}")

    articles, vectors = load_index()
    existing = {(a["law_name"], a["article_id"]) for a in articles}
    existing_map = {(a["law_name"], a["article_id"]): a for a in articles}
    existing_laws = Counter(a["law_name"] for a in articles)
    print(f"现有索引: {len(articles)} 条 / {len(existing_laws)} 部法规")

    # 1) 解析全部原文文件，筛出「新增条款」与「正文被污染、需修正的条款」
    new_articles, changed_articles, file_report = [], [], []
    for fp in scan_files(raw_dirs):
        parsed = parse_to_articles(fp)
        fresh = [a for a in parsed if (a["law_name"], a["article_id"]) not in existing]
        changed = [
            a for a in parsed
            if (a["law_name"], a["article_id"]) in existing_map
            and _normalize(existing_map[(a["law_name"], a["article_id"])]["text"]) != _normalize(a["text"])
            and len(a["text"]) < len(existing_map[(a["law_name"], a["article_id"])]["text"])
        ]
        file_report.append((os.path.basename(fp), len(parsed), len(fresh), len(changed)))
        new_articles.extend(fresh)
        changed_articles.extend(changed)

    print("\n文件解析结果（文件 / 解析条款 / 新增 / 正文修正）：")
    for name, total, fresh, changed in file_report:
        flag = "" if (fresh or changed) else "  ← 已是最新"
        print(f"  {name}: {total} 条，新增 {fresh} 条，修正 {changed} 条{flag}")

    new_laws = sorted({a["law_name"] for a in new_articles} - set(existing_laws))
    print(f"\n待新增条款: {len(new_articles)} 条")
    if new_laws:
        print(f"涉及新增法规: {', '.join(new_laws)}")
    if changed_articles:
        print(f"待修正条款（原正文被并入上一条，需恢复拆分）: {len(changed_articles)} 条")

    pending = new_articles + changed_articles
    if not pending:
        print("知识库已是最新，无需更新。")
        if args.apply:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(build_manifest(articles), f, ensure_ascii=False, indent=2)
        return

    if not args.apply:
        print("\n（预览模式，未写入。确认后运行: python tools/sync_knowledge_base.py --apply）")
        return

    # 2) 增量向量化（新增条款 + 正文被修正的条款）
    print("\n=== 增量向量化 ===")
    embeddings = get_embeddings()
    new_vectors, BATCH = [], 50
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        texts = [f"【{a['law_name']}】{a['article_id']}\n{a['text'][:MAX_CHARS_FOR_EMBEDDING]}"
                 for a in batch]
        new_vectors.extend(embeddings.embed_documents(texts))
        print(f"  向量化进度: {min(i + BATCH, len(pending))}/{len(pending)}")

    new_vectors = np.array(new_vectors, dtype=np.float32)
    if vectors is not None:
        assert new_vectors.shape[1] == vectors.shape[1], \
            f"向量维度不一致: {new_vectors.shape[1]} vs {vectors.shape[1]}"

    # 3) 合并：新增条款追加，被修正条款原位替换（正文与向量同步更新）
    pending_map = {(a["law_name"], a["article_id"]): a for a in pending}
    replaced = 0
    final_articles, final_vectors = [], []
    for i, a in enumerate(articles):
        key = (a["law_name"], a["article_id"])
        if key in pending_map:
            new_a = dict(a)
            new_a["text"] = pending_map[key]["text"]
            new_a["chapter"] = pending_map[key].get("chapter", a.get("chapter", ""))
            final_articles.append(new_a)
            final_vectors.append(new_vectors[pending.index(pending_map[key])])
            replaced += 1
        else:
            final_articles.append(a)
            if vectors is not None:
                final_vectors.append(vectors[i])

    pure_new = [a for a in new_articles]
    for a in pure_new:
        final_articles.append(a)
        final_vectors.append(new_vectors[pending.index(a)])

    merged_vectors = np.array(final_vectors, dtype=np.float32)
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(final_articles, f, ensure_ascii=False, indent=2)
    np.save(VECTORS_PATH, merged_vectors)

    manifest = build_manifest(final_articles)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log_kb_update({
        "action": "incremental_update",
        "added_articles": len(new_articles),
        "fixed_articles": replaced,
        "added_laws": new_laws,
        "total_articles": len(final_articles),
        "law_count": manifest["law_count"],
    })

    print(f"\n✅ 更新完成：{len(articles)} → {len(final_articles)} 条，"
          f"修正 {replaced} 条，{len(existing_laws)} → {manifest['law_count']} 部法规")
    print(f"   manifest: {MANIFEST_PATH}")
    print("   变更记录: feedback/update_log.jsonl")


if __name__ == "__main__":
    main()
