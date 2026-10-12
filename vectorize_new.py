"""
新增法规向量化脚本

用法（在 legal-review-assistant 目录下运行）:
    python vectorize_new.py              # 仅解析，打印统计，不写入
    python vectorize_new.py --apply      # 解析 + 向量化 + 合并到 processed/

职责：
1. 解析 data/raw/ 下新增的 8 个文件（2 docx + 6 pdf）
2. 对每条条款调用 SiliconFlow 生成向量（格式与 build_index 完全一致）
3. 追加合并到 processed/articles.json + processed/vectors.npy
"""
import os
import re
import sys
import json
import numpy as np

from docx import Document
import fitz  # PyMuPDF

from src.rag.embedding import get_embeddings

# bge-large-zh-v1.5 上限 512 token，中文约 1 字符/token（实测 520 字成功、540 字失败）
# 保留 margin，截断到 450 字，加上法条前缀后仍 <512 token
MAX_CHARS_FOR_EMBEDDING = 450

# 原始法规文件实际位于父级目录 data/raw（用户放置位置）
# legal-review-assistant/data/raw 为空
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # legal-review-assistant
RAW_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "raw")  # 父级 data/raw

# 新增的 8 个文件
NEW_FILES = [
    "中华人民共和国基本医疗卫生与健康促进法_20191228.docx",
    "中华人民共和国食品安全法_20250912.docx",
    "互联网信息服务深度合成管理规定__2023年第4号国务院公报.pdf",
    "互联网信息服务算法推荐管理规定__2022年第9号国务院公报_.pdf",
    "化妆品监督管理条例__2020年第19号国务院公报.pdf",
    "数据出境安全评估办法__2022年第24号国务院公报.pdf",
    "未成年人网络保护条例_教育_.pdf",
    "生成式人工智能服务管理暂行办法__2023年第24号国务院公报.pdf",
]

ARTICLE_RE = re.compile(r'^第[一二三四五六七八九十百千万零]+条')
CHAPTER_RE = re.compile(r'^第[一二三四五六七八九十百千万零]+章')


def law_name_from_filename(filename: str) -> str:
    """从文件名提取法规名称（去掉日期/公报等后缀）"""
    stem = os.path.splitext(filename)[0]
    # 名称以第一个下划线（半角或全角）为界
    return re.split(r'[_＿]', stem)[0]


def parse_single_docx(filepath: str) -> list[dict]:
    """解析 docx 法规（逻辑与 src.preprocess.parser 一致）"""
    doc = Document(filepath)
    filename = os.path.basename(filepath)
    law_name = law_name_from_filename(filename)

    articles = []
    current_chapter = "总则"
    current_article_id = ""
    current_text_lines = []
    in_body = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if CHAPTER_RE.match(text):
            current_chapter = text
            in_body = True
            continue
        m = ARTICLE_RE.match(text)
        if m:
            if current_article_id and current_text_lines:
                articles.append({
                    "law_name": law_name,
                    "article_id": current_article_id,
                    "chapter": current_chapter,
                    "text": "\n".join(current_text_lines),
                    "source_file": filename,
                })
            current_article_id = m.group()
            current_text_lines = [text]
            continue
        if in_body and current_article_id:
            current_text_lines.append(text)

    if current_article_id and current_text_lines:
        articles.append({
            "law_name": law_name,
            "article_id": current_article_id,
            "chapter": current_chapter,
            "text": "\n".join(current_text_lines),
            "source_file": filename,
        })
    return articles


def parse_single_pdf(filepath: str) -> list[dict]:
    """
    解析 pdf 法规（国务院公报版式）。

    公报版式特点：
    - 开头是"第　X　号"、署名、日期等导语，需跳过
    - 标题重复出现两次
    - 每条条款的条号会以"裸条号"单独成行，紧跟"第X条　正文"行
      例如: "第一条"  （裸条号，跳过）
            "第一条　为了规范数据出境活动……"  （真正正文，作为条款起点）
    """
    filename = os.path.basename(filepath)
    law_name = law_name_from_filename(filename)

    doc = fitz.open(filepath)
    lines = []
    for page in doc:
        for raw in page.get_text("text").split("\n"):
            t = raw.strip()
            if t:
                lines.append(t)
    doc.close()

    articles = []
    current_chapter = "总则"
    current_article_id = ""
    current_text_lines = []
    in_body = False

    for text in lines:
        if CHAPTER_RE.match(text):
            current_chapter = text
            in_body = True
            continue

        m = ARTICLE_RE.match(text)
        if m:
            # 裸条号（"第一条" 后无正文）→ 跳过；有正文 → 条款起点
            if len(text) == len(m.group()):
                continue
            if current_article_id and current_text_lines:
                articles.append({
                    "law_name": law_name,
                    "article_id": current_article_id,
                    "chapter": current_chapter,
                    "text": "\n".join(current_text_lines),
                    "source_file": filename,
                })
            current_article_id = m.group()
            current_text_lines = [text]
            in_body = True
            continue

        # 正文续行（进入正文后才收集）
        if in_body and current_article_id:
            current_text_lines.append(text)

    if current_article_id and current_text_lines:
        articles.append({
            "law_name": law_name,
            "article_id": current_article_id,
            "chapter": current_chapter,
            "text": "\n".join(current_text_lines),
            "source_file": filename,
        })
    return articles


def parse_new_files() -> list[dict]:
    """解析全部新增文件，返回条款列表"""
    all_new = []
    for filename in NEW_FILES:
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [跳过] 找不到 {filename}")
            continue
        if filename.endswith(".docx"):
            arts = parse_single_docx(filepath)
        else:
            arts = parse_single_pdf(filepath)
        print(f"  {filename} -> {len(arts)} 条")
        all_new.extend(arts)
    return all_new


def main():
    apply_flag = "--apply" in sys.argv

    # 读取现有索引
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
    articles_path = os.path.join(base, "articles.json")
    vectors_path = os.path.join(base, "vectors.npy")

    with open(articles_path, "r", encoding="utf-8") as f:
        existing_articles = json.load(f)
    existing_vectors = np.load(vectors_path)
    print(f"现有索引: {len(existing_articles)} 条, 向量 {existing_vectors.shape}")

    existing_laws = set(a["law_name"] for a in existing_articles)

    # 解析新增文件
    print("\n=== 解析新增文件 ===")
    new_articles = parse_new_files()
    print(f"\n新增条款合计: {len(new_articles)} 条")

    # 按法规统计
    from collections import Counter
    cnt = Counter(a["law_name"] for a in new_articles)
    print("\n=== 新增法规明细 ===")
    for law, n in cnt.items():
        dup = "  [与现有库重名!]" if law in existing_laws else ""
        print(f"  {law}: {n} 条{dup}")

    # 去重检查（同名法规）
    overlap = [law for law in cnt if law in existing_laws]
    if overlap:
        print(f"\n警告: 以下法规名称与现有库重复，合并会造成重复索引: {overlap}")
        print("已中止，未写入。请确认后再运行 --apply")
        return

    if not apply_flag:
        print("\n（仅解析模式，未写入。确认无误后运行: python vectorize_new.py --apply）")
        return

    # 向量化
    print("\n=== 向量化 ===")
    embeddings = get_embeddings()
    new_vectors = []
    BATCH_SIZE = 50
    total = len(new_articles)
    for i in range(0, total, BATCH_SIZE):
        batch = new_articles[i:i + BATCH_SIZE]
        texts = []
        for a in batch:
            full_text = a["text"]
            if len(full_text) > MAX_CHARS_FOR_EMBEDDING:
                full_text = full_text[:MAX_CHARS_FOR_EMBEDDING]
            texts.append(f"【{a['law_name']}】{a['article_id']}\n{full_text}")
        vecs = embeddings.embed_documents(texts)
        new_vectors.extend(vecs)
        print(f"  生成向量: {min(i + BATCH_SIZE, total)}/{total}")

    new_vectors = np.array(new_vectors, dtype=np.float32)
    assert new_vectors.shape[1] == existing_vectors.shape[1], \
        f"向量维度不一致: {new_vectors.shape[1]} vs {existing_vectors.shape[1]}"

    # 合并
    merged_articles = existing_articles + new_articles
    merged_vectors = np.vstack([existing_vectors, new_vectors])

    with open(articles_path, "w", encoding="utf-8") as f:
        json.dump(merged_articles, f, ensure_ascii=False, indent=2)
    np.save(vectors_path, merged_vectors)

    print(f"\n合并完成！总计 {len(merged_articles)} 条，向量 {merged_vectors.shape}")


if __name__ == "__main__":
    main()
