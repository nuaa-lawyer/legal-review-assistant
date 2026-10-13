"""
法规文档解析器：将Word格式的法律法规文件按条款拆分为结构化JSON。

输入：data/raw/ 下的 .docx 文件
输出：data/processed/articles.json（每个条款一条记录）

核心逻辑：
1. 读取Word文档的段落
2. 识别"第X章"（章节标题）和"第X条"（条款边界）
3. 将同一条款下的多个段落合并
4. 过滤掉目录、页眉页脚等非正文内容
"""
import os
import re
import json
from docx import Document

from src.utils.config import DATA_RAW, PRIORITY_LAWS


def parse_single_docx(filepath: str) -> list[dict]:
    """
    解析单个Word法规文件，返回条款列表。

    每个条款的格式：
    {
        "law_name": "中华人民共和国个人信息保护法",
        "article_id": "第十三条",
        "chapter": "第二章 个人信息处理规则",
        "text": "条款全文...",
        "source_file": "中华人民共和国个人信息保护法_20210820.docx"
    }
    """
    doc = Document(filepath)
    filename = os.path.basename(filepath)

    # 从文件名提取法规名称（去掉日期后缀）
    law_name = re.sub(r'_\d{8}\.docx?$', '', filename)

    articles = []
    current_chapter = "总则"  # 默认章节
    current_article_id = ""
    current_text_lines = []
    in_body = False  # 是否已进入正文（跳过目录部分）

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 检测正文开始（正文以"第"字开头的条款为主）
        # 跳过目录：目录中的章节名后面紧跟的"第X条"不是真正的正文
        # 注意：字符集必须覆盖「千/零/两」，否则「第一百零一条」「第一千零一条」
        # 这类条款号会被误判为正文续行，被并入上一条（曾导致民法典漏 342 条）
        if re.match(r'^第[一二三四五六七八九十百千万零两]+章', text):
            current_chapter = text
            in_body = True
            continue

        # 检测条款开始
        article_match = re.match(r'^第[一二三四五六七八九十百千万零两]+条', text)
        if article_match:
            # 保存上一条
            if current_article_id and current_text_lines:
                articles.append({
                    "law_name": law_name,
                    "article_id": current_article_id,
                    "chapter": current_chapter,
                    "text": "\n".join(current_text_lines),
                    "source_file": filename,
                })

            # 开始新条款
            current_article_id = article_match.group()
            current_text_lines = [text]
            continue

        # 正文段落（只有进入正文后才收集）
        if in_body and current_article_id:
            current_text_lines.append(text)

    # 保存最后一条
    if current_article_id and current_text_lines:
        articles.append({
            "law_name": law_name,
            "article_id": current_article_id,
            "chapter": current_chapter,
            "text": "\n".join(current_text_lines),
            "source_file": filename,
        })

    return articles


def parse_all_docx(raw_dir: str = DATA_RAW) -> list[dict]:
    """
    解析 data/raw/ 下所有 .docx 文件，返回全部条款。
    优先处理核心法规（PRIORITY_LAWS）。
    """
    all_articles = []
    priority_articles = []
    other_articles = []

    for filename in os.listdir(raw_dir):
        if not filename.endswith('.docx'):
            continue

        filepath = os.path.join(raw_dir, filename)
        print(f"  解析中: {filename}...")
        try:
            articles = parse_single_docx(filepath)
            # 判断是否为核心法规
            law_name = articles[0]["law_name"] if articles else ""
            is_priority = any(pl in law_name for pl in PRIORITY_LAWS)
            if is_priority:
                priority_articles.extend(articles)
            else:
                other_articles.extend(articles)
            print(f"    -> 提取 {len(articles)} 条")
        except Exception as e:
            print(f"    -> 解析失败: {e}")

    # 核心法规排在前面
    all_articles = priority_articles + other_articles
    return all_articles


def save_articles(articles: list[dict], output_path: str):
    """保存为JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"\n总共 {len(articles)} 条法规条款 -> {output_path}")


def print_statistics(articles: list[dict]):
    """打印解析统计信息"""
    law_counts = {}
    for a in articles:
        name = a["law_name"]
        law_counts[name] = law_counts.get(name, 0) + 1

    print("\n=== 解析统计 ===")
    print(f"法规总数: {len(law_counts)} 部")
    print(f"条款总数: {len(articles)} 条")
    print("\n核心法规条款数:")
    for law, count in sorted(law_counts.items(), key=lambda x: -x[1]):
        is_priority = any(pl in law for pl in PRIORITY_LAWS)
        marker = "★" if is_priority else " "
        print(f"  {marker} {law}: {count} 条")


if __name__ == "__main__":
    # 用法: python -m src.preprocess.parser
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "processed", "articles.json"
    )
    articles = parse_all_docx(DATA_RAW)
    save_articles(articles, output_path)
    print_statistics(articles)
