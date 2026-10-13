"""
法条引用解析器（评测用）

从生成的报告 Markdown 中抽取「(法规名, 条款号)」引用，用于：
1. 计算 ground-truth 法条召回率
2. 检测幻觉引用（引用了知识库里不存在的法规/条款）
"""
import json
import os
import re

KB_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed", "articles.json"
)

# 条款号正则：第X条 / 第 6 条 / 第X条之X
ARTICLE_RE = re.compile(r"第\s*([0-9０-９一二三四五六七八九十百千零两]+)\s*条")
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}


def parse_cn_number(text: str) -> int:
    """中文/阿拉伯数字转 int，支持「一百三十一」「一千二百六十」"""
    text = text.strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if text.isdigit():
        return int(text)
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return -1
    return total + section + number


def _short_name(law: str) -> str:
    name = law
    for prefix in ("中华人民共和国",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _extended_alias(law: str) -> str:
    """去掉「互联网信息服务」等常见前缀，兼容报告中的简称写法"""
    name = _short_name(law)
    for prefix in ("互联网信息服务",):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_kb(path: str = KB_PATH_DEFAULT) -> dict:
    """加载知识库，返回索引结构"""
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    law_articles = {}          # law -> set(article_no)
    aliases = {}               # alias -> law
    for a in articles:
        law = a["law_name"]
        no = parse_cn_number(a["article_id"].replace("第", "").replace("条", ""))
        law_articles.setdefault(law, set()).add(no)

    for law in law_articles:
        for alias in {law, _short_name(law), _extended_alias(law)}:
            # 别名冲突时保留更长的法规全名
            if alias not in aliases or len(law) > len(aliases[alias]):
                aliases[alias] = law

    return {
        "law_articles": law_articles,
        "aliases": aliases,
        "alias_list": sorted(aliases.keys(), key=len, reverse=True),
        "total_articles": len(articles),
    }


def extract_citations(report: str, kb: dict) -> dict:
    """
    抽取报告中的法条引用。

    返回:
    {
      "citations": [(law, article_no), ...],          # 去重后的引用
      "unknown_laws": [str, ...],                     # 引用了知识库外的法规
      "hallucinated": [(law, article_no), ...],       # 引用条款号不存在
    }
    """
    aliases = kb["aliases"]
    alias_list = kb["alias_list"]
    law_articles = kb["law_articles"]

    citations = set()
    unknown_laws = set()
    carried_law = None

    for line in report.split("\n"):
        # 1) 找出本行全部法规别名出现位置（长别名优先，避免子串误判）
        law_hits = []
        consumed = [False] * len(line)
        for alias in alias_list:
            start = 0
            while True:
                pos = line.find(alias, start)
                if pos < 0:
                    break
                if any(consumed[pos:pos + len(alias)]):
                    start = pos + 1
                    continue
                for i in range(pos, pos + len(alias)):
                    consumed[i] = True
                law_hits.append((pos, aliases[alias]))
                start = pos + len(alias)
        law_hits.sort()

        # 2) 找出本行全部条款号
        art_hits = [(m.start(), parse_cn_number(m.group(1))) for m in ARTICLE_RE.finditer(line)]

        if law_hits:
            carried_law = law_hits[-1][1]

        for pos, no in art_hits:
            # 就近归属：条款号前面最近的法规名；本行没有法规名则沿用上一行的法规
            owner = None
            for lpos, law in law_hits:
                if lpos <= pos:
                    owner = law
                else:
                    break
            if owner is None:
                owner = carried_law
            if owner is None:
                continue
            if no > 0:
                if no in law_articles.get(owner, set()):
                    citations.add((owner, no))
                else:
                    citations.add((owner, no))

    hallucinated = []
    for law, no in citations:
        if law not in law_articles:
            unknown_laws.add(law)
        elif no not in law_articles[law]:
            hallucinated.append((law, no))

    # 引用了知识库外法规：匹配「《XXX法》/《XXX条例》/《XXX办法》」且不命中别名表
    for m in re.finditer(r"《([^》]{3,30})》", report):
        name = m.group(1)
        if name in aliases:
            continue
        if any(name in alias or alias in name for alias in alias_list):
            continue
        unknown_laws.add(name)

    return {
        "citations": sorted(citations, key=lambda x: (x[0], x[1])),
        "unknown_laws": sorted(unknown_laws),
        "hallucinated": sorted(hallucinated, key=lambda x: (x[0], x[1])),
    }
