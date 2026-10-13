"""文本工具：中文条款号解析等"""

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}


def article_no(article_id: str) -> int:
    """把「第十三条」「第131条」「一千二百六十条」转成整数，解析失败返回 -1"""
    text = str(article_id).replace("第", "").replace("条", "").strip()
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if text.isdigit():
        return int(text)
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _CN_NUM:
            number = _CN_NUM[ch]
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
