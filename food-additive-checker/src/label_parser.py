"""OCRテキストから「原材料名」欄を取り出し、添加物の候補に分割する。"""
from __future__ import annotations

import re
import unicodedata

_FIELD_END = r"(内容量|内容総量|固形量|賞味期限|消費期限|保存方法|製造者|製造所|販売者|加工者|輸入者|原産国名|栄養成分|使用上の注意|アレルギー)"
_START = re.compile(r"原\s*材\s*料\s*名?")
_END = re.compile(_FIELD_END)
_SEP_TOP = "、,，"


def normalize(text: str) -> str:
    """全角英数を半角に、全角スラッシュ・括弧を統一。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("／", "/").replace("（", "(").replace("）", ")")
    # OCRが行頭の「/」を「ノ」「|」「l」と誤読することがある
    text = re.sub(r"(^|\n)\s*[ノ|l]\s*(?=\S)", r"\1/", text)
    text = re.sub(r"<bbox>.*?</bbox>", "", text)          # OCRの図版座標を除去
    text = re.sub(r"</?(table|tr|td|th)[^>]*>", " ", text)  # 表のHTMLタグを除去
    return text


def extract_ingredients_section(text: str) -> str:
    text = normalize(text)
    m = _START.search(text)
    if not m:
        return text  # 欄名が読めなかった場合は全文を対象にする
    rest = text[m.end():]
    e = _END.search(rest)
    section = rest[: e.start()] if e else rest
    return re.sub(r"\s+", "", section).lstrip(":：")


def _split_top(s: str) -> list[str]:
    """括弧の外にある読点でだけ分割する。"""
    out, buf, depth = [], "", 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in _SEP_TOP and depth == 0:
            if buf:
                out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [x.strip() for x in out if x.strip()]


def split_additives(section: str) -> tuple[list[str], list[str]]:
    """(原材料, 添加物) を返す。「/」以降、または「添加物」欄以降を添加物とみなす。"""
    if "/" in section:
        ing, add = section.split("/", 1)
    elif "添加物" in section:
        ing, add = section.split("添加物", 1)
    else:
        ing, add = section, ""
    return _split_top(ing), _split_top(add.replace("/", "、"))


def expand(item: str) -> list[str]:
    """「発色剤(亜硝酸Na)」→ [発色剤(亜硝酸Na), 亜硝酸Na] のように中身も候補にする。"""
    names = [item]
    m = re.match(r"^(.*?)\((.*)\)$", item)
    if m:
        names.append(m.group(1))
        for inner in re.split(r"[、,，]", m.group(2)):
            inner = inner.strip()
            if inner and not inner.endswith("等"):
                names.append(inner)
    return [n for n in names if n]


def parse_label(text: str) -> dict:
    section = extract_ingredients_section(text)
    ingredients, additives = split_additives(section)
    return {"section": section, "ingredients": ingredients, "additives": additives}
