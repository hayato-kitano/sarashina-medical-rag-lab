"""文字の正規化・OCRノイズ除去・表示用エスケープ（文字化け対策）。

文字化けの主な原因と対策
- OCRが出す制御文字・私用領域・結合文字・置換文字（U+FFFD） → clean_ocr_text() で除去
- OCRの誤読記号（[ * _ # | など）がMarkdownとして解釈され表示が崩れる → md_escape() で必ずエスケープ
- ラベル上の表記が読めない文字列のまま表示される → is_readable() を満たさない表記は表示しない
"""
from __future__ import annotations

import re
import unicodedata

# 日本語・英数字・食品表示でよく使う記号だけを残す
_ALLOWED = re.compile(
    r"[^぀-ゟ゠-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿"
    r"　-〿！-～｡-ﾟ"
    r"0-9A-Za-z\s/・、。，．,.:：;；()（）\[\]［］「」『』%％℃+\-－ー~〜※*＊&＆'’\"”α-ωΑ-Ω]"
)
_JA = re.compile(r"[぀-ヿ㐀-鿿]")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def normalize(text: str) -> str:
    """NFKC正規化（全角英数→半角、半角カナ→全角）と記号の統一。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("／", "/").replace("（", "(").replace("）", ")")
    return text


def japanese_ratio(text: str) -> float:
    t = re.sub(r"\s", "", text)
    return len(_JA.findall(t)) / max(1, len(t))


def clean_ocr_text(text: str) -> str:
    """OCR結果からノイズ（制御文字・不明な記号・意味のない行）を取り除く。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("�", "")
    text = re.sub(r"<bbox>.*?</bbox>", "", text)            # Sarashina-OCR の図版座標
    text = re.sub(r"</?(table|tr|td|th|thead|tbody)[^>]*>", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    text = _ALLOWED.sub("", text)
    # Tesseract は日本語の文字間に空白を入れるので詰める
    text = re.sub(r"(?<=[^\x00-\x7F]) +(?=[^\x00-\x7F])", "", text)
    text = re.sub(r"(?<=[^\x00-\x7F]) +(?=[(),、。])|(?<=[(),、。]) +(?=[^\x00-\x7F])", "", text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # 日本語がほぼ無く記号だらけの行は読み取りノイズとみなす（「内容量 180g」などは残す）
        if japanese_ratio(s) < 0.15 and not re.search(r"[A-Za-z]{2,}|\d", s):
            continue
        lines.append(re.sub(r"[ \t]{2,}", " ", s))
    return "\n".join(lines)


def is_readable(s: str) -> bool:
    """画面に出してよい（読める）表記か。"""
    if not s or len(s) > 40:
        return False
    if _ALLOWED.search(s):
        return False
    return bool(_JA.search(s)) or bool(re.fullmatch(r"[A-Za-z0-9\-]{2,}", s))


def md_escape(s: str) -> str:
    """Markdownとして解釈される記号をエスケープする（st.markdown に渡す文字列は必ず通す）。"""
    return _MD_SPECIAL.sub(r"\\\1", s or "")
