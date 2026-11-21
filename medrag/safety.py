"""医療安全性に関するルールベースのチェック。

- 質問側: 緊急性が疑われる相談の検出（診断せず受診・相談を促す）
- 回答側: 危険な助言・個別診断の疑いがある表現の検出（評価画面で警告表示）
- 学習データ側: 個別診断や危険な助言を含む回答の除外

キーワードベースの一次スクリーニングであり、最終判断は評価者が行う前提。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- 緊急性 -------------------------------------------------------------
_EMERGENCY_PATTERNS: dict[str, list[str]] = {
    "身体的緊急": [
        r"胸(が|の)?(痛|苦し|締め付け)", r"息(が)?(でき|苦し|吸えな)", r"呼吸(が)?(苦し|困難|でき)",
        r"意識(が)?(な|もうろう|朦朧|はっきりしな)", r"けいれん|痙攣", r"ろれつ(が)?回らな",
        r"(顔|手足|片側)(が|の)?(しびれ|麻痺|動かな)", r"大量(に)?(出血|吐血|下血)", r"血を吐",
        r"(激しい|突然の|今までにない)(頭痛|腹痛)", r"アナフィラキシー", r"(誤飲|誤って飲)",
        r"(薬|錠剤)を(大量|たくさん|全部)(に)?飲", r"倒れ(て|た)", r"(熱中症|低体温).*(意識|反応)",
    ],
    "精神的緊急": [
        r"死にたい", r"消えたい", r"自殺", r"自傷", r"生きていたくない", r"命を絶",
        r"(リストカット|リスカ|OD|オーバードーズ)",
    ],
}

EMERGENCY_GUIDANCE = {
    "身体的緊急": (
        "緊急性が高い可能性があるため、このアプリでは判断・回答を行いません。\n"
        "・意識がない、呼吸が苦しい、強い胸の痛みなどがある場合は、ためらわず119番に通報してください。\n"
        "・救急車を呼ぶか迷う場合は、救急安心センター（#7119、実施地域のみ）や医療機関に相談してください。"
    ),
    "精神的緊急": (
        "とてもつらい状況にあるのかもしれません。このアプリでは対応できないため、人に相談してください。\n"
        "・危険が差し迫っている場合は119番へ。\n"
        "・こころの健康相談統一ダイヤル：0570-064-556\n"
        "・よりそいホットライン：0120-279-338\n"
        "身近な信頼できる人や、かかりつけの医療機関に話すことも助けになります。"
    ),
}

# ---- 回答・学習データ中の危険表現 ---------------------------------------------
_DANGEROUS_ADVICE_PATTERNS: dict[str, list[str]] = {
    "個別診断": [
        r"あなた(は|の症状は|の病気は).{0,20}(です|でしょう|と診断|に間違いありません)",
        r"(と|に)診断します", r"診断結果は",
    ],
    "自己判断での服薬変更": [
        r"(医師|主治医|薬剤師)に(相談|確認)(せず|しないで|する必要はありません)",
        r"(自己判断で|勝手に)?(服薬|服用|内服|投薬)を(中止|やめ)(して|てください|ましょう)",
        r"(量|用量)を(倍|2倍|増や)(して|してください|しましょう)",
    ],
    "受診不要の断定": [
        r"(病院|医療機関|受診)(に行く|は)?(必要は?ありません|不要です)",
        r"(放置|様子見)して(も)?(大丈夫|問題ありません)",
    ],
    "断定的な治癒保証": [r"(必ず|絶対に?)(治り|治ります|治せ)", r"100[%％](治|効)"],
}


@dataclass
class SafetyCheck:
    flagged: bool
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "、".join(self.categories)


def _scan(text: str, table: dict[str, list[str]]) -> SafetyCheck:
    cats, matches = [], []
    for cat, pats in table.items():
        for pat in pats:
            m = re.search(pat, text or "")
            if m:
                if cat not in cats:
                    cats.append(cat)
                matches.append(m.group(0))
    return SafetyCheck(bool(cats), cats, matches)


def detect_emergency(question: str) -> SafetyCheck:
    return _scan(question, _EMERGENCY_PATTERNS)


def emergency_message(check: SafetyCheck) -> str:
    parts = [EMERGENCY_GUIDANCE[c] for c in ("精神的緊急", "身体的緊急") if c in check.categories]
    return "\n\n".join(parts)


def detect_dangerous_advice(answer: str) -> SafetyCheck:
    return _scan(answer, _DANGEROUS_ADVICE_PATTERNS)


DISCLAIMER = (
    "本アプリは技術検証用です。表示される内容は一般的な医療情報であり、診断・処方・治療判断には使用できません。"
    "体調に不安がある場合は医療機関に相談してください。"
)
