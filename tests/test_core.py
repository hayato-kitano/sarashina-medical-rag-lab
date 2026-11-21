"""モデル不要のユニットテスト（pytest）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.data_prep import check_record, japanese_ratio, number_consistency  # noqa: E402
from medrag.evaluation import parse_choice  # noqa: E402
from medrag.pipeline import MODES, build_messages, mode_of  # noqa: E402
from medrag.rag import split_text  # noqa: E402
from medrag.safety import detect_dangerous_advice, detect_emergency  # noqa: E402

F = dict(min_prompt_chars=5, max_prompt_chars=600, min_response_chars=20, max_response_chars=1500,
         min_japanese_ratio=0.35, max_ja_en_length_ratio=1.6, min_ja_en_length_ratio=0.12,
         check_number_consistency=True)


def test_modes_cover_four_conditions():
    assert set(MODES.values()) == {(False, False), (True, False), (False, True), (True, True)}
    assert mode_of(True, True) == "D" and mode_of(False, False) == "A"


def test_same_prompt_for_both_models():
    chunks = [{"title": "T", "text": "本文です。", "version": "1", "updated": "2026-01-01"}]
    m = build_messages("質問", "SYS", chunks, "指示")
    assert m[0] == {"role": "system", "content": "SYS"}
    assert "[1] 出典: T" in m[1]["content"] and "【質問】" in m[1]["content"]


def test_emergency_detection():
    assert detect_emergency("胸が締め付けられるように痛い").flagged
    assert "精神的緊急" in detect_emergency("死にたいです").categories
    assert not detect_emergency("高血圧の予防法は？").flagged


def test_dangerous_advice_detection():
    assert detect_dangerous_advice("自己判断で服薬を中止してください。").flagged
    assert not detect_dangerous_advice("服薬の変更は医師に相談してください。").flagged


def test_filters():
    ok = {"prompt_ja": "高血圧とは何ですか？", "response_ja": "高血圧は血圧が慢性的に高い状態で、脳卒中などのリスクを高めます。"}
    assert check_record(ok, F) is None
    assert check_record({**ok, "response_ja": "This answer remains entirely in English text here."}, F)
    assert number_consistency("dose 25 mg for 12 weeks", "25mgを12週間") == 1.0
    assert number_consistency("dose 25 mg for 12 weeks", "少量を数週間") == 0.0
    assert japanese_ratio("日本語") == 1.0


def test_split_text_respects_size():
    text = "。".join(["これはテスト用の文です"] * 100) + "。"
    chunks = split_text(text, 100, 20)
    assert chunks and all(len(c) <= 120 for c in chunks)


def test_parse_choice():
    assert parse_choice("答え: c", 1) == ["c"]
    assert parse_choice("答え：b, d", 2) == ["b", "d"]
