"""モデル・ネットワーク不要のテスト：python -m pytest tests"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from foodrag.config import load_config  # noqa: E402
from foodrag.label import parse_label  # noqa: E402
from foodrag.pipeline import analyze_text, ground_sentences  # noqa: E402
from foodrag.rag import Retriever, build_index, load_catalog  # noqa: E402
from foodrag.text import clean_ocr_text, is_readable, md_escape  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    idx = tmp_path_factory.mktemp("idx")
    cfg = load_config(overrides={"rag": {"embedding_model": "hashing", "index_dir": str(idx)}})
    build_index(cfg)
    return cfg, Retriever(cfg), load_catalog(cfg)


def run(env, text, model=None):
    cfg, r, cat = env
    return analyze_text(text, "test", model, r, cat, cfg)


def test_parse_slash():
    p = parse_label("原材料名 豚肉、食塩／発色剤（亜硝酸Na）、調味料（アミノ酸等）\n内容量 100g")
    assert p["ingredients"] == ["豚肉", "食塩"]
    assert p["additives"] == ["発色剤(亜硝酸Na)", "調味料(アミノ酸等)"]


def test_findings_and_order(env):
    res = run(env, open("data/eval/sample_label.txt", encoding="utf-8").read())
    names = [f.name for f in res.findings]
    assert names[0] == "亜硝酸ナトリウム"
    assert "硝酸カリウム・硝酸ナトリウム" not in names           # 亜硝酸Na を 硝酸Na と誤認しない
    assert {"カラメル色素", "アスパルテーム"} <= set(names)


def test_benign_label(env):
    res = run(env, "原材料名 小麦粉、砂糖/膨張剤、香料、乳化剤、トレハロース、セルロース")
    assert res.findings == []
    assert "見つかりませんでした" in res.answer


def test_ocr_typo(env):
    assert [f.name for f in run(env, "原材料名 豚肉/亜硝酸ナトリウ厶").findings] == ["亜硝酸ナトリウム"]


def test_mojibake_cleanup():
    raw = "原材料名​ 豚肉 、 食塩 ／ 発色剤 ( 亜硝酸 Na )�\n@@##|||~~\n内容量 180g"
    t = clean_ocr_text(raw)
    assert "�" not in t and "​" not in t and "@@" not in t
    assert "発色剤(亜硝酸Na)" in t.replace(" ", "")
    assert "内容量 180g" in t


def test_unreadable_label_text_hidden(env):
    res = run(env, "原材料名 豚肉/甘味料(アスパルテーム・ [-フェニルアラニン化合物)")
    assert res.findings and all(is_readable(f.label_text) or f.label_text == "" for f in res.findings)
    assert md_escape("[-フェ*ニル_") == "\\[\\-フェ\\*ニル\\_"


def test_grounding_with_fake_model(env):
    cfg, r, cat = env
    out = ("カラメル色素はコーラ、ソース、しょうゆ加工品などに使われる茶色の着色料です。[2]\n"
           "この食品を食べると必ずがんになります。[2]\n"
           "IARC分類は実際の摂取量でのリスクを直接示すものではなく")
    fake = SimpleNamespace(device="cpu", generate=lambda m, g, n: SimpleNamespace(text=out, time_sec=0.1),
                           version_info=lambda: {"base_model": "fake"})
    res = run(env, "原材料名 豚肉/発色剤(亜硝酸Na)、カラメル色素", fake)
    assert res.answer_source == "sarashina+資料"
    assert "必ずがんになります" not in res.answer             # 根拠の無い文は採用しない
    assert "直接示すものではなく" not in res.answer           # 途中で切れた文は採用しない
    assert res.answer.startswith("（資料より）亜硝酸ナトリウム")  # 注意度「高」は必ず含める
