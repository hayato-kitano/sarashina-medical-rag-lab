"""学習データ前処理（要件 4. データ前処理 1〜6）。

1. 空欄・不正な文字列・重複を除外
2. 翻訳誤りの疑いを除外（日本語率、英語原文との長さ比・数値の不一致、繰り返しループ）
   ※「医学的に不正確な内容」は自動判定できないため、レビュー用CSVを出力し、
     人手で除外したIDを manual_exclusion_file に記載して再実行する運用とする
3. 個別診断・危険な助言を含む回答を除外（safety.detect_dangerous_advice）
4. train / validation / test に分割
5. Sarashina のチャットテンプレートで学習できる会話形式（prompt / completion）へ変換
6. 内容ハッシュからデータ版（data_version）を発行し、manifest と一緒に保存
"""
from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import model_ref, resolve
from .safety import detect_dangerous_advice
from .versioning import now_iso, sha256_text, write_json

_JA = re.compile(r"[぀-ゟ゠-ヿ一-鿿㐀-䶿々ー]")
_BAD_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOOP = re.compile(r"(.{2,30}?)\1{4,}", re.S)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def japanese_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(bool(_JA.match(c)) for c in chars) / len(chars) if chars else 0.0


def number_consistency(en: str, ja: str) -> float:
    """英語原文に含まれる数値のうち、訳文にも現れる割合（数値が2個未満なら1.0）。"""
    en_nums = {n for n in _NUM.findall(normalize(en).replace(",", "")) if len(n) <= 8}
    if len(en_nums) < 2:
        return 1.0
    ja_text = normalize(ja).replace(",", "")
    return sum(n in ja_text for n in en_nums) / len(en_nums)


def check_record(rec: dict, f: dict) -> str | None:
    """除外理由を返す（問題なければ None）。"""
    p, r = normalize(rec.get("prompt_ja", "")), normalize(rec.get("response_ja", ""))
    if not p or not r:
        return "空欄"
    if _BAD_CHARS.search(p) or _BAD_CHARS.search(r):
        return "不正な文字列"
    if not (f["min_prompt_chars"] <= len(p) <= f["max_prompt_chars"]):
        return "質問の長さが範囲外"
    if not (f["min_response_chars"] <= len(r) <= f["max_response_chars"]):
        return "回答の長さが範囲外"
    if japanese_ratio(p) < f["min_japanese_ratio"] or japanese_ratio(r) < f["min_japanese_ratio"]:
        return "翻訳誤り疑い:日本語率が低い"
    if _LOOP.search(r) or _LOOP.search(p):
        return "翻訳誤り疑い:繰り返しループ"
    en_r = rec.get("response") or ""
    if en_r:
        ratio = len(r) / max(len(en_r), 1)
        if not (f["min_ja_en_length_ratio"] <= ratio <= f["max_ja_en_length_ratio"]):
            return "翻訳誤り疑い:原文との長さ比"
        if f.get("check_number_consistency", True) and number_consistency(en_r, r) < 0.7:
            return "翻訳誤り疑い:数値の不一致"
    if detect_dangerous_advice(r).flagged:
        return "安全性:個別診断・危険な助言の疑い"
    return None


# ---- 読み込み ---------------------------------------------------------------
def load_apollo(dataset_name: str, source_file: str | None, max_rows: int | None, seed: int) -> list[dict]:
    if source_file:
        rows = read_jsonl(resolve(source_file))
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        if max_rows and max_rows < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_rows))
        rows = ds.to_list()
    out = []
    for i, r in enumerate(rows):
        out.append({"id": f"apollo-{r.get('index', i)}", "source": "ApolloCorpus-ja",
                    "prompt_ja": r.get("prompt_ja", ""), "response_ja": r.get("response_ja", ""),
                    "prompt": r.get("prompt", ""), "response": r.get("response", "")})
    if source_file and max_rows and max_rows < len(out):
        random.Random(seed).shuffle(out)
        out = out[:max_rows]
    return out


def load_alpacare(max_rows: int | None, seed: int) -> list[dict]:
    """補助データ JP-AlpaCare-MedInstruct-52k（非商用条件に注意。権利確認後のみ使用）。"""
    from datasets import load_dataset
    ds = load_dataset("li-lab/JP-AlpaCare-MedInstruct-52k", split="train")
    if max_rows and max_rows < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_rows))
    out = []
    for i, r in enumerate(ds):
        inst, inp = r.get("instruction_ja") or "", r.get("input_ja") or ""
        if inp.strip() and inp.strip() != "<noinput>":
            inst = f"{inst}\n\n{inp}"
        out.append({"id": f"alpacare-{i}", "source": "JP-AlpaCare-MedInstruct-52k",
                    "prompt_ja": inst, "response_ja": r.get("output_ja") or ""})
    return out


def load_extra_jsonl(path: str, prompt_field: str, response_field: str, source: str) -> list[dict]:
    """任意の追加データ（例：利用規約に同意して取得した AnswerCarefully を JSONL 化したもの）。"""
    return [{"id": f"{source}-{i}", "source": source, "prompt_ja": r.get(prompt_field, ""),
             "response_ja": r.get(response_field, "")} for i, r in enumerate(read_jsonl(resolve(path)))]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- 変換 -------------------------------------------------------------------
def to_conversation(rec: dict, system_prompt: str) -> dict:
    """TRL の会話形式 prompt/completion。学習時にトークナイザーのチャットテンプレートで展開され、
    回答（completion）部分のみに損失がかかる。"""
    return {
        "id": rec["id"],
        "source": rec["source"],
        "prompt": [{"role": "system", "content": system_prompt.strip()},
                   {"role": "user", "content": normalize(rec["prompt_ja"])}],
        "completion": [{"role": "assistant", "content": normalize(rec["response_ja"])}],
    }


def prepare(cfg: dict, records: list[dict], n_samples: int | None = None,
            tokenizer_name: str | None = None) -> dict[str, Any]:
    dcfg = cfg["data"]
    f = dcfg["filters"]
    seed = int(dcfg["seed"])
    n_samples = int(n_samples or dcfg["n_samples"])

    excl_path = resolve(dcfg["manual_exclusion_file"])
    manual_excl = set()
    if excl_path.exists():
        manual_excl = {ln.strip() for ln in excl_path.read_text(encoding="utf-8").splitlines()
                       if ln.strip() and not ln.startswith("#")}

    reasons: Counter = Counter()
    rejected, kept, seen_p, seen_r = [], [], set(), set()
    for rec in records:
        reason = "人手レビューで除外" if rec["id"] in manual_excl else check_record(rec, f)
        if reason is None:
            kp = re.sub(r"\s+", "", normalize(rec["prompt_ja"]))
            kr = re.sub(r"\s+", "", normalize(rec["response_ja"]))
            if kp in seen_p or kr in seen_r:
                reason = "重複"
            seen_p.add(kp)
            seen_r.add(kr)
        if reason:
            reasons[reason] += 1
            rejected.append({**rec, "reason": reason})
        else:
            kept.append(rec)

    rng = random.Random(seed)
    rng.shuffle(kept)
    selected = kept[:n_samples]
    n = len(selected)
    n_val = max(1, int(n * dcfg["split"]["validation"])) if n >= 20 else max(1, n // 10)
    n_test = max(1, int(n * dcfg["split"]["test"])) if n >= 20 else max(1, n // 10)
    splits = {"validation": selected[:n_val], "test": selected[n_val:n_val + n_test],
              "train": selected[n_val + n_test:]}
    conv = {k: [to_conversation(r, cfg["system_prompt"]) for r in v] for k, v in splits.items()}

    content = json.dumps(conv, ensure_ascii=False, sort_keys=True)
    data_version = "data-" + sha256_text(content)[:10]
    out_dir = resolve(dcfg["processed_dir"]) / data_version
    for k, rows in conv.items():
        write_jsonl(out_dir / f"{k}.jsonl", rows)
    # 人手レビュー用：採用データの一覧と、除外データ（理由付き）
    _write_review_csv(out_dir / "review_selected.csv", selected)
    _write_review_csv(out_dir / "rejected.csv", rejected, with_reason=True)

    manifest = {
        "data_version": data_version,
        "created_at": now_iso(),
        "sources": dict(Counter(r["source"] for r in records)),
        "n_source_rows": len(records),
        "n_passed_filters": len(kept),
        "n_selected": n,
        "split_sizes": {k: len(v) for k, v in conv.items()},
        "rejected_by_reason": dict(reasons.most_common()),
        "filters": f,
        "manual_exclusions": len(manual_excl),
        "system_prompt_sha": sha256_text(cfg["system_prompt"])[:10],
        "seed": seed,
    }
    if tokenizer_name:
        manifest["template_preview"] = render_preview(tokenizer_name, conv["train"][:1])
    write_json(out_dir / "manifest.json", manifest)
    (resolve(dcfg["processed_dir"]) / "LATEST").write_text(data_version, encoding="utf-8")
    return manifest


def _write_review_csv(path: Path, rows: list[dict], with_reason: bool = False) -> None:
    cols = ["id", "source"] + (["reason"] if with_reason else []) + ["prompt_ja", "response_ja"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render_preview(tokenizer_name: str, rows: list[dict]) -> str:
    """Sarashina のチャットテンプレートで展開した学習テキストの例（確認用）。"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_ref(tokenizer_name))
        r = rows[0]
        return tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
    except Exception as e:  # noqa: BLE001
        return f"(プレビュー生成失敗: {e})"


def resolve_data_version(cfg: dict, version: str | None) -> Path:
    root = resolve(cfg["data"]["processed_dir"])
    if not version or version == "latest":
        version = (root / "LATEST").read_text(encoding="utf-8").strip()
    return root / version
