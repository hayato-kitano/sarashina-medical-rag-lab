"""Sarashina2.2-0.5B-Instruct + LoRA による根拠付き回答生成と、根拠の検証。"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import config

RISK_LABEL = {"high": "高", "medium": "中", "low": "低", "info": "-"}

SYSTEM_PROMPT = (
    "あなたは食品添加物とがんの関係を一般の人に説明するアシスタントです。"
    "必ず【参考資料】に書かれている内容だけを根拠に回答し、根拠にした資料の番号を各文の末尾に[D1]のように付けてください。"
    "資料に書かれていないことは推測で書かず、診断や断定はしないでください。"
)


# ---------------------------------------------------------------- プロンプト
def build_context(hits: list[dict], guides: list) -> tuple[str, dict[str, dict]]:
    """参考資料ブロックと、[D番号]→KB文書 の対応表を作る。"""
    refs, lines = {}, []
    docs = [h["doc"] for h in hits] + list(guides)
    for i, d in enumerate(docs, 1):
        key = f"D{i}"
        refs[key] = d.data
        x = d.data
        lines.append(
            f"[{key}] {x['name']}｜用途:{x['category']}｜IARC分類:{x['iarc_group']}｜注意度:{RISK_LABEL[x['risk_level']]}\n"
            f"概要:{x['summary']}\n根拠:{x['evidence']}\n日本での扱い:{x['japan_status']}\n助言:{x['advice']}"
        )
    return "\n\n".join(lines), refs


def build_user_prompt(additives: list[str], context: str) -> str:
    add = "、".join(additives) if additives else "（読み取れませんでした）"
    return (
        f"【食品ラベルの添加物】\n{add}\n\n【参考資料】\n{context}\n\n"
        "質問：この食品の添加物に、がんとの関連が指摘されているものはありますか？"
        "該当する添加物ごとに、注意度・IARC分類・理由・日本での扱いを説明し、最後に全体のまとめを書いてください。"
    )


def build_messages(additives, hits, guides):
    context, refs = build_context(hits, guides)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(additives, context)},
    ], refs


# ---------------------------------------------------------------- 模範回答（学習データ兼フォールバック）
def template_answer(hits: list[dict], refs: dict[str, dict]) -> str:
    key_of = {v["id"]: k for k, v in refs.items()}
    g = key_of.get("G01")
    gref = f"[{g}]" if g else ""
    if not hits:
        return (
            "## 判定結果\n"
            f"読み取れた添加物の中に、参考資料でがんとの関連が指摘されているものは見つかりませんでした。{gref}\n\n"
            "## まとめ\n"
            f"ただし「香料」「乳化剤」などの一括名で表示される添加物は個別の物質が分からないため、判定の対象外です。[{key_of.get('G02', 'D1')}]"
        )
    cnt = {lv: sum(h["doc"].data["risk_level"] == lv for h in hits) for lv in ("high", "medium", "low")}
    out = [
        "## 判定結果",
        f"がんとの関連について情報がある添加物が{len(hits)}件見つかりました"
        f"（注意度 高:{cnt['high']}件・中:{cnt['medium']}件・低:{cnt['low']}件）。{gref}",
        "",
    ]
    for n, h in enumerate(hits, 1):
        x = h["doc"].data
        k = key_of[x["id"]]
        out += [
            f"### {n}. {h['matched']} → {x['name']}（注意度：{RISK_LABEL[x['risk_level']]}）",
            f"- IARC分類：{x['iarc_group']}[{k}]",
            f"- 概要：{x['summary']}[{k}]",
            f"- 根拠：{x['evidence']}[{k}]",
            f"- 日本での扱い：{x['japan_status']}[{k}]",
            f"- アドバイス：{x['advice']}[{k}]",
            "",
        ]
    highs = [h for h in hits if h["doc"].data["risk_level"] == "high"]
    out.append("## まとめ")
    if highs:
        names = "、".join(h["doc"].data["name"] for h in highs)
        cites = "".join(f"[{key_of[h['doc'].data['id']]}]" for h in highs)
        out.append(f"特に{names}はIARCでグループ2A以上に関連づけられており、摂取頻度に注意が必要です。{cites}")
    if cnt["low"]:
        out.append(f"注意度「低」の添加物は、現時点でヒトへの発がん性を示す確立した証拠がないものです。{gref}")
    out.append(f"IARC分類は有害性の証拠の強さを示すもので、実際の摂取量でのリスクの大きさではありません。{gref}")
    return "\n".join(out)


# ---------------------------------------------------------------- 根拠の検証
_CITE = re.compile(r"\[(D\d+)\]")


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s|\[D\d+\]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def verify_citations(answer: str, refs: dict[str, dict]) -> dict:
    """各行の出典番号が実在するか、その資料と内容が一致しているかを確認する。"""
    lines = [l for l in answer.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    report, used, invalid = [], set(), set()
    for l in lines:
        keys = _CITE.findall(l)
        valid = [k for k in keys if k in refs]
        invalid |= set(keys) - set(valid)
        used |= set(valid)
        if not valid:
            report.append({"text": l, "status": "出典なし", "support": 0.0})
            continue
        src = " ".join(" ".join(str(v) for v in refs[k].values()) for k in valid)
        b = _bigrams(l)
        sup = len(b & _bigrams(src)) / max(1, len(b))
        report.append({"text": l, "status": "根拠あり" if sup >= 0.5 else "根拠弱い", "support": round(sup, 2), "refs": valid})
    supported = sum(r["status"] == "根拠あり" for r in report)
    return {
        "used_refs": sorted(used, key=lambda k: int(k[1:])),
        "invalid_refs": sorted(invalid),
        "coverage": round(supported / max(1, len(report)), 2),
        "lines": report,
    }


def clean_answer(answer: str, refs: dict[str, dict]) -> str:
    """存在しない出典番号を取り除く。"""
    return _CITE.sub(lambda m: m.group(0) if m.group(1) in refs else "", answer)


# ---------------------------------------------------------------- モデル
@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.LLM_BASE)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(config.LLM_BASE, dtype=torch.bfloat16, device_map="auto")
    else:  # CPU（Streamlit Cloud）はメモリ節約のためbf16で読み込む
        model = AutoModelForCausalLM.from_pretrained(config.LLM_BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    adapter = Path(config.LORA_ADAPTER)
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        print(f"[generator] LoRA adapter loaded: {adapter}")
    else:
        print(f"[generator] LoRA adapter not found at {adapter}; base model only")
    model.eval()
    return tok, model


def generate(messages: list[dict], max_new_tokens: int | None = None) -> str:
    import torch

    tok, model = _load()
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    ids = enc["input_ids"]
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def answer(additives: list[str], hits: list[dict], guides: list) -> dict:
    messages, refs = build_messages(additives, hits, guides)
    fallback = False
    if config.MOCK:
        text = template_answer(hits, refs)
    else:
        text = clean_answer(generate(messages), refs)
    check = verify_citations(text, refs)
    # 根拠の裏付けが弱い回答は採用せず、資料から組み立てた回答に差し替える
    if not config.MOCK and (check["coverage"] < 0.6 or not check["used_refs"]):
        text, fallback = template_answer(hits, refs), True
        check = verify_citations(text, refs)
    return {"text": text, "refs": refs, "verification": check, "fallback": fallback}


# ---------------------------------------------------------------- 要約モード（CPU向け）
SUMMARY_SYSTEM = (
    "あなたは食品添加物とがんの関係を一般の人に説明するアシスタントです。"
    "【参考資料】に書かれている内容だけを使い、3〜5文の短い日本語でまとめてください。"
    "各文の末尾に根拠にした資料の番号を[D1]のように付けてください。資料にないことは書かないでください。\n"
    "書き方の例：\n"
    "亜硝酸ナトリウムはIARC分類で2Aに関連づけられ、加工肉の発色剤として使われます。[D1]\n"
    "IARC分類は有害性の証拠の強さを示すもので、摂取量でのリスクではありません。[D3]"
)


def build_summary_messages(additives, hits, guides):
    _, refs = build_context(hits, guides)
    lines = [
        f"[{k}] {v['name']}｜注意度:{RISK_LABEL[v['risk_level']]}｜IARC分類:{v['iarc_group']}｜{v['summary']}"
        for k, v in refs.items()
    ]
    add = "、".join(additives) if additives else "（読み取れませんでした）"
    user = (
        f"【食品ラベルの添加物】\n{add}\n\n【参考資料】\n" + "\n".join(lines) +
        "\n\n質問：この食品の添加物について、がんとの関連の観点から要点をまとめてください。"
    )
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}], refs


def template_summary(hits, refs) -> str:
    key_of = {v["id"]: k for k, v in refs.items()}
    g = f"[{key_of['G01']}]" if "G01" in key_of else ""
    if not hits:
        return (f"読み取れた添加物の中に、参考資料でがんとの関連が指摘されているものは見つかりませんでした。{g}"
                f"一括名（香料・乳化剤など）で表示された添加物は判定できません。[{key_of.get('G02', 'D1')}]")
    out = [f"がんとの関連について情報がある添加物が{len(hits)}件見つかりました。{g}"]
    for lv, word in (("high", "IARCでグループ2A以上に関連づけられており、摂取頻度に注意が必要です"),
                     ("medium", "グループ2Bに分類されているか、海外で規制上の論点がある添加物です"),
                     ("low", "現時点でヒトへの発がん性を示す確立した証拠はありません")):
        hs = [h for h in hits if h["doc"].data["risk_level"] == lv]
        if hs:
            names = "、".join(h["doc"].data["name"] for h in hs)
            cites = "".join(f"[{key_of[h['doc'].id]}]" for h in hs)
            out.append(f"{names}は{word}。{cites}")
    out.append(f"IARC分類は有害性の証拠の強さを示すもので、通常の摂取量でのリスクの大きさではありません。{g}")
    return "\n".join(out)


def _ref_text(ref: dict) -> str:
    return " ".join(str(v) for v in ref.values())


def ground_sentences(raw: str, refs: dict[str, dict], min_support: float = 0.5) -> list[str]:
    """モデル出力を文に分け、出典の内容で裏付けられる文だけを残す。

    - Sarashinaが付けた[D番号]が実在し、その資料と内容が一致する文 → 採用
    - 出典番号がない文は、最も内容が一致する資料の番号を補う（一致率が高い場合のみ）
    """
    raw = clean_answer(raw, refs)
    parts = [p.strip() for p in re.split(r"(?<=[。．！？])|\n", raw) if p and p.strip()]
    # 「…です。[D1]」のように句点の後ろに来た出典番号を前の文に戻す
    sents: list[str] = []
    for p in parts:
        if sents and re.fullmatch(r"(\[D\d+\])+", p):
            sents[-1] += p
        else:
            sents.append(p)
    kept, seen = [], set()
    for snt in sents:
        body = _CITE.sub("", snt).strip()
        if len(body) < 8 or body in seen:
            continue
        b = _bigrams(body)
        keys = [k for k in _CITE.findall(snt) if k in refs]
        if keys:
            sup = len(b & _bigrams(" ".join(_ref_text(refs[k]) for k in keys))) / max(1, len(b))
        else:
            scored = [(len(b & _bigrams(_ref_text(v))) / max(1, len(b)), k) for k, v in refs.items()]
            sup, best = max(scored) if scored else (0, None)
            if best and sup >= 0.6:
                keys = [best]
        if keys and sup >= min_support:
            seen.add(body)
            kept.append(body + "".join(f"[{k}]" for k in dict.fromkeys(keys)))
    return kept


def summarize(additives, hits, guides) -> dict:
    """Sarashinaに短い要約を書かせ、根拠が確認できた文だけを採用する。

    根拠のある文が1つもなければ資料からの要約に差し替える。"""
    messages, refs = build_summary_messages(additives, hits, guides)
    text, raw, fallback, err = "", "", False, None
    if config.MOCK or config.LLM_MODE == "off":
        text, fallback = template_summary(hits, refs), True
    else:
        try:
            raw = generate(messages, max_new_tokens=config.SUMMARY_MAX_NEW_TOKENS)
            kept = ground_sentences(raw, refs)
            if kept:
                text = "\n".join(kept)
        except Exception as e:  # メモリ不足などでもアプリを止めない
            err = str(e)
        if not text:
            text, fallback = template_summary(hits, refs), True
    check = verify_citations(text, refs)
    return {"text": text, "raw": raw, "refs": refs, "verification": check, "fallback": fallback, "error": err}
