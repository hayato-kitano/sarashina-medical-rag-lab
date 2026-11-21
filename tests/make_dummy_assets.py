"""【動作確認専用】ネットワーク無し・GPU無しで全工程を通すためのダミー資材を作る。

- outputs/dummy/tiny-llama : Sarashina と同じ Llama 構造の極小ランダムモデル（回答は無意味な文字列）
- data/raw/apollo_dummy.jsonl : ApolloCorpus-ja 形式の合成データ（フィルタ検証用の不正行を含む）

本番では使わないこと。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_tokenizer_and_model(out: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = [p.read_text(encoding="utf-8") for p in (ROOT / "data/rag_docs").glob("*.md")]
    specials = ["<unk>", "<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(corpus * 3, trainers.BpeTrainer(vocab_size=1200, special_tokens=specials))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                 eos_token="</s>", pad_token="<pad>")
    hf.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out)
    cfg = LlamaConfig(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      bos_token_id=hf.bos_token_id, eos_token_id=hf.eos_token_id,
                      pad_token_id=hf.pad_token_id, tie_word_embeddings=False)
    import torch
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(out)


GOOD = [
    ("高血圧の診断基準は何ですか？", "What is the diagnostic criterion for hypertension?",
     "診察室血圧で140/90mmHg以上、家庭血圧で135/85mmHg以上が高血圧の目安です。診断は複数回の測定に基づいて医師が行います。",
     "Office blood pressure of 140/90 mmHg or higher, or home blood pressure of 135/85 mmHg or higher, indicates hypertension."),
    ("HbA1cは何を反映しますか？", "What does HbA1c reflect?",
     "HbA1cは過去1〜2か月の平均的な血糖の状態を反映する指標で、糖尿病の診断や血糖コントロールの評価に用いられます。",
     "HbA1c reflects average blood glucose over the past 1 to 2 months and is used in diagnosis and monitoring of diabetes."),
    ("熱中症の応急処置を教えてください。", "What is first aid for heat stroke?",
     "涼しい場所に移動し、衣服をゆるめて首や脇の下を冷やします。意識がおかしい場合はすぐに救急車を呼びます。",
     "Move to a cool place, loosen clothing and cool the neck and armpits. Call an ambulance if consciousness is impaired."),
]


def make_apollo(out: Path, n_good: int = 400) -> None:
    rng = random.Random(0)
    rows, idx = [], 0
    topics = ["高血圧", "糖尿病", "脂質異常症", "慢性腎臓病", "インフルエンザ", "熱中症", "骨粗しょう症", "貧血"]
    aspects = ["原因", "症状", "予防", "検査", "合併症", "生活上の注意"]
    for i in range(n_good):
        t, a = rng.choice(topics), rng.choice(aspects)
        q = f"{t}の{a}について、研究{i}番の知見を踏まえて説明してください。"
        r = (f"{t}の{a}については、研究{i}番で複数の要因が検討されています。一般に、生活習慣の見直しや"
             f"定期的な検査が重要とされ、個別の対応は医師と相談して決めることが推奨されます。")
        rows.append({"index": idx, "prompt": f"Explain the {a} of {t} in study {i}.",
                     "response": f"Study {i} examined several factors related to {t}. Lifestyle review and regular check-ups are important.",
                     "prompt_ja": q, "response_ja": r}); idx += 1
    for q, qe, r, re_ in GOOD:
        rows.append({"index": idx, "prompt": qe, "response": re_, "prompt_ja": q, "response_ja": r}); idx += 1
    bad = [
        ("", "回答だけがある行です。十分な長さの回答文になっています。"),                         # 空欄
        ("不正な文字を含む質問です�", "回答の本文です。これは十分な長さの回答文になっています。"),  # 不正文字
        ("English only question here?", "This answer was not translated at all and remains in English."),  # 日本語率
        ("繰り返しの例を教えてください", "これは繰り返しです。" + "ですです" * 10 + "以上です。十分な長さ"),  # ループ
        ("あなたの病気は何ですか？", "あなたの症状は胃がんです。医師に相談する必要はありません。自己判断で服薬を中止してください。"),  # 危険
    ]
    for q, r in bad:
        rows.append({"index": idx, "prompt": "q", "response": "a" * 60, "prompt_ja": q, "response_ja": r}); idx += 1
    # 数値の不一致（原文 25, 30, 1.5 が訳に無い）
    rows.append({"index": idx, "prompt": "Dose?", "response": "Patients aged 25 to 30 received 1.5 mg daily for 12 weeks.",
                 "prompt_ja": "研究での投与量は？", "response_ja": "患者は一定期間、少量の薬剤を毎日投与されました。期間は数週間でした。"}); idx += 1
    rows.append(dict(rows[0], index=idx))  # 重複
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    make_tokenizer_and_model(ROOT / "outputs/dummy/tiny-llama")
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    make_apollo(ROOT / "data/raw/apollo_dummy.jsonl")
    print("dummy assets ready")
