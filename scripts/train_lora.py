"""LoRA ファインチューニング（SFT / PEFT / TRL）。

例:
  python scripts/train_lora.py                         # 最新データ版・config の設定で学習
  python scripts/train_lora.py --data-version data-xxxx --epochs 3 --run-name ep3
  python scripts/train_lora.py --max-steps 20          # 動作確認用の短時間学習

保存物（outputs/adapters/<run_id>/）:
  adapter_model.safetensors / adapter_config.json  … LoRAアダプター
  run_meta.json                                    … 学習条件・データ版・モデル版・損失履歴
  loss_history.csv                                 … ステップごとの損失
  README.md                                        … モデルカード（PEFT の自動生成テンプレートを上書き）
  outputs/experiments.csv に1行追記、outputs/adapters/LATEST を更新
"""
import argparse
import csv
import json
import time

import _bootstrap  # noqa: F401
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from medrag.config import load_config, model_ref, resolve
from medrag.data_prep import resolve_data_version
from medrag.versioning import library_versions, now_id, now_iso, read_json, write_json


def _warmup_kwargs(ratio: float) -> dict:
    """transformers v5 では warmup_ratio が廃止され、warmup_steps に 1 未満の値を渡すと比率として扱われる。"""
    import inspect
    if "warmup_ratio" in inspect.signature(SFTConfig.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


# モデルカードに載せる Hugging Face 上の公開先と、学習データ名 → データセットID の対応
HF_REPO_ID = "hayato-kitano/sarashina2.2-0.5b-medical-ja-lora"
DATASET_IDS = {
    "ApolloCorpus-ja": "kunishou/ApolloCorpus-ja",
    "JP-AlpaCare-MedInstruct-52k": "li-lab/JP-AlpaCare-MedInstruct-52k",
    "AnswerCarefully": "llm-jp/AnswerCarefully",
}


def write_model_card(out_dir, meta: dict, final_eval: dict) -> None:
    """save_pretrained() が作る PEFT のテンプレート README（[More Information Needed]）を、
    run_meta の内容から作ったモデルカードで上書きする。"""
    t, lora = meta["training"], meta["lora"]
    base = meta["base_model"]
    sources = list((meta.get("data_sources") or {}).keys())
    ds_ids = [DATASET_IDS.get(s, s) for s in sources]
    ds_links = "、".join(f"[{d}](https://huggingface.co/datasets/{d})" if "/" in d else d for d in ds_ids) or "-"
    evals = [h["eval_loss"] for h in meta.get("log_history", []) if "eval_loss" in h]
    first_eval = evals[0] if evals else None
    final_loss = meta.get("final_eval_loss")
    acc = final_eval.get("eval_mean_token_accuracy")
    minutes = round(meta["train_runtime_sec"] / 60)
    device = "GPU" if meta["device"] == "cuda" else "CPU"
    steps = f"（max_steps={meta['max_steps']}）" if meta.get("max_steps", -1) > 0 else ""

    eval_row = f"{final_loss:.3f}" if final_loss is not None else "-"
    if final_loss is not None and first_eval is not None:
        eval_row += f"（開始時 {first_eval:.3f}）"
    front = ["---", f"base_model: {base}", "library_name: peft", "pipeline_tag: text-generation",
             "license: apache-2.0", "language:", "- ja"]
    if ds_ids:
        front += ["datasets:"] + [f"- {d}" for d in ds_ids if "/" in d]
    front += ["tags:", f"- base_model:adapter:{base}", "- lora", "- sft", "- transformers", "- trl",
              "- medical", "- japanese", "---"]
    other = [s for s in sources if s != "ApolloCorpus-ja"]
    data_license = "、".join(f"{DATASET_IDS.get(s, s)}" for s in sources) or "-"

    card = "\n".join(front) + f"""

# {meta['run_id']}

[{base}](https://huggingface.co/{base}) を、日本語医療QAデータ（{ds_links}）で LoRA 追加学習したアダプターです。
このリポジトリの Streamlit アプリで「学習後LLM」として読み込まれます。

🤗 **Hugging Face で公開中**：[{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID})（モデルカード・使い方の詳細はこちら。公開版はこの run と異なる場合があります）

> ⚠️ **技術検証用です。** 診断・処方・個別の治療判断・実患者への医療助言には使用できません。
> 小型モデルであり、誤った医療情報を生成する可能性があります。

## 学習の詳細

| 項目 | 値 |
|---|---|
| ベースモデル | {base} |
| 手法 | SFT + LoRA（PEFT / TRL） |
| データ版 | {meta['data_version']}（学習 {meta['n_train']:,}件 / 検証 {meta['n_validation']:,}件） |
| LoRA | r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']} |
| 対象モジュール | {', '.join(lora['target_modules'])} |
| 学習可能パラメーター | {meta['trainable_params']:,} |
| エポック | {t['num_train_epochs']:g}{steps} |
| 学習率 | {t['learning_rate']:g}（{t['lr_scheduler_type']}, warmup {t['warmup_ratio']:.0%}） |
| バッチ | {t['per_device_train_batch_size']} × 勾配累積 {t['gradient_accumulation_steps']} |
| 最大長 | {t['max_length']} |
| 精度 / 環境 | {meta['precision']} / {device}（約{minutes}分） |

再現に必要な設定・ライブラリのバージョン・学習ログは `run_meta.json` と `loss_history.csv` にあります。

### 結果

| 指標 | 値 |
|---|---|
| train loss | {meta['train_loss']:.3f} |
| eval loss（最終） | {eval_row} |
| eval token accuracy（最終） | {f"{acc:.3f}" if acc is not None else "-"} |

## 制限事項

- 学習データには英語医療コーパスの機械翻訳が含まれ、翻訳誤りや不正確な内容を含む可能性があります。
- 小型モデルのため、事実誤り（ハルシネーション）が起こりやすいです。根拠文書を与える RAG との併用を推奨します。
- 緊急性のある症状（胸痛・意識障害など）への回答には使わないでください。

## ライセンス

- このアダプター：Apache 2.0{"（ただし学習データの利用条件にも従ってください）" if other else ""}
- ベースモデル：SB Intuitions のライセンス（{base} のモデルカードを参照）
- 学習データ：{data_license}（各データセットの利用条件に従う）
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data-version", default="latest")
    ap.add_argument("--base-model", help="既定: config の model.base_model")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--no-set-latest", action="store_true", help="LATEST を更新しない")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = dict(cfg["training"])
    lcfg = dict(tcfg["lora"])
    if args.epochs is not None:
        tcfg["num_train_epochs"] = args.epochs
    if args.lr is not None:
        tcfg["learning_rate"] = args.lr
    if args.lora_r is not None:
        lcfg["r"], lcfg["alpha"] = args.lora_r, args.lora_r * 2
    base_model = model_ref(args.base_model or cfg["model"]["base_model"])

    data_dir = resolve_data_version(cfg, args.data_version)
    data_manifest = read_json(data_dir / "manifest.json")
    ds = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"),
                                          "validation": str(data_dir / "validation.jsonl")})
    ds = ds.remove_columns([c for c in ("id", "source") if c in ds["train"].column_names])

    run_id = now_id("lora-") + (f"-{args.run_name}" if args.run_name else "")
    out_dir = resolve(tcfg["output_dir"]) / run_id
    tmp_dir = resolve(tcfg["output_dir"]).parent / "checkpoints" / run_id

    cuda = torch.cuda.is_available()
    # T4（compute capability 7.5）では is_bf16_supported() が True を返すがエミュレーションで、
    # TRL のカーネルが失敗するため、実際に bf16 対応（Ampere 以降 = 8.0+）の GPU に限定する
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 非対応GPU（Colab の T4 など）は重みを fp32 で読み込み fp16 AMP で学習する
    # （重みを fp16 で読み込むと LoRA 勾配の unscale でエラーになるため）
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16 if bf16 else torch.float32)
    base_revision = getattr(model.config, "_commit_hash", None) or "local"

    peft_config = LoraConfig(r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
                             lora_dropout=float(lcfg["dropout"]),
                             target_modules=list(lcfg["target_modules"]), task_type="CAUSAL_LM")
    sft_args = SFTConfig(
        output_dir=str(tmp_dir),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        max_steps=args.max_steps,
        learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        **_warmup_kwargs(float(tcfg["warmup_ratio"])),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=int(tcfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(tcfg["eval_steps"]),
        save_strategy="no",
        max_length=int(tcfg["max_length"]),
        completion_only_loss=True,   # 回答部分のみに損失をかける
        packing=False,
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=cuda,
        report_to="none",
        seed=int(cfg["data"]["seed"]),
    )
    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], processing_class=tokenizer,
                         peft_config=peft_config)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"[{run_id}] data={data_manifest['data_version']} train={len(ds['train'])} "
          f"val={len(ds['validation'])} trainable={trainable:,}/{total:,}")

    t0 = time.time()
    train_out = trainer.train()
    final_eval = trainer.evaluate()
    elapsed = time.time() - t0

    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    history = trainer.state.log_history
    with open(out_dir / "loss_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "train_loss", "eval_loss", "learning_rate"])
        for h in history:
            if "loss" in h or "eval_loss" in h:
                w.writerow([h.get("step"), h.get("epoch"), h.get("loss", ""), h.get("eval_loss", ""),
                            h.get("learning_rate", "")])

    meta = {
        "run_id": run_id,
        "created_at": now_iso(),
        "base_model": base_model,
        "base_revision": base_revision,
        "data_version": data_manifest["data_version"],
        "data_sources": data_manifest.get("sources"),
        "n_train": len(ds["train"]),
        "n_validation": len(ds["validation"]),
        "training": {k: v for k, v in tcfg.items() if k not in ("lora", "output_dir")},
        "max_steps": args.max_steps,
        "lora": lcfg,
        "trainable_params": trainable,
        "train_loss": train_out.training_loss,
        "final_eval_loss": final_eval.get("eval_loss"),
        "train_runtime_sec": round(elapsed, 1),
        "device": "cuda" if cuda else "cpu",
        "precision": "bf16" if bf16 else ("fp16" if cuda else "fp32"),
        "libraries": library_versions(),
        "log_history": history,
    }
    write_json(out_dir / "run_meta.json", meta)
    write_model_card(out_dir, meta, final_eval)

    exp_csv = resolve(tcfg["output_dir"]).parent / "experiments.csv"
    row = {"run_id": run_id, "created_at": meta["created_at"], "base_model": base_model,
           "base_revision": base_revision, "data_version": meta["data_version"],
           "n_train": meta["n_train"], "epochs": tcfg["num_train_epochs"], "max_steps": args.max_steps,
           "lr": tcfg["learning_rate"], "lora_r": lcfg["r"], "lora_alpha": lcfg["alpha"],
           "train_loss": round(meta["train_loss"], 4),
           "final_eval_loss": round(meta["final_eval_loss"], 4) if meta["final_eval_loss"] else "",
           "runtime_sec": meta["train_runtime_sec"]}
    new = not exp_csv.exists()
    exp_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(exp_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    if not args.no_set_latest:
        (resolve(tcfg["output_dir"]) / "LATEST").write_text(run_id, encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("run_id", "data_version", "train_loss", "final_eval_loss",
                                            "train_runtime_sec")}, ensure_ascii=False, indent=2))
    print(f"アダプター保存先: {out_dir}")


if __name__ == "__main__":
    main()
