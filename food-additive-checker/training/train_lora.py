"""Sarashina2.2-0.5B-Instruct に LoRA を学習する。

  python -m training.train_lora --train training/data/train.jsonl --val training/data/val.jsonl

T4 (16GB) で約30〜60分（2,000件・2エポック）が目安。
"""
from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class SFTDataset(torch.utils.data.Dataset):
    """プロンプト部分は loss を計算せず（-100）、アシスタントの回答だけを学習する。"""

    def __init__(self, rows, tok, max_len):
        self.items = []
        skipped = 0
        for r in rows:
            msgs = r["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True, return_dict=False)
            full = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
            if len(full) > max_len:
                skipped += 1
                continue
            labels = [-100] * len(prompt) + full[len(prompt):]
            self.items.append({"input_ids": full, "labels": labels})
        print(f"dataset: {len(self.items)} 件（長すぎて除外 {skipped} 件）")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def f(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--train", default="training/data/train.jsonl")
    ap.add_argument("--val", default="training/data/val.jsonl")
    ap.add_argument("--out", default="outputs/sarashina-additive-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=a.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train = SFTDataset(load_jsonl(a.train), tok, a.max_len)
    val = SFTDataset(load_jsonl(a.val), tok, a.max_len)

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to="none",
        remove_unused_columns=False,
    )
    Trainer(
        model=model, args=args, train_dataset=train, eval_dataset=val,
        data_collator=collate(tok.pad_token_id),
    ).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"LoRA adapter saved → {a.out}")


if __name__ == "__main__":
    main()
