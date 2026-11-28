---
base_model: sbintuitions/sarashina2.2-0.5b-instruct-v0.1
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- ja
datasets:
- kunishou/ApolloCorpus-ja
tags:
- base_model:adapter:sbintuitions/sarashina2.2-0.5b-instruct-v0.1
- lora
- sft
- transformers
- trl
- medical
- japanese
---

# lora-20261121-073926-ep2

[sbintuitions/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1) を、日本語医療QAデータ（[kunishou/ApolloCorpus-ja](https://huggingface.co/datasets/kunishou/ApolloCorpus-ja)）で LoRA 追加学習したアダプターです。
このリポジトリの Streamlit アプリで「学習後LLM」として読み込まれます。

🤗 **Hugging Face で公開中**：[hayato-kitano/sarashina2.2-0.5b-medical-ja-lora](https://huggingface.co/hayato-kitano/sarashina2.2-0.5b-medical-ja-lora)（モデルカード・使い方の詳細はこちら）

> ⚠️ **技術検証用です。** 診断・処方・個別の治療判断・実患者への医療助言には使用できません。
> 0.5B の小型モデルであり、誤った医療情報を生成する可能性があります。

## 学習の詳細

| 項目 | 値 |
|---|---|
| ベースモデル | sbintuitions/sarashina2.2-0.5b-instruct-v0.1 |
| 手法 | SFT + LoRA（PEFT / TRL） |
| データ版 | data-91258c9425（ApolloCorpus-ja から品質フィルタ後に抽出、学習 4,500件 / 検証 250件） |
| LoRA | r=16, alpha=32, dropout=0.05 |
| 対象モジュール | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 学習可能パラメーター | 10,076,160 |
| エポック | 2 |
| 学習率 | 2e-4（cosine, warmup 3%） |
| バッチ | 8 × 勾配累積 2 |
| 最大長 | 1024 |
| 精度 / 環境 | fp16 / Google Colab T4 GPU（約52分） |

再現に必要な設定・ライブラリのバージョン・学習ログは `run_meta.json` と `loss_history.csv` にあります。

### 結果

| 指標 | 値 |
|---|---|
| train loss | 1.992 |
| eval loss（最終） | 2.007（開始時 2.137 → 改善） |
| eval token accuracy（最終） | 0.606 |

## 制限事項

- 学習データは英語医療コーパスの機械翻訳であり、翻訳誤りや不正確な内容を含む可能性があります。
- 小型モデルのため、事実誤り（ハルシネーション）が起こりやすいです。根拠文書を与える RAG との併用を推奨します。
- 緊急性のある症状（胸痛・意識障害など）への回答には使わないでください。

## ライセンス

- このアダプター：Apache 2.0
- ベースモデル：MIT License（SB Intuitions）
- 学習データ：kunishou/ApolloCorpus-ja（Apache 2.0）
