# Sarashina Medical RAG Lab

Sarashina2.2-0.5B-Instruct を日本語医療QA（ApolloCorpus-ja）で **LoRA 追加学習**し、**RAG** と組み合わせて、
**学習前・学習後の回答を左右に並べて比較**する技術検証アプリです（MVP要件定義書 Ver.1.0 準拠）。

> ⚠️ 技術検証用です。診断・処方・個別の治療判断・実患者への医療助言には使用できません。

## 構成

| レイヤー | 技術 |
|---|---|
| 学習前LLM | `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` |
| 学習後LLM | 同モデル ＋ LoRAアダプター（**ベースモデル1つを読み込み、アダプターのON/OFFだけを切り替える**） |
| 学習 | SFT / LoRA（PyTorch・Transformers・PEFT・TRL）— Google Colab（T4 GPU） |
| RAG | `intfloat/multilingual-e5-small` ＋ FAISS |
| 画面 | Streamlit（ローカル / Streamlit Community Cloud） |
| 実験ログ | JSON / CSV |

```
app.py                     Streamlit アプリ（左右比較・一括評価・採点・RAG文書・実験/バージョン）
config.yaml                全設定（比較の公平性に関わる値は学習前後で共通）
medrag/
  models.py                ベース＋LoRA の読み込み、同一条件での生成（時間・トークン計測）
  pipeline.py              安全性チェック → 検索1回 → 両モデルへ同じプロンプト、4モード(A〜D)
  rag.py                   PDF/TXT/MD 読込・メタデータ・分割・e5・FAISS・版管理
  data_prep.py             学習データ前処理（フィルタ・分割・チャット形式・データ版）
  evaluation.py            一括実行・手動採点・集計・実験間比較・JMedQA
  safety.py                緊急性検出、危険な助言の検出
scripts/
  prepare_data.py          学習データ変換
  train_lora.py            LoRA 学習
  build_index.py           RAG インデックス作成
  run_eval.py              評価質問の一括実行（CLI）
  jmedqa_eval.py           JMedQA 正答率（学習前後）
notebooks/train_colab.ipynb  Colab で学習 → GitHub へ push
data/rag_docs/             RAG 文書（検証用に作成した医療知識文書 6 件）
data/eval/general_questions.csv  学習に使わない一般向け評価質問 150 問
models/adapters/           学習済み LoRA アダプター（GitHub にコミット）
tests/                     ユニットテスト・ダミーモデルでの動作確認用資材
```

## 実行手順

### 1. ローカルで起動（学習前モデルのみでも動きます）

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   / macOS・Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

初回はモデル（約1.6GB）と Embedding モデルをダウンロードします。RAG インデックスは初回起動時に自動作成されます。

### 2. Google Colab で LoRA 学習

1. https://colab.research.google.com/github/hayato-kitano/sarashina-medical-rag-lab/blob/main/notebooks/train_colab.ipynb を開く
2. ランタイムを **T4 GPU** に変更
3. 🔑シークレットに `GITHUB_TOKEN`（このリポジトリへの Contents: Read and write 権限を持つ Fine-grained token）を登録
4. 上から順に実行

ノートブックは「データ変換 → 学習 → 簡易比較 → `models/adapters/<run_id>` を push」まで行います。
push 後、`git pull` するとローカルの画面にも学習後モデルが現れます。

コマンドで実行する場合（GPU環境）:

```bash
pip install -r requirements-train.txt
python scripts/prepare_data.py --n-samples 5000 --max-source-rows 60000
python scripts/train_lora.py --epochs 2 --run-name ep2
```

学習データの品質確認：`data/processed/<data_version>/review_selected.csv`（採用分）と `rejected.csv`（除外理由付き）を確認し、
医学的に不正確な行の ID を `data/review/excluded_ids.txt` に追記して再実行します。

### 3. 評価

- 画面の「一括評価」タブ、または `python scripts/run_eval.py --modes A B C D`
- 「採点・集計」タブで、質問ごとに左右比較しながら 正確性・関連性・根拠整合性・根拠不足時の応答・安全性 を採点 → 集計表・CSV
- JMedQA：`python scripts/jmedqa_eval.py --prepare` → `python scripts/jmedqa_eval.py --limit 300`（画像不要のテキスト問題のみ）

### 4. Streamlit Community Cloud で公開

1. [share.streamlit.io](https://share.streamlit.io) に GitHub アカウントでサインイン
2. 「Create app」→ このリポジトリ / ブランチ `main` / Main file `app.py`
3. Advanced settings で Python 3.12 を選択して Deploy

無料枠のメモリは 1 アプリ約 2.7GB です。本アプリは CPU では bfloat16 で読み込み（約1.6GB）収まるよう調整していますが、
同時アクセスが多いと上限を超えることがあります。その場合は Hugging Face Spaces など、より大きな環境を検討してください。
CPU 実行のため、回答生成には 1 回あたり数十秒〜1 分程度かかります（`generation.max_new_tokens` で調整）。

## 比較の公平性（要件 6.3）

- 同じ質問・同じシステムプロンプト・同じ生成パラメーター（`do_sample=False`）・同じ最大出力トークン
- RAG ON 時は検索を 1 回だけ行い、**同一の検索結果を両モデルに渡す**
- 回答ごとに base model / revision / adapter ID / data version / RAG index version / プロンプトハッシュを記録

| モード | LoRA | RAG | 見たい効果 |
|---|---|---|---|
| A | OFF | OFF | — |
| B | ON | OFF | A対B：LoRA学習の効果 |
| C | OFF | ON | A対C：RAG単体の効果 |
| D | ON | ON | C対D：同じRAG下のLoRA効果 ／ B対D：RAG追加効果 |

## 安全性（要件 9）

- 緊急性が疑われる相談（胸痛・意識障害・自殺念慮など）は、**モデルで生成せず**受診・相談先の案内のみ表示（`safety.emergency_action`）
- RAG 時に閾値以上の根拠が見つからなければ回答を保留（`rag.min_score`、`rag.insufficient_action`）
- 回答中の「個別診断」「自己判断での服薬変更」「受診不要の断定」等の表現を自動検出して警告（最終判断は評価者）
- 外部 API 不要・ローカル実行可能。患者の個人情報を入力・学習に使わないこと

## データ・ライセンス

| データ | 用途 | 注意 |
|---|---|---|
| kunishou/ApolloCorpus-ja | 学習（メイン） | Apache 2.0 表示。機械翻訳誤りを含むため品質フィルタ＋人手確認 |
| li-lab/JP-AlpaCare-MedInstruct-52k | 学習（任意） | 原典の非商用条件。`--include-alpacare N --i-confirmed-license` |
| llm-jp/AnswerCarefully | 安全性（任意） | 利用規約への同意が必要。JSONL 化して `--extra-jsonl` |
| SIP-med-LLM/JMedQA | 評価 | 利用条件を確認のうえ使用 |
| data/rag_docs | RAG | 検証用に作成した文書。実運用では使用許諾を確認した公的資料に差し替え |

## 開発者向け：ネットワーク・GPU 無しでの動作確認

```bash
python tests/make_dummy_assets.py                 # 極小ダミーモデルと合成データ
export MEDRAG_CONFIG=tests/config.test.yaml
python scripts/prepare_data.py --source-file data/raw/apollo_dummy.jsonl
python scripts/train_lora.py --max-steps 10
python scripts/build_index.py
python scripts/run_eval.py --limit 5
streamlit run app.py
pytest -q
```
