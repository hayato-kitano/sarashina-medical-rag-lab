# 添加物チェック（Built with Sarashina）

食品ラベル（原材料名）の写真をアップすると、**がんとの関連が指摘されている添加物**を見つけ、
**根拠（出典）つき**で説明するWebアプリです。画像処理・検索・回答生成をすべて Sarashina 系モデルで構成しています。

機能は2つだけです。

1. 写真をアップする
2. 危険な添加物を説明する（注意度・IARC分類・理由・日本での扱い＋出典）

---

## 仕組み

```
写真 ─▶ ① Sarashina2.2-OCR ─▶ ② 原材料名の解析 ─▶ ③ RAG検索 ─▶ ④ Sarashina2.2-0.5B-Instruct + LoRA ─▶ ⑤ 根拠チェック ─▶ 画面
        （画像→文字）          （「/」以降を添加物に分割）  （添加物ナレッジベース）  （[D1]形式で出典つき回答）     （出典番号と内容の照合）
```

| 段階 | 使用モデル / 処理 | ファイル |
|---|---|---|
| ① 画像処理 | `sbintuitions/sarashina2.2-ocr`（Sarashina2.2-3B-Instruct + SigLIP2。文書OCR特化）<br>切替可：`sbintuitions/sarashina2.2-vision-3b` | `src/vision.py` |
| ② ラベル解析 | 「原材料名」欄の抽出、「/」で原材料と添加物を区分、「発色剤(亜硝酸Na)」の括弧内も展開 | `src/label_parser.py` |
| ③ RAG | 別名辞書の最長一致 ＋ `sbintuitions/sarashina-embedding-v1-1b` の意味検索（OCRの誤読・表記ゆれを補完）。モデルなし環境では文字2-gram BM25 | `src/rag.py` |
| ④ 回答生成 | `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` に LoRA を適用。参考資料だけを根拠に、各文末へ `[D番号]` を付けて回答 | `src/generator.py` |
| ⑤ 根拠チェック | 存在しない出典番号の除去、各文と出典本文の一致率を計算。一致率60%未満なら資料から組み立てた説明に差し替え | `src/generator.py` |

「根拠を示す機能」は、Sarashina が回答中に出典番号を出力するように LoRA で学習させ、その番号をアプリ側で検証する方式で実現しています。画面では `D1` などのタグをタップすると、該当する出典（本文とURL）にジャンプします。

## 準備したデータ

### 添加物ナレッジベース（RAG用）`data/additives_kb.jsonl`

18種類の添加物と2本の解説（IARC分類の読み方／日本の添加物表示の見分け方）。1行1件で、次の項目を持ちます。

| 項目 | 内容 |
|---|---|
| `aliases` | ラベル上の表記ゆれ（例：亜硝酸Na、亜硝酸ナトリウム、E250） |
| `risk_level` | 高＝IARC 1/2A関連、中＝2B または規制上の論点あり、低＝3・未分類 |
| `iarc_group` / `summary` / `evidence` / `japan_status` / `advice` | 回答の根拠になる本文 |
| `sources` | 出典URL（IARC、消費者庁、FDA、EFSA など） |

収録：亜硝酸ナトリウム、硝酸塩、カラメル色素、赤色3号、二酸化チタン、臭素酸カリウム、BHA、アスパルテーム、OPP、安息香酸Na、カラギーナン、BHT、サッカリン、赤色2号、その他タール色素、過酸化水素、亜硫酸塩、スクラロース/アセスルファムK

行を追加するだけで対象を増やせます（再学習なしでも検索・回答に反映されます。表現を安定させたい場合はデータ生成と学習をやり直してください）。

### LoRA学習データ `training/data/{train,val}.jsonl`

ナレッジベースから架空のラベルを2,000件作り、**本番と同じ解析→検索→プロンプト**を通して、出典番号つきの模範回答を教師データにしています（添加物0〜5件、該当なしの例も含む）。

```bash
python -m training.make_sft_data --n 2000
```

---

## Streamlit版（公開用）

`streamlit_app.py` が Streamlit Community Cloud 用の入口です。無料枠はGPUがないため、自動で次の構成になります。

| 段階 | GPUあり | Streamlit Cloud（CPU） |
|---|---|---|
| 文字読み取り | Sarashina2.2-OCR | Tesseract（日本語）※読み取り結果は画面で修正可 |
| RAG検索 | 辞書＋Sarashina-Embedding | 辞書＋BM25 |
| 説明生成 | Sarashina 0.5B＋LoRA（詳細説明） | Sarashina 0.5B（CPU・短い要約）＋資料からの詳細カード |

デプロイ設定：Branch `food-additive-checker`、Main file path `food-additive-checker/streamlit_app.py`。`food-additive-checker/requirements.txt`（CPU版torch）と、ブランチのルートにある `packages.txt`（tesseract-ocr / tesseract-ocr-jpn。Streamlit Cloudはルートの packages.txt だけを読みます）を使います。既存アプリのある main ブランチには影響しません。

```bash
streamlit run streamlit_app.py
```

## セットアップ

```bash
pip install -r requirements-gpu.txt   # GPU環境（FastAPI版・学習）
```

### まず動きを見る（GPU不要・モデル不要）

```bash
SARASHINA_MOCK=1 EMBED_BACKEND=bm25 uvicorn app.main:app --port 8000
# → http://localhost:8000 （OCRは data/sample_label.txt を返すモック）
SARASHINA_MOCK=1 python -m pytest tests
```

### LoRA学習（GPU 1枚。Google Colab T4 で可）

```bash
python -m training.make_sft_data --n 2000
python -m training.train_lora            # → outputs/sarashina-additive-lora
python -m training.evaluate --limit 50   # 検出率・出典の正しさ・根拠一致率
```

主な設定：rank 16 / alpha 32 / 全線形層 / lr 2e-4 / 2エポック / プロンプト部分はlossから除外。

### 本番起動

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

GPUメモリ目安：OCR 3B（約8GB）＋ 0.5B（約1GB）＋ Embedding 1B（約2.5GB）で 16GB GPU に収まります。
メモリが足りない場合は `EMBED_BACKEND=bm25` で埋め込みモデルを省けます（辞書一致が主なので精度への影響は小さめ）。

| 環境変数 | 既定値 | 説明 |
|---|---|---|
| `VISION_MODEL` | `sbintuitions/sarashina2.2-ocr` | `sbintuitions/sarashina2.2-vision-3b` に切替可 |
| `LLM_BASE` | `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` | 回答生成のベースモデル |
| `LORA_ADAPTER` | `outputs/sarashina-additive-lora` | LoRAアダプタの場所（なければベースのみ） |
| `EMBED_BACKEND` | `sarashina` | `bm25` でモデル不要の検索 |
| `SARASHINA_MOCK` | `0` | `1` でモデルを読み込まずに動作確認 |

## API

`POST /api/analyze`（multipart, `file`=画像）→ JSON

```json
{
  "additives": ["発色剤(亜硝酸Na)", "..."],
  "findings": [{"ref": "D1", "label_text": "亜硝酸Na", "name": "亜硝酸ナトリウム", "risk_level": "high", "iarc_group": "2A ..."}],
  "explanation": "## 判定結果 ... [D1]",
  "sources": {"D1": {"name": "...", "evidence": "...", "sources": [{"title": "...", "url": "..."}]}},
  "verification": {"coverage": 0.9, "used_refs": ["D1"], "invalid_refs": []},
  "fallback_used": false
}
```

---

## 注意事項

- 結果は公開情報に基づく一般的な情報で、医学的な診断ではありません。IARC分類は「がんを起こす証拠の強さ」であり、通常の摂取量でのリスクを示すものではありません（画面にも表示しています）。
- 「香料」「乳化剤」などの一括名表示は具体的な物質が分からないため判定できません。
- ナレッジベースの内容は2026年11月時点の公開情報をもとに作成しています。公開前に専門家による確認と、出典URLの最新化を行ってください。
- ライセンス：Sarashina2.2-OCR は MIT。sarashina-embedding-v1-1b は非商用ライセンスで提供されています（商用利用時はモデルカードで条件を確認し、必要なら `EMBED_BACKEND=bm25` に切替）。その他は各モデルカードを確認してください。Sarashina 派生モデルを配布する場合は名前の先頭に「Sarashina」を付け、システムでは「Built with Sarashina」を表示する必要があります（画面下部に表示済み）。
