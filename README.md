# 半導体サイクル分析ダッシュボード

## 概要

経産省IIP（鉱工業指数）の電子部品・デバイス工業データと日本半導体関連12銘柄の
株価データを組み合わせた分析ダッシュボード。
半導体サイクルの局面判定（回復期/好況期/後退期/不況期）と株価リターンの関係を可視化。
マクロ経済指標（米金利・為替・GDP・CPI・ISM PMI）のオーバーレイ表示にも対応。

## 前提条件

- **Python 3.12+**（`dict[str, str] | None` 等の型ヒント構文を使用しているため）
- ネット接続推奨（株価データ取得に yfinance を使用）

## 対象銘柄

| コード | 銘柄名 |
|--------|--------|
| 8035.T | 東京エレクトロン |
| 6857.T | アドバンテスト |
| 6920.T | レーザーテック |
| 7735.T | SCREENホールディングス |
| 6723.T | ルネサスエレクトロニクス |
| 6146.T | ディスコ |
| 6963.T | ローム |
| 4063.T | 信越化学工業 |
| 3436.T | SUMCO |
| 6526.T | ソシオネクスト |
| 6315.T | TOWA |
| 6728.T | アルバック |

## セットアップ

```bash
git clone https://github.com/ShinichiMine/semiconductor-cycle-analysis.git
cd semiconductor-cycle-analysis
pip install -r requirements.txt
```

## 使い方

### ダッシュボード

```bash
python dashboard/app.py
```

→ http://localhost:8050 でアクセス

### ノートブック

```bash
jupyter notebook notebooks/01_semiconductor_cycle_analysis.ipynb
```

## WSL2での利用

1. WSL2ターミナルでダッシュボードを起動: `python dashboard/app.py`
2. Windows側ブラウザで http://localhost:8050 を開く

## ダッシュボードのタブ

1. **サイクル指標**: 出荷指数・在庫指数とSI比率の推移、局面判定の背景色表示
2. **在庫循環図**: 出荷YoY×在庫YoY散布図、4象限色分け、反時計回りサイクル可視化
3. **株価推移**: 全12銘柄の正規化株価チャート（対数スケール対応）
4. **サイクル×株価**: SI比率と個別銘柄の2軸オーバーレイ（対数スケール切替対応、マクロ指標オーバーレイ機能付き）
5. **局面別リターン**: 局面別グループ棒グラフ＋月次リターン統計テーブル
6. **グローバル指数**: SOX指数・日経平均・IIP出荷指数のYoY比較
7. **相関分析**: SI比率と各銘柄リターンの相関ヒートマップ（期間選択対応）
8. **リードラグ分析**: IIPサイクル指標と株価リターンのリードラグ相関（lag -12〜+12ヶ月、期間分割比較）
9. **半導体シグナル**: SOX vs IIP乖離指数、SOX×日経ローリング相関によるEPS低下リスク検知

### 半導体シグナルタブについて

「半導体シグナル.docx」で提案された5手法のうち、無料APIで取得可能な2手法を実装:

| # | 手法 | 実装 |
|---|------|------|
| ① | SOX YoY vs IIP乖離指数 | **実装済み** — 乖離が±1σ超で警戒シグナル |
| ④ | SOX×日経ローリング相関 | **実装済み** — 相関0.8超でテーマ相場過熱シグナル |
| ②③⑤ | SEMI BBレシオ / コンセンサスEPS / CapEx乖離 | 未実装（有料データ） |

## 起動時間について

初回起動時はyfinanceから株価データ（12銘柄+ベンチマーク2種+マクロ2種 = 計16銘柄）を取得するため、ネットワーク環境により **1〜3分** かかります。

### 起動を高速化するには

```bash
# 推奨: debug=Falseで起動（reloaderによる二重起動を防止）
python -c "from dashboard.app import app; app.run(host='0.0.0.0', port=8050, debug=False)"
```

### 現在のキャッシュ状況

| データ | キャッシュ | 鮮度 |
|--------|-----------|------|
| IIPデータ | `data/iip_electronic_parts.csv` | 30日 |
| FRED マクロ指標 | `data/macro_fred.csv` | 30日 |
| 株価データ（yfinance） | **なし** | 毎回取得 |

### 今後の改善予定

1. **株価CSVキャッシュ追加**: IIP/FREDと同パターンで1日キャッシュを導入し、2回目以降の起動を数秒に短縮
2. **yfinance一括ダウンロード**: 16銘柄個別取得→1回の一括取得に変更してHTTPオーバーヘッドを削減
3. **遅延ロード**: 起動時はキャッシュのみ読み込み、データ更新をバックグラウンドスレッドで実行

## e-Stat APIキー設定（最新データ取得）

APIキーなしでも動作します。同梱のCSVファイル（`data/iip_electronic_parts.csv`）に
IIP実データが含まれているため、そのままダッシュボードを利用できます。

APIキーを設定すると、e-Stat APIから最新のIIPデータを自動取得して更新します：

1. [e-Stat](https://www.e-stat.go.jp/) でユーザー登録
2. APIキー（appId）を取得
3. 環境変数を設定:
   ```bash
   export ESTAT_APP_ID=your_app_id_here
   ```
4. ダッシュボードまたはノートブックを再起動

## FRED APIキー設定（マクロ経済指標、オプショナル）

FRED APIキーなしでも基本機能（株価・IIP・yfinance経由の金利/為替）は動作します。
キーを設定すると、米GDP成長率・CPI・ISM PMIがサイクル×株価タブのオーバーレイで選択可能になります。

1. [FRED](https://fred.stlouisfed.org/) でアカウント作成
2. [API Keys](https://fredaccount.stlouisfed.org/apikeys) ページでキー取得
3. 環境変数を設定:
   ```bash
   export FRED_API_KEY=your_api_key_here
   ```
4. ダッシュボードを再起動

### マクロ指標一覧

| 指標 | ソース | APIキー |
|------|--------|---------|
| 米10年国債利回り | yfinance (^TNX) | 不要 |
| ドル円為替 | yfinance (JPY=X) | 不要 |
| 米GDP成長率(前期比年率%) | FRED (A191RL1Q225SBEA) | FRED_API_KEY |
| 米CPI(都市消費者) | FRED (CPIAUCSL) | FRED_API_KEY |
| 米鉱工業生産指数 | FRED (INDPRO) | FRED_API_KEY |
| 米耐久財受注(百万ドル) | FRED (DGORDER) | FRED_API_KEY |

## 技術スタック

Python 3.12+ / pandas / plotly / dash / yfinance / fredapi

## ライセンス

MIT
