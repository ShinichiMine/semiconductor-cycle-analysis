"""日本半導体関連銘柄の株価データ取得"""

import pandas as pd
import yfinance as yf
from datetime import datetime

# 主要日本半導体銘柄
SEMICONDUCTOR_STOCKS = {
    "8035.T": "東京エレクトロン",
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "7735.T": "SCREENホールディングス",
    "6723.T": "ルネサスエレクトロニクス",
    "6146.T": "ディスコ",
    "6963.T": "ローム",
    "4063.T": "信越化学工業",
}

# ベンチマーク
BENCHMARKS = {
    "^N225": "日経平均",
    "SOX": "フィラデルフィア半導体指数",
}


def fetch_stock_prices(
    tickers: dict[str, str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """複数銘柄の月次終値を取得し、1つのDataFrameにまとめる"""
    if tickers is None:
        tickers = SEMICONDUCTOR_STOCKS
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    frames = {}
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  [WARN] {name} ({ticker}): データなし")
                continue
            # 月次リサンプル（月末終値）
            # yfinance>=1.0 returns MultiIndex columns (Price, Ticker)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            monthly = close.resample("ME").last()
            frames[name] = monthly
        except Exception as e:
            print(f"  [ERROR] {name} ({ticker}): {e}")

    result = pd.DataFrame(frames)
    result.index.name = "date"
    return result


def fetch_sox_index(start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """SOX指数（フィラデルフィア半導体指数）を取得"""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    df = yf.download("^SOX", start=start, end=end, progress=False)
    if df.empty:
        return pd.Series(dtype=float, name="SOX")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    monthly = close.resample("ME").last()
    monthly.name = "SOX"
    return monthly


def normalize_prices(df: pd.DataFrame, base_date: str | None = None) -> pd.DataFrame:
    """基準日を100として正規化（比較用）"""
    if base_date:
        base = df.loc[base_date]
    else:
        base = df.iloc[0]
    return (df / base) * 100
