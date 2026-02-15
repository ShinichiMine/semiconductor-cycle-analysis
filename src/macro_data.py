"""FRED API（セントルイス連銀）マクロ経済指標取得

GDP成長率・CPI・鉱工業生産・耐久財受注を取得し、半導体サイクルとの比較に使用する。
FRED APIキーはオプショナル — 未設定時は None を返す。

データソース: Federal Reserve Bank of St. Louis (FRED)
https://fred.stlouisfed.org/
"""

import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FRED_CSV_PATH = DATA_DIR / "macro_fred.csv"

# FRED系列ID → 日本語ラベル
FRED_SERIES = {
    "A191RL1Q225SBEA": "米GDP成長率(前期比年率%)",
    "CPIAUCSL": "米CPI(都市消費者)",
    "INDPRO": "米鉱工業生産指数",
    "DGORDER": "米耐久財受注(百万ドル)",
}


def fetch_fred_data(
    api_key: str | None = None,
    start: str = "2010-01-01",
) -> pd.DataFrame | None:
    """FRED APIからマクロ経済指標を取得する。

    Args:
        api_key: FRED APIキー。None の場合は環境変数 FRED_API_KEY を使用。
        start: 取得開始日。

    Returns:
        DataFrame (columns=指標名, index=date, 月次) or None (キー未設定時)
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        return None

    # キャッシュ確認（30日以内なら再利用）
    if FRED_CSV_PATH.exists():
        cached = pd.read_csv(FRED_CSV_PATH, parse_dates=["date"], index_col="date")
        last_date = cached.index.max()
        days_old = (pd.Timestamp.now() - last_date).days
        if days_old <= 30:
            return cached

    try:
        from fredapi import Fred
    except ImportError:
        print("  [WARN] fredapi未インストール。pip install fredapi で追加してください。")
        return _load_cache_fallback()

    try:
        fred = Fred(api_key=api_key)
        frames = {}

        for series_id, label in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                if s is None or s.empty:
                    print(f"  [WARN] FRED {label} ({series_id}): データなし")
                    continue
                # 月次リサンプル（月末基準）
                monthly = s.resample("ME").last()
                # GDPは四半期 → ffillで月次補間
                if series_id == "A191RL1Q225SBEA":
                    monthly = monthly.ffill()
                frames[label] = monthly
            except Exception as e:
                print(f"  [ERROR] FRED {label} ({series_id}): {e}")

        if not frames:
            return _load_cache_fallback()

        result = pd.DataFrame(frames)
        result.index.name = "date"
        result = result.dropna(how="all")

        # CSVキャッシュ保存
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(FRED_CSV_PATH)
        return result

    except Exception as e:
        print(f"  [ERROR] FRED API取得失敗: {e}")
        return _load_cache_fallback()


def _load_cache_fallback() -> pd.DataFrame | None:
    """キャッシュCSVが存在すればフォールバックとして読み込む"""
    if FRED_CSV_PATH.exists():
        print("  既存キャッシュを使用します。")
        return pd.read_csv(FRED_CSV_PATH, parse_dates=["date"], index_col="date")
    return None
