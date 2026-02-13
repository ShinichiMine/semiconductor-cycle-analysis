"""経済産業省 鉱工業指数（IIP）データ取得・加工

電子部品・デバイス工業の出荷指数・在庫指数を取得し、
半導体サイクル（出荷/在庫比率）を算出する。

データソース: 経済産業省 鉱工業指数
https://www.meti.go.jp/statistics/tyo/iip/
"""

import pandas as pd
import requests
from io import BytesIO
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# 経産省 鉱工業指数 公開API (e-Stat)
# 電子部品・デバイス工業: 業種コード=340
# 出荷指数・在庫指数の時系列データ

# e-Stat API用の設定
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"


def load_iip_from_csv(filepath: str | Path) -> pd.DataFrame:
    """ローカルCSVからIIPデータを読み込む

    経産省サイトから手動DLしたCSVを想定。
    列: date, shipment_index, inventory_index
    """
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    return df


def create_sample_iip_data() -> pd.DataFrame:
    """分析用サンプルデータ生成（実データ取得までの代替）

    経産省IIP「電子部品・デバイス工業」の出荷・在庫指数の
    実績値に基づく近似データ。半導体サイクルの特徴を再現:
    - 2010-2011: 回復期
    - 2012-2013: 調整期
    - 2014-2016: 緩やかな回復
    - 2017-2018: スーパーサイクル
    - 2019: 調整
    - 2020-2021: コロナ需要
    - 2022-2023: 在庫調整
    - 2024-2025: AI需要回復
    """
    import numpy as np

    dates = pd.date_range("2010-01-01", "2025-12-31", freq="ME")
    n = len(dates)
    t = np.arange(n)

    # 基調トレンド（緩やかな上昇）
    trend = 95 + t * 0.08

    # 半導体サイクル（約3-4年周期）
    cycle = 12 * np.sin(2 * np.pi * t / 42)

    # 短期変動
    np.random.seed(42)
    noise = np.random.normal(0, 2, n)

    # イベント効果
    event = np.zeros(n)
    # 2017-2018 スーパーサイクル
    event[84:108] += 8
    # 2020 コロナショック → 回復
    event[120:126] -= 10
    event[126:144] += 12
    # 2022-2023 在庫調整
    event[144:162] -= 6
    # 2024-2025 AI需要
    event[168:] += 10

    shipment = trend + cycle + event + noise
    shipment = np.clip(shipment, 70, 140)

    # 在庫は出荷の逆位相（3-6ヶ月遅れ）
    inventory_base = 100 + t * 0.05
    inventory_cycle = -8 * np.sin(2 * np.pi * (t - 5) / 42)
    inventory_event = np.zeros(n)
    inventory_event[126:150] += 15  # コロナ後の在庫積み上がり
    inventory_event[150:168] += 8
    inventory_event[168:] -= 5  # AI需要で在庫消化

    inventory = inventory_base + inventory_cycle + inventory_event + noise * 0.8
    inventory = np.clip(inventory, 75, 135)

    df = pd.DataFrame(
        {"shipment_index": shipment, "inventory_index": inventory},
        index=dates,
    )
    df.index.name = "date"
    return df


def calc_cycle_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """半導体サイクル指標を算出

    Returns:
        DataFrame with columns:
        - shipment_index: 出荷指数
        - inventory_index: 在庫指数
        - si_ratio: 出荷/在庫比率（>1で好況局面）
        - si_ratio_ma3: 出荷/在庫比率の3ヶ月移動平均
        - shipment_yoy: 出荷指数の前年同月比(%)
        - inventory_yoy: 在庫指数の前年同月比(%)
        - cycle_phase: サイクル局面判定
    """
    result = df.copy()

    # 出荷/在庫比率
    result["si_ratio"] = result["shipment_index"] / result["inventory_index"]
    result["si_ratio_ma3"] = result["si_ratio"].rolling(3).mean()

    # 前年同月比
    result["shipment_yoy"] = result["shipment_index"].pct_change(12) * 100
    result["inventory_yoy"] = result["inventory_index"].pct_change(12) * 100

    # サイクル局面判定
    result["cycle_phase"] = _classify_phase(result)

    return result


def _classify_phase(df: pd.DataFrame) -> pd.Series:
    """出荷/在庫比率と変化方向からサイクル局面を4分類

    1. 回復期 (Recovery):  SI比率上昇 & SI比率<1
    2. 好況期 (Expansion): SI比率上昇 & SI比率>=1
    3. 後退期 (Slowdown):  SI比率下降 & SI比率>=1
    4. 不況期 (Recession):  SI比率下降 & SI比率<1
    """
    si = df["si_ratio_ma3"]
    si_diff = si.diff()

    conditions = []
    for i in range(len(df)):
        if pd.isna(si.iloc[i]) or pd.isna(si_diff.iloc[i]):
            conditions.append("不明")
        elif si_diff.iloc[i] >= 0 and si.iloc[i] < 1:
            conditions.append("回復期")
        elif si_diff.iloc[i] >= 0 and si.iloc[i] >= 1:
            conditions.append("好況期")
        elif si_diff.iloc[i] < 0 and si.iloc[i] >= 1:
            conditions.append("後退期")
        else:
            conditions.append("不況期")

    return pd.Series(conditions, index=df.index, name="cycle_phase")
