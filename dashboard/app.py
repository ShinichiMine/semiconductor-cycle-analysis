"""半導体サイクル分析 WEBダッシュボード

Usage:
    python dashboard/app.py
    → http://localhost:8050
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.stock_data import (
    SEMICONDUCTOR_STOCKS,
    fetch_stock_prices,
    fetch_benchmark_prices,
    fetch_sox_index,
    fetch_macro_yfinance,
    normalize_prices,
)
from src.iip_data import load_or_fetch_iip, calc_cycle_indicator, load_iip_from_upload
from src.macro_data import fetch_fred_data

# ============================================================
# データ準備
# ============================================================

print("データを準備中...")
iip = calc_cycle_indicator(load_or_fetch_iip())

print("株価データを取得中...")
try:
    prices = fetch_stock_prices(start="2010-01-01")
    prices_available = True
except Exception as e:
    print(f"株価取得失敗（オフラインモード）: {e}")
    prices = pd.DataFrame()
    prices_available = False

print("ベンチマークデータを取得中...")
try:
    benchmarks = fetch_benchmark_prices(start="2010-01-01")
    benchmarks_available = not benchmarks.empty
except Exception as e:
    print(f"ベンチマーク取得失敗（オフラインモード）: {e}")
    benchmarks = pd.DataFrame()
    benchmarks_available = False

print("マクロ指標（yfinance）を取得中...")
try:
    macro_yf = fetch_macro_yfinance(start="2010-01-01")
    macro_yf_available = not macro_yf.empty
except Exception as e:
    print(f"マクロ指標（yfinance）取得失敗: {e}")
    macro_yf = pd.DataFrame()
    macro_yf_available = False

print("マクロ指標（FRED）を取得中...")
try:
    macro_fred = fetch_fred_data()
    macro_fred_available = macro_fred is not None and not macro_fred.empty
except Exception as e:
    print(f"マクロ指標（FRED）取得失敗: {e}")
    macro_fred = None
    macro_fred_available = False

# 全マクロ指標を1つのDataFrameにまとめる
macro_all = pd.DataFrame()
if macro_yf_available:
    macro_all = macro_yf.copy()
if macro_fred_available:
    macro_all = pd.concat([macro_all, macro_fred], axis=1) if not macro_all.empty else macro_fred.copy()

PHASE_COLORS = {
    "回復期": {"bg": "rgba(76, 175, 80, 0.15)", "text": "#4CAF50"},
    "好況期": {"bg": "rgba(33, 150, 243, 0.15)", "text": "#2196F3"},
    "後退期": {"bg": "rgba(255, 152, 0, 0.15)", "text": "#FF9800"},
    "不況期": {"bg": "rgba(244, 67, 54, 0.15)", "text": "#F44336"},
}

# ============================================================
# チャート生成関数
# ============================================================


def create_cycle_chart():
    """出荷/在庫指数 + SI比率チャート"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "電子部品・デバイス工業 出荷指数・在庫指数",
            "出荷/在庫比率（SI比率）とサイクル局面",
        ),
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Scatter(
            x=iip.index, y=iip["shipment_index"],
            name="出荷指数", line=dict(color="#2196F3", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=iip.index, y=iip["inventory_index"],
            name="在庫指数", line=dict(color="#FF9800", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=iip.index, y=iip["si_ratio_ma3"],
            name="SI比率(3MA)", line=dict(color="#673AB7", width=2.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    phases = iip["cycle_phase"]
    for phase, colors in PHASE_COLORS.items():
        mask = phases == phase
        if not mask.any():
            continue
        starts = mask & ~mask.shift(1, fill_value=False)
        ends = mask & ~mask.shift(-1, fill_value=False)
        for s, e in zip(iip.index[starts], iip.index[ends]):
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor=colors["bg"], layer="below", line_width=0,
                row=2, col=1,
            )

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="指数", row=1, col=1)
    fig.update_yaxes(title_text="SI比率", row=2, col=1)
    return fig


def create_stock_chart(selected_stocks=None, use_log=True):
    """正規化株価チャート"""
    if not prices_available or prices.empty:
        fig = go.Figure()
        fig.add_annotation(text="株価データ未取得", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    df = prices
    if selected_stocks:
        cols = [c for c in selected_stocks if c in df.columns]
        if cols:
            df = df[cols]

    norm = normalize_prices(df)
    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(
            go.Scatter(
                x=norm.index, y=norm[col],
                name=col,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        height=500,
        template="plotly_white",
        yaxis_title="正規化株価 (基準=100)",
        yaxis_type="log" if use_log else "linear",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=20, b=40),
    )
    return fig


def create_overlay_chart(stock_name, use_log=False, macro_indicators=None):
    """SI比率と個別銘柄の2軸チャート（マクロ指標オーバーレイ対応）"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if prices_available and stock_name in prices.columns:
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices[stock_name],
                name=f"{stock_name} 株価",
                line=dict(color="#2196F3", width=2),
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=iip.index, y=iip["si_ratio_ma3"],
            name="SI比率(3MA)",
            line=dict(color="#FF5722", width=1.5, dash="dot"),
            opacity=0.8,
        ),
        secondary_y=True,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", secondary_y=True)

    # マクロ指標オーバーレイ
    if macro_indicators and not macro_all.empty:
        macro_colors = ["#9C27B0", "#009688", "#795548", "#607D8B", "#E91E63"]
        for i, indicator in enumerate(macro_indicators):
            if indicator in macro_all.columns:
                series = macro_all[indicator].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=series.index, y=series,
                        name=indicator,
                        line=dict(
                            color=macro_colors[i % len(macro_colors)],
                            width=1.5, dash="dashdot",
                        ),
                        opacity=0.7,
                    ),
                    secondary_y=True,
                )

    fig.update_layout(
        height=500 if macro_indicators else 400,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, t=20, b=40),
    )
    fig.update_yaxes(
        title_text="株価 (円)",
        type="log" if use_log else "linear",
        secondary_y=False,
    )
    fig.update_yaxes(title_text="SI比率 / マクロ指標", secondary_y=True)
    return fig


def create_inventory_cycle_chart(interval=1):
    """在庫循環図（散布図 + 4象限）

    Args:
        interval: サンプリング間隔（月数）。1=月次, 3=四半期, 6=半期, 12=年次
    """
    df_full = iip.dropna(subset=["shipment_yoy", "inventory_yoy"]).copy()
    if df_full.empty:
        fig = go.Figure()
        fig.add_annotation(text="データ不足", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    # 間隔でサンプリング（最新月は必ず含める）
    if interval > 1:
        sampled_idx = list(range(0, len(df_full), interval))
        if (len(df_full) - 1) not in sampled_idx:
            sampled_idx.append(len(df_full) - 1)
        df = df_full.iloc[sampled_idx]
    else:
        df = df_full

    n = len(df)
    interval_label = {1: "月次", 3: "3ヶ月毎", 6: "6ヶ月毎", 12: "12ヶ月毎"}.get(interval, f"{interval}ヶ月毎")
    show_arrows = n <= 30

    fig = go.Figure()

    # 線で時系列接続
    fig.add_trace(
        go.Scatter(
            x=df["shipment_yoy"], y=df["inventory_yoy"],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.5)", width=1.5 if show_arrows else 1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 矢印（プロット数が少ない場合のみ）
    if show_arrows:
        for i in range(len(df) - 1):
            x0, y0 = df["shipment_yoy"].iloc[i], df["inventory_yoy"].iloc[i]
            x1, y1 = df["shipment_yoy"].iloc[i + 1], df["inventory_yoy"].iloc[i + 1]
            # 色: 時系列に沿って薄→濃
            alpha = 0.3 + 0.7 * i / max(n - 2, 1)
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor=f"rgba(33, 150, 243, {alpha})",
            )

    # 散布点
    marker_size = 10 if show_arrows else 6
    fig.add_trace(
        go.Scatter(
            x=df["shipment_yoy"], y=df["inventory_yoy"],
            mode="markers+text" if show_arrows else "markers",
            marker=dict(size=marker_size, color=list(range(n)), colorscale="Blues",
                        showscale=True, colorbar=dict(title="時系列順"),
                        line=dict(width=1, color="white") if show_arrows else dict(width=0)),
            text=[d.strftime("%Y-%m") for d in df.index] if show_arrows else None,
            textposition="top center" if show_arrows else None,
            textfont=dict(size=9) if show_arrows else None,
            hovertemplate="出荷YoY: %{x:.1f}%<br>在庫YoY: %{y:.1f}%<br>%{customdata}<extra></extra>",
            customdata=[d.strftime("%Y-%m") for d in df.index],
            name=f"{interval_label}データ",
        )
    )

    # 最新点を星マーカーで強調
    latest_row = df.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[latest_row["shipment_yoy"]], y=[latest_row["inventory_yoy"]],
            mode="markers+text",
            marker=dict(size=16, color="#F44336", symbol="star"),
            text=[df.index[-1].strftime("%Y-%m")],
            textposition="top right",
            name="最新月",
        )
    )

    # 4象限の背景色 + ラベル（全データで範囲を計算 → 間隔変更で軸がぶれない）
    x_range = max(abs(df_full["shipment_yoy"].min()), abs(df_full["shipment_yoy"].max())) * 1.2
    y_range = max(abs(df_full["inventory_yoy"].min()), abs(df_full["inventory_yoy"].max())) * 1.2

    quadrants = [
        {"x0": 0, "x1": x_range, "y0": -y_range, "y1": 0,
         "color": PHASE_COLORS["回復期"]["bg"], "label": "回復期", "lx": x_range * 0.5, "ly": -y_range * 0.5},
        {"x0": 0, "x1": x_range, "y0": 0, "y1": y_range,
         "color": PHASE_COLORS["好況期"]["bg"], "label": "好況期", "lx": x_range * 0.5, "ly": y_range * 0.5},
        {"x0": -x_range, "x1": 0, "y0": 0, "y1": y_range,
         "color": PHASE_COLORS["後退期"]["bg"], "label": "後退期", "lx": -x_range * 0.5, "ly": y_range * 0.5},
        {"x0": -x_range, "x1": 0, "y0": -y_range, "y1": 0,
         "color": PHASE_COLORS["不況期"]["bg"], "label": "不況期", "lx": -x_range * 0.5, "ly": -y_range * 0.5},
    ]

    for q in quadrants:
        fig.add_shape(
            type="rect", x0=q["x0"], x1=q["x1"], y0=q["y0"], y1=q["y1"],
            fillcolor=q["color"], layer="below", line_width=0,
        )
        fig.add_annotation(
            x=q["lx"], y=q["ly"], text=q["label"],
            showarrow=False, font=dict(size=14, color="rgba(0,0,0,0.3)"),
        )

    # 原点の十字線
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        height=550,
        template="plotly_white",
        xaxis_title="出荷指数 前年比 (%)",
        yaxis_title="在庫指数 前年比 (%)",
        xaxis=dict(range=[-x_range, x_range]),
        yaxis=dict(range=[-y_range, y_range]),
        hovermode="closest",
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


def create_phase_return_bar_chart():
    """局面別リターンのグループ棒グラフ"""
    df = _calc_phase_returns()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="株価データ未取得", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    phase_order = ["回復期", "好況期", "後退期", "不況期"]
    fig = go.Figure()
    for phase in phase_order:
        subset = df[df["局面"] == phase]
        fig.add_trace(
            go.Bar(
                x=subset["銘柄"],
                y=subset["平均リターン(%)"],
                name=phase,
                marker_color=PHASE_COLORS[phase]["text"],
                opacity=0.85,
            )
        )

    fig.update_layout(
        barmode="group",
        height=400,
        template="plotly_white",
        yaxis_title="平均月次リターン (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def create_global_index_chart():
    """グローバル指数比較チャート（SOX/日経/IIP YoY）"""
    fig = go.Figure()

    # IIP出荷指数 YoY
    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    shipment_yoy = iip_me["shipment_yoy"].dropna()
    fig.add_trace(
        go.Scatter(
            x=shipment_yoy.index, y=shipment_yoy,
            name="IIP出荷指数 YoY",
            line=dict(color="#673AB7", width=2),
        )
    )

    # ベンチマーク YoY
    if benchmarks_available and not benchmarks.empty:
        bench_yoy = benchmarks.pct_change(12) * 100  # 12ヶ月前比
        bench_yoy = bench_yoy.dropna(how="all")
        colors = {"日経平均": "#2196F3", "フィラデルフィア半導体指数": "#FF5722"}
        labels = {"日経平均": "日経平均 YoY", "フィラデルフィア半導体指数": "SOX指数 YoY"}
        for col in bench_yoy.columns:
            series = bench_yoy[col].dropna()
            if series.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series,
                    name=labels.get(col, f"{col} YoY"),
                    line=dict(color=colors.get(col, "#888"), width=2),
                )
            )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        height=500,
        template="plotly_white",
        yaxis_title="前年比 (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ============================================================
# Dashアプリ
# ============================================================

app = dash.Dash(
    __name__,
    title="半導体サイクル分析ダッシュボード",
    suppress_callback_exceptions=True,
)


def _status_card(title, value, color):
    return html.Div(
        style={
            "flex": "1", "minWidth": "150px", "padding": "15px",
            "borderRadius": "8px", "backgroundColor": "#fff",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "textAlign": "center",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#888",
                                    "marginBottom": "5px"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "bold",
                                    "color": color}),
        ],
    )


# 現在のサイクル情報
latest = iip.iloc[-1]
current_phase = latest["cycle_phase"]
phase_color = PHASE_COLORS.get(current_phase, {"text": "#666"})["text"]

app.layout = html.Div(
    style={"fontFamily": "'Noto Sans JP', sans-serif", "maxWidth": "1200px",
           "margin": "0 auto", "padding": "20px"},
    children=[
        # ヘッダー
        html.H1(
            "半導体サイクル分析ダッシュボード",
            style={"textAlign": "center", "color": "#333", "marginBottom": "5px"},
        ),
        html.P(
            "電子部品・デバイス工業 IIP × 日本半導体銘柄",
            style={"textAlign": "center", "color": "#888", "marginBottom": "20px"},
        ),

        # サイクル状態カード
        html.Div(
            style={"display": "flex", "gap": "15px", "marginBottom": "25px",
                   "flexWrap": "wrap"},
            children=[
                _status_card("現在の局面", current_phase, phase_color),
                _status_card("SI比率", f"{latest['si_ratio_ma3']:.3f}",
                             "#673AB7"),
                _status_card("出荷指数", f"{latest['shipment_index']:.1f}",
                             "#2196F3"),
                _status_card("在庫指数", f"{latest['inventory_index']:.1f}",
                             "#FF9800"),
                _status_card("出荷YoY",
                             f"{latest['shipment_yoy']:+.1f}%",
                             "#4CAF50" if latest["shipment_yoy"] > 0 else "#F44336"),
            ],
        ),

        # IIPデータアップロード
        html.Details(
            style={"marginBottom": "15px", "padding": "10px", "backgroundColor": "#f9f9f9",
                   "borderRadius": "8px", "border": "1px solid #eee"},
            children=[
                html.Summary("IIPデータ手動アップロード",
                            style={"cursor": "pointer", "fontSize": "13px", "color": "#666"}),
                html.Div(
                    style={"marginTop": "10px"},
                    children=[
                        html.P("経産省サイトからDLしたExcel（b2020_gsm1j.xlsx等）をそのままアップロードできます。手動整形は不要です。",
                               style={"fontSize": "12px", "color": "#888", "marginBottom": "8px"}),
                        html.P("整形済みCSV/Excel（date, shipment_index, inventory_index列）にも対応しています。",
                               style={"fontSize": "12px", "color": "#888", "marginBottom": "8px"}),
                        dcc.Upload(
                            id="iip-upload",
                            children=html.Div([
                                "ファイルをドラッグ&ドロップ、または ",
                                html.A("クリックして選択", style={"color": "#2196F3"}),
                            ]),
                            style={
                                "width": "100%", "height": "50px", "lineHeight": "50px",
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "5px", "textAlign": "center",
                                "borderColor": "#ccc", "backgroundColor": "#fff",
                            },
                            accept=".csv,.xlsx,.xls",
                        ),
                        html.Div(id="iip-upload-status",
                                style={"marginTop": "8px", "fontSize": "12px"}),
                    ],
                ),
            ],
        ),

        # タブ
        dcc.Tabs(
            id="tabs",
            value="tab-cycle",
            children=[
                dcc.Tab(label="サイクル指標", value="tab-cycle"),
                dcc.Tab(label="在庫循環図", value="tab-inventory-cycle"),
                dcc.Tab(label="株価推移", value="tab-stocks"),
                dcc.Tab(label="サイクル×株価", value="tab-overlay"),
                dcc.Tab(label="局面別リターン", value="tab-phase-return"),
                dcc.Tab(label="グローバル指数", value="tab-global-index"),
                dcc.Tab(label="相関分析", value="tab-correlation"),
                dcc.Tab(label="リードラグ分析", value="tab-leadlag"),
                dcc.Tab(label="半導体シグナル", value="tab-signal"),
            ],
            style={"marginBottom": "15px"},
        ),
        html.Div(id="tab-content"),
    ],
)


def _calc_phase_returns():
    """局面別リターン統計を計算して返す（DataFrame）"""
    if not prices_available or prices.empty:
        return pd.DataFrame()
    monthly_returns = prices.pct_change() * 100  # %
    phase_series = iip["cycle_phase"].reindex(monthly_returns.index, method="ffill")
    combined = monthly_returns.copy()
    combined["cycle_phase"] = phase_series
    combined = combined.dropna(subset=["cycle_phase"])

    rows = []
    for phase in ["回復期", "好況期", "後退期", "不況期"]:
        subset = combined[combined["cycle_phase"] == phase]
        for stock in prices.columns:
            vals = subset[stock].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "局面": phase,
                "銘柄": stock,
                "平均リターン(%)": round(vals.mean(), 2),
                "中央値リターン(%)": round(vals.median(), 2),
                "勝率(%)": round((vals > 0).mean() * 100, 1),
                "サンプル数": len(vals),
            })
    return pd.DataFrame(rows)


def _build_phase_return_tab():
    """局面別リターン分析タブのレイアウト"""
    df = _calc_phase_returns()
    if df.empty:
        return html.Div("株価データ未取得のため表示できません。",
                        style={"padding": "40px", "textAlign": "center", "color": "#888"})

    # 局面ごとに背景色を付けるための style_data_conditional
    style_cond = []
    for phase, colors in PHASE_COLORS.items():
        style_cond.append({
            "if": {"filter_query": f'{{局面}} = "{phase}"'},
            "backgroundColor": colors["bg"],
            "color": "#333",
        })

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f5f5f5",
            "fontWeight": "bold",
            "textAlign": "center",
            "borderBottom": "2px solid #ddd",
        },
        style_cell={
            "textAlign": "center",
            "padding": "8px 12px",
            "fontFamily": "'Noto Sans JP', sans-serif",
            "fontSize": "13px",
        },
        style_data_conditional=style_cond,
        sort_action="native",
        filter_action="native",
        page_size=40,
    )

    legend = html.Div(
        style={"display": "flex", "gap": "15px", "justifyContent": "center",
               "marginBottom": "12px", "flexWrap": "wrap"},
        children=[
            html.Span(f"■ {phase}", style={"color": c["text"], "fontSize": "13px"})
            for phase, c in PHASE_COLORS.items()
        ],
    )

    return html.Div([
        html.H3("局面別リターン分析", style={"marginBottom": "8px"}),
        html.P("各サイクル局面における銘柄ごとの月次リターン統計（IIPサンプルデータベース）",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
        dcc.Graph(figure=create_phase_return_bar_chart()),
        html.Hr(style={"margin": "20px 0"}),
        legend,
        table,
        _build_phase_return_guide(),
    ])


def _build_correlation_tab():
    """相関分析タブのレイアウト"""
    if not prices_available or prices.empty:
        return html.Div("株価データ未取得のため表示できません。",
                        style={"padding": "40px", "textAlign": "center", "color": "#888"})

    # 年範囲スライダーの設定
    # iipインデックスは月初(2013-01-01等)、pricesは月末(2013-01-31等) → MonthEndに正規化
    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    common_idx = iip_me.index.intersection(prices.pct_change().dropna(how="all").index)
    if len(common_idx) == 0:
        return html.Div("共通データなし", style={"padding": "40px", "textAlign": "center"})

    min_year = int(common_idx.min().year)
    max_year = int(common_idx.max().year)

    return html.Div([
        html.H3("相関分析: SI比率 × 月次リターン", style={"marginBottom": "8px"}),
        html.P("SI比率(3ヶ月移動平均)と各銘柄月次リターンのピアソン相関係数",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "20px",
                   "marginBottom": "16px"},
            children=[
                html.Label("対象期間（年）:", style={"whiteSpace": "nowrap"}),
                dcc.RangeSlider(
                    id="corr-year-range",
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[min_year, max_year],
                    marks={y: str(y) for y in range(min_year, max_year + 1, 2)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        ),
        dcc.Graph(id="corr-heatmap"),
        _build_correlation_guide(),
    ])


@callback(
    Output("corr-heatmap", "figure"),
    Input("corr-year-range", "value"),
    prevent_initial_call=False,
)
def update_corr_heatmap(year_range):
    if not prices_available or prices.empty:
        return go.Figure()

    start_year, end_year = year_range if year_range else [None, None]
    monthly_returns = prices.pct_change() * 100
    # iipインデックスを月末に正規化してから reindex（月初/月末の不一致を解消）
    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    si = iip_me["si_ratio_ma3"].reindex(monthly_returns.index, method="ffill")

    combined = monthly_returns.copy()
    combined["SI比率(3MA)"] = si
    combined = combined.dropna()

    if start_year:
        combined = combined[combined.index.year >= start_year]
    if end_year:
        combined = combined[combined.index.year <= end_year]

    if combined.empty or len(combined) < 3:
        fig = go.Figure()
        fig.add_annotation(text="データ不足", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        return fig

    # SI比率と各銘柄リターンの相関のみ（1行×N列）
    stock_cols = [c for c in prices.columns if c in combined.columns]
    corr_vals = [[combined["SI比率(3MA)"].corr(combined[s]) for s in stock_cols]]
    corr_df = pd.DataFrame(corr_vals, index=["SI比率(3MA)"], columns=stock_cols)

    fig = px.imshow(
        corr_df,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title=f"SI比率 × 月次リターン 相関係数 ({start_year}〜{end_year}年)",
    )
    fig.update_layout(
        height=250,
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
        coloraxis_colorbar=dict(title="相関係数"),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


_DETAILS_STYLE = {
    "marginTop": "20px", "padding": "12px", "backgroundColor": "#fafafa",
    "borderRadius": "8px", "border": "1px solid #eee",
}
_SUMMARY_STYLE = {"fontWeight": "bold", "cursor": "pointer", "fontSize": "14px"}
_SECTION_STYLE = {"marginTop": "10px", "fontSize": "13px", "lineHeight": "1.7", "color": "#444"}


def _make_guide(title, content_lines):
    """折りたたみセクション1つを生成"""
    return html.Details(
        style=_DETAILS_STYLE,
        children=[
            html.Summary(title, style=_SUMMARY_STYLE),
            html.Div(style=_SECTION_STYLE, children=[
                html.P(line) if isinstance(line, str) else line
                for line in content_lines
            ]),
        ],
    )


def _build_cycle_guide():
    """サイクル指標タブの解説"""
    # 動的: 最新値
    si_val = latest["si_ratio_ma3"]
    si_trend = "上昇" if iip["si_ratio_ma3"].diff().iloc[-1] >= 0 else "下降"
    phase = current_phase
    guide = _make_guide("読み方ガイド", [
        "SI比率（出荷/在庫比率）> 1.0 は好況圏、< 1.0 は不況圏を示します。",
        "3ヶ月移動平均（3MA）でノイズを平滑化。背景色は4局面（回復・好況・後退・不況）を表します。",
        "上段: 出荷指数（青）と在庫指数（橙）の原系列。下段: SI比率と局面判定。",
    ])
    dynamic = _make_guide("現在の状況", [
        f"最新SI比率(3MA): {si_val:.3f}（{si_trend}トレンド）",
        f"現在の局面: {phase}",
        f"最新データ: {iip.index[-1].strftime('%Y年%m月')}",
    ])
    insight = _make_guide("分析知見", [
        "2019年以降、SI比率は「遅行指標」化しています。株価が先行し、8-12ヶ月後にSI比率が追随するパターンが確認されています。",
        "2013-2018年は同時〜先行指標として有効でしたが、AI/先端半導体需要の構造変化により指標特性が変化しました。",
        "SOX指数が事実上の先行・同時指標（日本株とlag=0でr=0.6-0.75）として機能しています。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_inventory_cycle_guide():
    """在庫循環図タブの解説"""
    df = iip.dropna(subset=["shipment_yoy", "inventory_yoy"])
    if df.empty:
        return html.Div()
    latest_row = df.iloc[-1]
    sx, iy = latest_row["shipment_yoy"], latest_row["inventory_yoy"]
    if sx >= 0 and iy < 0:
        quadrant = "回復期（右下）"
    elif sx >= 0 and iy >= 0:
        quadrant = "好況期（右上）"
    elif sx < 0 and iy >= 0:
        quadrant = "後退期（左上）"
    else:
        quadrant = "不況期（左下）"

    guide = _make_guide("読み方ガイド", [
        "横軸: 出荷指数YoY、縦軸: 在庫指数YoY。理論上は反時計回りに循環します。",
        "右下=回復期（出荷↑在庫↓）→ 右上=好況期（出荷↑在庫↑）→ 左上=後退期（出荷↓在庫↑）→ 左下=不況期（出荷↓在庫↓）。",
        "色の濃淡は時系列順（薄い=古い、濃い=新しい）。赤い星は最新月です。",
    ])
    dynamic = _make_guide("現在の状況", [
        f"最新月: {df.index[-1].strftime('%Y年%m月')} — {quadrant}",
        f"出荷YoY: {sx:+.1f}%、在庫YoY: {iy:+.1f}%",
    ])
    insight = _make_guide("分析知見", [
        "実際のサイクルは教科書通りの滑らかな循環ではなく、同一象限に長期滞在→急転換が多い傾向があります。",
        "回復→回復（同象限内遷移）が28回で最多。2024-2025年は回復期と不況期を頻繁に往復しています。",
        "第一生命経済研究所: 2025年1月にシリコンサイクルのピーク到達の可能性を指摘。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_stock_guide():
    """株価推移タブの解説"""
    parts = [
        "正規化株価: 各銘柄の始点を100として比較。対数スケールで長期トレンドを確認できます。",
    ]
    dynamic_parts = []
    if prices_available and not prices.empty:
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        top = total_return.idxmax()
        bottom = total_return.idxmin()
        dynamic_parts = [
            f"期間内騰落率トップ: {top}（{total_return[top]:+.0f}%）",
            f"期間内騰落率ボトム: {bottom}（{total_return[bottom]:+.0f}%）",
        ]
    guide = _make_guide("読み方ガイド", parts)
    dynamic = _make_guide("現在の状況", dynamic_parts if dynamic_parts else ["株価データ未取得"])
    insight = _make_guide("分析知見", [
        "2019年以降はAI/先端半導体関連（東京エレクトロン、アドバンテスト等）がアウトパフォームしています。",
        "ソシオネクスト（ファブレス）はIIPサイクルとの連動が弱く、独自の業績サイクルで動きます。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_overlay_guide():
    """サイクル×株価タブの解説"""
    guide = _make_guide("読み方ガイド", [
        "左軸: 個別銘柄の株価、右軸: SI比率(3MA)。両者の連動/乖離を視覚的に確認できます。",
        "マクロ指標（金利・為替・FRED系列）をオーバーレイして、複合的な分析が可能です。",
    ])
    dynamic = _make_guide("現在の状況", [
        f"最新SI比率(3MA): {latest['si_ratio_ma3']:.3f}",
        f"現在の局面: {current_phase}",
    ])
    insight = _make_guide("分析知見", [
        "核心の知見: 2013-2018年はSI比率がlag=0で株価と正相関（先行〜同時指標として有効）。",
        "2019-2025年はlag=0で負の相関に転換（遅行化）。仮説: (1)エッジ消失 (2)予想EPSが先行 (3)SOXが代替指標化。",
        "SOX指数は日本半導体株とlag=0でr=0.6-0.75の強い同時相関を示しています。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_phase_return_guide():
    """局面別リターンタブの解説"""
    guide = _make_guide("読み方ガイド", [
        "各サイクル局面（回復・好況・後退・不況）における銘柄ごとの平均月次リターンを表示。",
        "棒グラフ: 局面ごとのリターン比較。テーブル: 詳細統計（平均・中央値・勝率・サンプル数）。",
    ])
    dynamic_parts = []
    df = _calc_phase_returns()
    if not df.empty:
        phase_df = df[df["局面"] == current_phase]
        if not phase_df.empty:
            top_stock = phase_df.loc[phase_df["平均リターン(%)"].idxmax()]
            worst_stock = phase_df.loc[phase_df["平均リターン(%)"].idxmin()]
            dynamic_parts = [
                f"現在の局面（{current_phase}）での月次リターン:",
                f"  トップ: {top_stock['銘柄']}（{top_stock['平均リターン(%)']:+.2f}%/月）",
                f"  ワースト: {worst_stock['銘柄']}（{worst_stock['平均リターン(%)']:+.2f}%/月）",
            ]
    dynamic = _make_guide("現在の状況", dynamic_parts if dynamic_parts else ["データ不足"])
    insight = _make_guide("分析知見", [
        "2013-2018年: 回復期にルネサス(+3.6%/月)、好況期にSUMCO(+10.0%/月)。教科書通りの局面投資が有効でした。",
        "2019-2025年: 不況期にソシオネクスト(+15.0%/月)が最高リターン — 局面投資の前提が崩壊しています。",
        "要因: AI需要による構造変化で、従来のサイクル局面と株価パフォーマンスの対応が変化。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_global_index_guide():
    """グローバル指数タブの解説"""
    guide = _make_guide("読み方ガイド", [
        "SOX指数（フィラデルフィア半導体指数）、日経平均、IIP出荷指数の前年比（YoY）を比較。",
        "3指標の連動性とリード/ラグの関係を視覚的に確認できます。",
    ])
    dynamic_parts = []
    if not iip.empty:
        ship_yoy = iip["shipment_yoy"].dropna()
        if not ship_yoy.empty:
            dynamic_parts.append(
                f"最新IIP出荷YoY: {ship_yoy.iloc[-1]:+.1f}%（{ship_yoy.index[-1].strftime('%Y年%m月')}）")
    if benchmarks_available and not benchmarks.empty:
        bench_yoy = benchmarks.pct_change(12) * 100
        for col in bench_yoy.columns:
            s = bench_yoy[col].dropna()
            if not s.empty:
                dynamic_parts.append(f"{col} YoY: {s.iloc[-1]:+.1f}%")
    dynamic = _make_guide("現在の状況", dynamic_parts if dynamic_parts else ["データ未取得"])
    insight = _make_guide("分析知見", [
        "SOX vs 日経: 2019-2025年でr=0.82（2013-2018年はr=0.40）→ 日本株の半導体グローバル連動が大幅強化。",
        "SOX YoYはIIP出荷YoYに3ヶ月先行（r=0.55）。SOXが先行指標として有効です。",
        "AMOVA（旧日興AM）: SOXは半導体売上高に先行性を伴い連動。スーパーサイクル論では2030年に市場規模1兆ドル予測。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_correlation_guide():
    """相関分析タブの解説"""
    guide = _make_guide("読み方ガイド", [
        "SI比率(3MA)と各銘柄月次リターンのピアソン相関係数（lag=0のみ）。",
        "年レンジスライダーで対象期間を絞れます。RdBu色: 赤=正相関、青=負相関。",
    ])
    dynamic = _make_guide("現在の状況", [
        "全期間・サブ期間での相関変化を確認してください。2019年以降は多くの銘柄で負相関です。",
    ])
    insight = _make_guide("分析知見", [
        "2019-2025年は全銘柄でlag=0の相関が負に転じています（SI比率の遅行化）。",
        "ただしこのタブはlag=0のみの分析です。リードラグ分析タブで多様なlagでの相関を確認してください。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_leadlag_guide():
    """リードラグ分析タブの解説"""
    guide = _make_guide("読み方ガイド", [
        "lag正値=IIP指標が先行（IIPがN月前の値 vs 今月の株価リターン）。lag負値=IIP指標が遅行。",
        "ヒートマップ: 全銘柄×lag -12〜+12の相関。棒グラフ: 選択銘柄のlag別相関と最大|r|のlag。",
        "期間分割: 2013-2018年 vs 2019年以降で先行性の変化を確認。SOXバー: SOX指数との同時相関。",
    ])
    dynamic = _make_guide("現在の状況", [
        "相関対象・銘柄・期間を変更して、リードラグ構造の変化を確認してください。",
    ])
    insight = _make_guide("分析知見", [
        "2013-2018年: SI比率はlag=0〜+3で正相関（先行〜同時指標として機能）。",
        "2019年以降: 正相関のlagが大きくシフト（lag=+8〜+12）→ 遅行指標化。リアルタイムの投資判断には不向き。",
        "代替としてSOX指数が有効。日本半導体株とlag=0でr=0.6-0.75の強い同時相関。",
    ])
    return html.Div([guide, dynamic, insight])


def _create_divergence_chart():
    """SOX YoY vs IIP出荷YoY 乖離指数チャート（方法①）"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "SOX指数 YoY vs IIP出荷指数 YoY",
            "乖離指数（SOX YoY − IIP出荷YoY）",
        ),
        row_heights=[0.5, 0.5],
    )

    if not benchmarks_available or "フィラデルフィア半導体指数" not in benchmarks.columns:
        fig.add_annotation(text="SOX指数データ未取得", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=500)
        return fig

    sox_yoy = benchmarks["フィラデルフィア半導体指数"].pct_change(12) * 100

    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    ship_yoy = iip_me["shipment_yoy"].dropna()

    # 上段: SOX YoY + IIP出荷YoY
    sox_yoy_clean = sox_yoy.dropna()
    fig.add_trace(
        go.Scatter(
            x=sox_yoy_clean.index, y=sox_yoy_clean,
            name="SOX指数 YoY",
            line=dict(color="#FF5722", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ship_yoy.index, y=ship_yoy,
            name="IIP出荷指数 YoY",
            line=dict(color="#673AB7", width=2),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)

    # 下段: 乖離指数
    common_idx = sox_yoy.dropna().index.intersection(ship_yoy.index)
    if len(common_idx) > 0:
        divergence = sox_yoy.reindex(common_idx) - ship_yoy.reindex(common_idx)
        divergence = divergence.dropna()

        if len(divergence) > 0:
            mean_div = divergence.mean()
            std_div = divergence.std()
            upper = mean_div + std_div
            lower = mean_div - std_div

            colors = [
                "#F44336" if v > upper else ("#FF9800" if v < lower else "#2196F3")
                for v in divergence
            ]

            fig.add_trace(
                go.Bar(
                    x=divergence.index, y=divergence,
                    name="乖離指数",
                    marker_color=colors,
                    hovertemplate="%{x|%Y-%m}<br>乖離: %{y:.1f}pp<extra></extra>",
                ),
                row=2, col=1,
            )

            # ±1σ帯
            fig.add_hline(y=upper, line_dash="dot", line_color="#F44336",
                          annotation_text=f"+1σ ({upper:.1f})", row=2, col=1)
            fig.add_hline(y=lower, line_dash="dot", line_color="#FF9800",
                          annotation_text=f"−1σ ({lower:.1f})", row=2, col=1)
            fig.add_hline(y=mean_div, line_dash="dash", line_color="gray",
                          annotation_text=f"平均 ({mean_div:.1f})", row=2, col=1)

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="前年比 (%)", row=1, col=1)
    fig.update_yaxes(title_text="乖離 (pp)", row=2, col=1)
    return fig


def _create_rolling_corr_chart(window=12):
    """SOX×日経ローリング相関チャート（方法④）"""
    fig = go.Figure()

    if not benchmarks_available or benchmarks.empty:
        fig.add_annotation(text="ベンチマークデータ未取得", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=400)
        return fig, ""

    sox_col = "フィラデルフィア半導体指数"
    nikkei_col = "日経平均"
    if sox_col not in benchmarks.columns or nikkei_col not in benchmarks.columns:
        fig.add_annotation(text="SOXまたは日経平均データなし", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=400)
        return fig, ""

    sox_ret = benchmarks[sox_col].pct_change()
    nikkei_ret = benchmarks[nikkei_col].pct_change()
    rolling_corr = sox_ret.rolling(window).corr(nikkei_ret).dropna()

    if rolling_corr.empty:
        fig.add_annotation(text="データ不足", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=400)
        return fig, ""

    # メインの相関線
    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            name=f"ローリング相関（{window}ヶ月）",
            line=dict(color="#2196F3", width=2),
            hovertemplate="%{x|%Y-%m}<br>r=%{y:.3f}<extra></extra>",
        )
    )

    # 0.8超の過熱帯をハイライト
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(244, 67, 54, 0.1)",
                  layer="below", line_width=0,
                  annotation_text="過熱帯（r > 0.8）",
                  annotation_position="top right")

    fig.add_hline(y=0.8, line_dash="dot", line_color="#F44336", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title=f"SOX × 日経平均 ローリング相関（{window}ヶ月ウィンドウ）",
        height=450,
        template="plotly_white",
        yaxis_title="相関係数",
        yaxis_range=[-1, 1],
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=40),
    )

    # 補助統計: r > 0.8の後の12ヶ月SOXリターン
    stat_text = ""
    overheat = rolling_corr[rolling_corr > 0.8]
    if len(overheat) > 0:
        sox_monthly = benchmarks[sox_col].pct_change()
        future_returns = []
        for dt in overheat.index:
            future_end = dt + pd.DateOffset(months=12)
            future = sox_monthly.loc[dt:future_end]
            if len(future) >= 6:
                cum_ret = (1 + future).prod() - 1
                future_returns.append(cum_ret * 100)
        if future_returns:
            avg_ret = np.mean(future_returns)
            median_ret = np.median(future_returns)
            stat_text = (
                f"r > 0.8局面の後12ヶ月SOXリターン: "
                f"平均 {avg_ret:+.1f}%, 中央値 {median_ret:+.1f}% "
                f"(N={len(future_returns)})"
            )

    return fig, stat_text


def _build_signal_summary_cards():
    """総合シグナルサマリーカード"""
    cards = []

    # 1. 乖離指数の最新値
    div_value = None
    div_color = "#888"
    div_label = "データなし"
    if benchmarks_available and "フィラデルフィア半導体指数" in benchmarks.columns:
        sox_yoy = benchmarks["フィラデルフィア半導体指数"].pct_change(12) * 100
        iip_me = iip.copy()
        iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
        ship_yoy = iip_me["shipment_yoy"].dropna()
        common_idx = sox_yoy.dropna().index.intersection(ship_yoy.index)
        if len(common_idx) > 0:
            divergence = sox_yoy.reindex(common_idx) - ship_yoy.reindex(common_idx)
            divergence = divergence.dropna()
            if len(divergence) > 0:
                div_value = divergence.iloc[-1]
                mean_d = divergence.mean()
                std_d = divergence.std()
                if div_value > mean_d + std_d:
                    div_color = "#F44336"
                    div_label = "過熱警戒"
                elif div_value < mean_d - std_d:
                    div_color = "#FF9800"
                    div_label = "過度な悲観"
                else:
                    div_color = "#4CAF50"
                    div_label = "正常範囲"

    cards.append(_signal_card(
        "乖離指数",
        f"{div_value:+.1f}pp" if div_value is not None else "N/A",
        div_label, div_color,
    ))

    # 2. ローリング相関の最新値
    corr_value = None
    corr_color = "#888"
    corr_label = "データなし"
    if benchmarks_available and not benchmarks.empty:
        sox_col = "フィラデルフィア半導体指数"
        nikkei_col = "日経平均"
        if sox_col in benchmarks.columns and nikkei_col in benchmarks.columns:
            sox_ret = benchmarks[sox_col].pct_change()
            nikkei_ret = benchmarks[nikkei_col].pct_change()
            rc = sox_ret.rolling(12).corr(nikkei_ret).dropna()
            if len(rc) > 0:
                corr_value = rc.iloc[-1]
                if corr_value > 0.8:
                    corr_color = "#F44336"
                    corr_label = "テーマ相場過熱"
                elif corr_value > 0.5:
                    corr_color = "#FF9800"
                    corr_label = "連動やや高"
                else:
                    corr_color = "#4CAF50"
                    corr_label = "正常"

    cards.append(_signal_card(
        "SOX×日経相関(12M)",
        f"{corr_value:.3f}" if corr_value is not None else "N/A",
        corr_label, corr_color,
    ))

    # 3. 現在のSI比率局面
    cards.append(_signal_card(
        "SI比率局面",
        current_phase,
        f"SI比率: {latest['si_ratio_ma3']:.3f}",
        phase_color,
    ))

    return html.Div(
        style={"display": "flex", "gap": "15px", "marginBottom": "20px", "flexWrap": "wrap"},
        children=cards,
    )


def _signal_card(title, value, sublabel, color):
    """シグナルカード（信号灯形式）"""
    # 信号灯の色
    if color == "#F44336":
        light = "🔴"
    elif color == "#FF9800":
        light = "🟡"
    elif color == "#4CAF50":
        light = "🟢"
    else:
        light = "⚪"

    return html.Div(
        style={
            "flex": "1", "minWidth": "200px", "padding": "15px",
            "borderRadius": "8px", "backgroundColor": "#fff",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "textAlign": "center", "borderTop": f"3px solid {color}",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#888", "marginBottom": "5px"}),
            html.Div(f"{light} {value}", style={"fontSize": "20px", "fontWeight": "bold",
                                                  "color": color, "marginBottom": "3px"}),
            html.Div(sublabel, style={"fontSize": "11px", "color": "#666"}),
        ],
    )


def _build_signal_guide():
    """半導体シグナルタブの解説"""
    guide = _make_guide("読み方ガイド", [
        "乖離指数（SOX YoY − IIP出荷YoY）: SOXが先行して上昇する局面（乖離拡大）は、"
        "市場期待が実体経済を大きく上回っていることを示す。±1σ超は警戒シグナル。",
        "ローリング相関（SOX × 日経）: 相関0.8超はテーマ相場の過熱を示唆。"
        "過去の実績では、過熱後12ヶ月のSOXリターンは平均的に低下する傾向。",
        "総合サマリー: 信号灯形式（🟢正常/🟡注意/🔴警戒）で現在の市場状態を直感的に表示。",
    ])
    dynamic_parts = []
    if benchmarks_available and "フィラデルフィア半導体指数" in benchmarks.columns:
        sox_yoy = benchmarks["フィラデルフィア半導体指数"].pct_change(12) * 100
        sox_latest = sox_yoy.dropna()
        if len(sox_latest) > 0:
            dynamic_parts.append(f"最新SOX YoY: {sox_latest.iloc[-1]:+.1f}%")
    iip_ship = iip["shipment_yoy"].dropna()
    if len(iip_ship) > 0:
        dynamic_parts.append(f"最新IIP出荷YoY: {iip_ship.iloc[-1]:+.1f}%")
    dynamic = _make_guide("現在の状況", dynamic_parts if dynamic_parts else ["データ未取得"])
    insight = _make_guide("分析知見", [
        "半導体シグナル.docxに基づく分析フレームワーク。5手法のうち無料データで実装可能な2手法を実装。",
        "方法①（乖離指数）: 乖離が+1σ超の局面の6-12ヶ月後にEPS低下が見られる傾向がある。"
        "ただしAI/データセンター需要による構造変化で、従来の閾値が有効とは限らない点に注意。",
        "方法④（ローリング相関）: 相関0.8超は市場全体が半導体テーマに集中していることを意味し、"
        "テーマ剥落時のドローダウンリスクが高い。逆張りシグナルとして活用可能。",
        "未実装: SEMI BBレシオ/DRAM価格（②）、コンセンサスEPS（③）、CapEx乖離（⑤）は有料データのため対象外。",
    ])
    return html.Div([guide, dynamic, insight])


def _build_signal_tab():
    """半導体シグナルタブのレイアウト"""
    return html.Div([
        html.H3("半導体シグナル分析", style={"marginBottom": "8px"}),
        html.P("SOXとIIPの乖離・相関からEPS低下リスクを検知するフレームワーク",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"}),

        # 総合シグナルサマリー
        _build_signal_summary_cards(),

        # チャート1: 乖離指数
        html.H4("① SOX vs IIP 乖離指数", style={"marginTop": "10px", "marginBottom": "8px"}),
        dcc.Graph(figure=_create_divergence_chart()),

        # チャート2: ローリング相関
        html.H4("④ SOX × 日経 ローリング相関",
                style={"marginTop": "20px", "marginBottom": "8px"}),
        html.Div(
            style={"display": "flex", "gap": "15px", "alignItems": "center",
                   "marginBottom": "10px"},
            children=[
                html.Label("ローリング期間:", style={"whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="signal-rolling-window",
                    options=[
                        {"label": "6ヶ月", "value": 6},
                        {"label": "12ヶ月（デフォルト）", "value": 12},
                        {"label": "24ヶ月", "value": 24},
                    ],
                    value=12,
                    clearable=False,
                    style={"width": "200px"},
                ),
            ],
        ),
        dcc.Graph(id="signal-rolling-corr-chart"),
        html.Div(id="signal-rolling-stat", style={
            "marginTop": "8px", "fontSize": "13px", "color": "#555",
            "padding": "8px", "backgroundColor": "#f9f9f9", "borderRadius": "5px",
        }),

        _build_signal_guide(),
    ])


def _calc_leadlag_corr(target_col, stock_returns, max_lag=12):
    """リードラグ相関を計算（lag -max_lag〜+max_lag）

    target_col: iipの系列（SI比率やYoY）、月末インデックス
    stock_returns: 銘柄月次リターン
    Returns: DataFrame (index=銘柄, columns=lag)
    """
    results = {}
    for stock in stock_returns.columns:
        corrs = []
        for lag in range(-max_lag, max_lag + 1):
            shifted = target_col.shift(lag)
            combined = pd.concat([shifted, stock_returns[stock]], axis=1).dropna()
            if len(combined) < 10:
                corrs.append(np.nan)
            else:
                corrs.append(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
        results[stock] = corrs
    return pd.DataFrame(results, index=range(-max_lag, max_lag + 1)).T


def _build_leadlag_tab():
    """リードラグ分析タブのレイアウト"""
    if not prices_available or prices.empty:
        return html.Div("株価データ未取得のため表示できません。",
                        style={"padding": "40px", "textAlign": "center", "color": "#888"})

    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    common_idx = iip_me.index.intersection(prices.pct_change().dropna(how="all").index)
    if len(common_idx) == 0:
        return html.Div("共通データなし", style={"padding": "40px", "textAlign": "center"})

    min_year = int(common_idx.min().year)
    max_year = int(common_idx.max().year)

    stock_options = [{"label": name, "value": name} for name in prices.columns]
    target_options = [
        {"label": "SI比率(3MA)", "value": "si_ratio_ma3"},
        {"label": "出荷指数 YoY", "value": "shipment_yoy"},
        {"label": "在庫指数 YoY", "value": "inventory_yoy"},
    ]

    return html.Div([
        html.H3("リードラグ分析", style={"marginBottom": "8px"}),
        html.P("IIPサイクル指標と株価リターンのリードラグ相関（lag -12〜+12ヶ月）",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"}),

        # コントロール
        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "center",
                   "marginBottom": "16px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("相関対象:", style={"fontSize": "13px"}),
                    dcc.Dropdown(
                        id="leadlag-target",
                        options=target_options,
                        value="si_ratio_ma3",
                        style={"width": "200px"},
                    ),
                ]),
                html.Div([
                    html.Label("銘柄（棒グラフ用）:", style={"fontSize": "13px"}),
                    dcc.Dropdown(
                        id="leadlag-stock",
                        options=stock_options,
                        value=stock_options[0]["value"] if stock_options else None,
                        style={"width": "250px"},
                    ),
                ]),
                html.Div([
                    html.Label("対象期間（年）:", style={"fontSize": "13px"}),
                    dcc.RangeSlider(
                        id="leadlag-year-range",
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=[min_year, max_year],
                        marks={y: str(y) for y in range(min_year, max_year + 1, 2)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "minWidth": "300px"}),
            ],
        ),

        # チャート1: ヒートマップ
        html.H4("リードラグ相関ヒートマップ", style={"marginTop": "20px", "marginBottom": "8px"}),
        dcc.Graph(id="leadlag-heatmap"),

        # チャート2: 個別銘柄のlag別棒グラフ
        html.H4("個別銘柄 lag別相関係数", style={"marginTop": "20px", "marginBottom": "8px"}),
        dcc.Graph(id="leadlag-bar"),

        # チャート3: 期間分割比較
        html.H4("期間分割比較（先行性→遅行性の変化）",
                style={"marginTop": "20px", "marginBottom": "8px"}),
        dcc.Graph(id="leadlag-period-compare"),

        # チャート4: SOX vs 個別株の同時相関
        html.H4("SOX指数 vs 個別株 同時相関（lag=0）",
                style={"marginTop": "20px", "marginBottom": "8px"}),
        dcc.Graph(id="leadlag-sox-bar"),
        _build_leadlag_guide(),
    ])


@callback(
    Output("leadlag-heatmap", "figure"),
    Output("leadlag-bar", "figure"),
    Output("leadlag-period-compare", "figure"),
    Input("leadlag-target", "value"),
    Input("leadlag-stock", "value"),
    Input("leadlag-year-range", "value"),
    prevent_initial_call=False,
)
def update_leadlag(target_col_name, selected_stock, year_range):
    empty = go.Figure()

    if not prices_available or prices.empty:
        return empty, empty, empty

    start_year, end_year = year_range if year_range else [None, None]

    # 準備: iipを月末に正規化
    iip_me = iip.copy()
    iip_me.index = iip_me.index + pd.offsets.MonthEnd(0)
    monthly_returns = prices.pct_change() * 100

    target = iip_me[target_col_name]

    # 期間フィルタ
    if start_year:
        target = target[target.index.year >= start_year]
        monthly_returns = monthly_returns[monthly_returns.index.year >= start_year]
    if end_year:
        target = target[target.index.year <= end_year]
        monthly_returns = monthly_returns[monthly_returns.index.year <= end_year]

    target = target.dropna()
    monthly_returns = monthly_returns.dropna(how="all")

    if target.empty or monthly_returns.empty:
        return empty, empty, empty

    target_label = {"si_ratio_ma3": "SI比率(3MA)", "shipment_yoy": "出荷YoY",
                    "inventory_yoy": "在庫YoY"}.get(target_col_name, target_col_name)

    # チャート1: ヒートマップ
    leadlag_df = _calc_leadlag_corr(target, monthly_returns)

    fig_heatmap = px.imshow(
        leadlag_df,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        labels={"x": "Lag（月）", "y": "銘柄", "color": "相関係数"},
        title=f"{target_label} × 月次リターン リードラグ相関 ({start_year}〜{end_year}年)",
    )
    fig_heatmap.update_layout(
        height=max(300, len(leadlag_df) * 35 + 100),
        template="plotly_white",
        margin=dict(l=120, r=20, t=60, b=60),
    )
    fig_heatmap.update_xaxes(
        tickvals=list(range(-12, 13, 2)),
        ticktext=[str(x) for x in range(-12, 13, 2)],
    )

    # チャート2: 個別銘柄の棒グラフ
    fig_bar = go.Figure()
    if selected_stock and selected_stock in leadlag_df.index:
        corr_series = leadlag_df.loc[selected_stock]
        lags = corr_series.index.tolist()
        vals = corr_series.values.tolist()
        best_lag = lags[np.nanargmax(np.abs(vals))]

        colors = ["#F44336" if lag == best_lag else
                  ("#2196F3" if v >= 0 else "#FF9800")
                  for lag, v in zip(lags, vals)]

        fig_bar.add_trace(go.Bar(
            x=lags, y=vals,
            marker_color=colors,
            hovertemplate="Lag=%{x}ヶ月<br>r=%{y:.3f}<extra></extra>",
        ))
        fig_bar.add_annotation(
            x=best_lag, y=vals[lags.index(best_lag)],
            text=f"最大|r| lag={best_lag}",
            showarrow=True, arrowhead=2, font=dict(size=11, color="#F44336"),
        )
        fig_bar.update_layout(
            title=f"{selected_stock} × {target_label}（lag別相関）",
            xaxis_title="Lag（月、正=IIPが先行）",
            yaxis_title="相関係数",
            height=350,
            template="plotly_white",
            margin=dict(l=60, r=20, t=60, b=60),
        )
    else:
        fig_bar.add_annotation(text="銘柄を選択してください",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=16))
        fig_bar.update_layout(height=300)

    # チャート3: 期間分割比較（2013-2018 vs 2019-最新）
    split_year = 2019
    iip_me_full = iip.copy()
    iip_me_full.index = iip_me_full.index + pd.offsets.MonthEnd(0)
    returns_full = prices.pct_change() * 100

    target_early = iip_me_full[target_col_name][
        (iip_me_full.index.year >= 2013) & (iip_me_full.index.year < split_year)].dropna()
    returns_early = returns_full[
        (returns_full.index.year >= 2013) & (returns_full.index.year < split_year)].dropna(how="all")

    target_late = iip_me_full[target_col_name][
        iip_me_full.index.year >= split_year].dropna()
    returns_late = returns_full[
        returns_full.index.year >= split_year].dropna(how="all")

    fig_period = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("2013〜2018年", f"2019〜{end_year}年"),
    )

    if not target_early.empty and not returns_early.empty:
        df_early = _calc_leadlag_corr(target_early, returns_early)
        fig_period.add_trace(
            go.Heatmap(
                z=df_early.values,
                x=list(range(-12, 13)),
                y=df_early.index.tolist(),
                colorscale="RdBu", zmin=-1, zmax=1,
                text=np.round(df_early.values, 2),
                texttemplate="%{text}",
                showscale=False,
            ),
            row=1, col=1,
        )

    if not target_late.empty and not returns_late.empty:
        df_late = _calc_leadlag_corr(target_late, returns_late)
        fig_period.add_trace(
            go.Heatmap(
                z=df_late.values,
                x=list(range(-12, 13)),
                y=df_late.index.tolist(),
                colorscale="RdBu", zmin=-1, zmax=1,
                text=np.round(df_late.values, 2),
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="r"),
            ),
            row=2, col=1,
        )

    fig_period.update_layout(
        height=max(500, len(prices.columns) * 35 * 2 + 150),
        template="plotly_white",
        margin=dict(l=120, r=20, t=60, b=60),
    )
    fig_period.update_xaxes(title_text="Lag（月）", row=2, col=1)

    return fig_heatmap, fig_bar, fig_period


@callback(
    Output("leadlag-sox-bar", "figure"),
    Input("leadlag-year-range", "value"),
    prevent_initial_call=False,
)
def update_leadlag_sox(year_range):
    """SOX vs 個別株の同時相関（lag=0）"""
    if not prices_available or prices.empty or not benchmarks_available or benchmarks.empty:
        fig = go.Figure()
        fig.add_annotation(text="SOXまたは株価データ未取得",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    start_year, end_year = year_range if year_range else [None, None]

    sox_col = "フィラデルフィア半導体指数"
    if sox_col not in benchmarks.columns:
        fig = go.Figure()
        fig.add_annotation(text="SOX指数データなし",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    sox_returns = benchmarks[sox_col].pct_change() * 100
    stock_returns = prices.pct_change() * 100

    if start_year:
        sox_returns = sox_returns[sox_returns.index.year >= start_year]
        stock_returns = stock_returns[stock_returns.index.year >= start_year]
    if end_year:
        sox_returns = sox_returns[sox_returns.index.year <= end_year]
        stock_returns = stock_returns[stock_returns.index.year <= end_year]

    corrs = {}
    for stock in stock_returns.columns:
        combined = pd.concat([sox_returns, stock_returns[stock]], axis=1).dropna()
        if len(combined) >= 10:
            corrs[stock] = combined.iloc[:, 0].corr(combined.iloc[:, 1])

    if not corrs:
        fig = go.Figure()
        fig.add_annotation(text="相関計算に必要なデータ不足",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    stocks = list(corrs.keys())
    values = list(corrs.values())
    colors = ["#2196F3" if v >= 0 else "#FF9800" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stocks, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}<br>r=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"SOX指数 vs 個別株 同時相関 ({start_year}〜{end_year}年)",
        yaxis_title="相関係数 (r)",
        yaxis_range=[-1, 1],
        height=400,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-cycle":
        return html.Div([
            dcc.Graph(figure=create_cycle_chart()),
            html.Div(
                style={"display": "flex", "gap": "10px", "justifyContent": "center",
                       "marginTop": "10px"},
                children=[
                    html.Span(f"■ {phase}", style={"color": c["text"], "fontSize": "13px"})
                    for phase, c in PHASE_COLORS.items()
                ],
            ),
            _build_cycle_guide(),
        ])

    elif tab == "tab-inventory-cycle":
        return html.Div([
            html.H3("在庫循環図", style={"marginBottom": "8px"}),
            html.P("出荷指数YoY × 在庫指数YoY の散布図（反時計回りサイクル）",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
            html.Div(
                style={"display": "flex", "gap": "15px", "alignItems": "center",
                       "marginBottom": "10px"},
                children=[
                    html.Label("表示間隔:", style={"whiteSpace": "nowrap"}),
                    dcc.Dropdown(
                        id="invcycle-interval",
                        options=[
                            {"label": "月次（1ヶ月毎）", "value": 1},
                            {"label": "3ヶ月毎", "value": 3},
                            {"label": "6ヶ月毎", "value": 6},
                            {"label": "12ヶ月毎", "value": 12},
                        ],
                        value=1,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
            ),
            dcc.Graph(id="invcycle-chart"),
            _build_inventory_cycle_guide(),
        ])

    elif tab == "tab-stocks":
        stock_options = [{"label": name, "value": name}
                         for name in (prices.columns if prices_available else [])]
        return html.Div([
            html.Div(
                style={"display": "flex", "gap": "15px", "alignItems": "center",
                       "marginBottom": "10px"},
                children=[
                    html.Label("銘柄選択:"),
                    dcc.Dropdown(
                        id="stock-selector",
                        options=stock_options,
                        value=[o["value"] for o in stock_options],
                        multi=True,
                        style={"flex": "1"},
                    ),
                    dcc.Checklist(
                        id="log-scale",
                        options=[{"label": "対数スケール", "value": "log"}],
                        value=["log"],
                    ),
                ],
            ),
            dcc.Graph(id="stock-chart"),
            _build_stock_guide(),
        ])

    elif tab == "tab-overlay":
        stock_options = [{"label": name, "value": name}
                         for name in (prices.columns if prices_available else [])]
        default = stock_options[0]["value"] if stock_options else None
        macro_options = [{"label": col, "value": col}
                         for col in (macro_all.columns if not macro_all.empty else [])]
        return html.Div([
            html.Div(
                style={"display": "flex", "gap": "15px", "alignItems": "center",
                       "marginBottom": "10px", "flexWrap": "wrap"},
                children=[
                    html.Label("銘柄:"),
                    dcc.Dropdown(
                        id="overlay-stock",
                        options=stock_options,
                        value=default,
                        style={"width": "300px"},
                    ),
                    dcc.Checklist(
                        id="overlay-log-scale",
                        options=[{"label": "対数スケール", "value": "log"}],
                        value=[],
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "15px", "alignItems": "center",
                       "marginBottom": "10px"}
                if macro_options else {"display": "none"},
                children=[
                    html.Label("マクロ指標:"),
                    dcc.Dropdown(
                        id="overlay-macro",
                        options=macro_options,
                        value=[],
                        multi=True,
                        placeholder="マクロ指標を選択（複数可）",
                        style={"flex": "1"},
                    ),
                ],
            ),
            dcc.Graph(id="overlay-chart"),
            _build_overlay_guide(),
        ])

    elif tab == "tab-phase-return":
        return _build_phase_return_tab()

    elif tab == "tab-global-index":
        return html.Div([
            html.H3("グローバル指数比較", style={"marginBottom": "8px"}),
            html.P("SOX指数・日経平均・IIP出荷指数の前年比推移",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
            dcc.Graph(figure=create_global_index_chart()),
            _build_global_index_guide(),
        ])

    elif tab == "tab-correlation":
        return _build_correlation_tab()

    elif tab == "tab-leadlag":
        return _build_leadlag_tab()

    elif tab == "tab-signal":
        return _build_signal_tab()


@callback(
    Output("stock-chart", "figure"),
    Input("stock-selector", "value"),
    Input("log-scale", "value"),
    prevent_initial_call=False,
)
def update_stock_chart(selected, log_val):
    return create_stock_chart(selected, "log" in (log_val or []))


@callback(
    Output("overlay-chart", "figure"),
    Input("overlay-stock", "value"),
    Input("overlay-log-scale", "value"),
    Input("overlay-macro", "value"),
    prevent_initial_call=False,
)
def update_overlay(stock_name, log_val, macro_val):
    if stock_name:
        return create_overlay_chart(
            stock_name,
            "log" in (log_val or []),
            macro_val or [],
        )
    return go.Figure()


@callback(
    Output("invcycle-chart", "figure"),
    Input("invcycle-interval", "value"),
    prevent_initial_call=False,
)
def update_inventory_cycle(interval):
    return create_inventory_cycle_chart(interval=interval or 1)


@callback(
    Output("iip-upload-status", "children"),
    Input("iip-upload", "contents"),
    State("iip-upload", "filename"),
    prevent_initial_call=True,
)
def handle_iip_upload(contents, filename):
    """IIPデータのアップロード処理"""
    if contents is None:
        return ""

    global iip
    df = load_iip_from_upload(contents, filename)
    if df is not None:
        iip = calc_cycle_indicator(df)
        return html.Span(
            f"アップロード成功: {filename}（{df.index.min().strftime('%Y-%m')} 〜 "
            f"{df.index.max().strftime('%Y-%m')}、{len(df)}件）。タブを再選択してグラフを更新してください。",
            style={"color": "#4CAF50"},
        )
    return html.Span(
        f"アップロード失敗: {filename}。経産省Excel（出荷/在庫シート付き）または整形済CSV/Excel（date, shipment_index, inventory_index列）が必要です。",
        style={"color": "#F44336"},
    )


@callback(
    Output("signal-rolling-corr-chart", "figure"),
    Output("signal-rolling-stat", "children"),
    Input("signal-rolling-window", "value"),
    prevent_initial_call=False,
)
def update_signal_rolling_corr(window):
    fig, stat_text = _create_rolling_corr_chart(window=window or 12)
    return fig, stat_text


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
