"""半導体サイクル分析 WEBダッシュボード

Usage:
    python dashboard/app.py
    → http://localhost:8050
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dash
from dash import dcc, html, callback, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.stock_data import (
    SEMICONDUCTOR_STOCKS,
    fetch_stock_prices,
    fetch_sox_index,
    normalize_prices,
)
from src.iip_data import load_or_fetch_iip, calc_cycle_indicator

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


def create_overlay_chart(stock_name):
    """SI比率と個別銘柄の2軸チャート"""
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

    fig.update_layout(
        height=400,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, t=20, b=40),
    )
    fig.update_yaxes(title_text="株価 (円)", secondary_y=False)
    fig.update_yaxes(title_text="SI比率", secondary_y=True)
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

        # タブ
        dcc.Tabs(
            id="tabs",
            value="tab-cycle",
            children=[
                dcc.Tab(label="サイクル指標", value="tab-cycle"),
                dcc.Tab(label="株価推移", value="tab-stocks"),
                dcc.Tab(label="サイクル×株価", value="tab-overlay"),
                dcc.Tab(label="局面別リターン", value="tab-phase-return"),
                dcc.Tab(label="相関分析", value="tab-correlation"),
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
        legend,
        table,
    ])


def _build_correlation_tab():
    """相関分析タブのレイアウト"""
    if not prices_available or prices.empty:
        return html.Div("株価データ未取得のため表示できません。",
                        style={"padding": "40px", "textAlign": "center", "color": "#888"})

    # 年範囲スライダーの設定
    common_idx = iip.index.intersection(prices.pct_change().dropna().index)
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
    si = iip["si_ratio_ma3"].reindex(monthly_returns.index, method="ffill")

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
        ])

    elif tab == "tab-overlay":
        stock_options = [{"label": name, "value": name}
                         for name in (prices.columns if prices_available else [])]
        default = stock_options[0]["value"] if stock_options else None
        return html.Div([
            html.Div(
                style={"marginBottom": "10px"},
                children=[
                    html.Label("銘柄:"),
                    dcc.Dropdown(
                        id="overlay-stock",
                        options=stock_options,
                        value=default,
                        style={"width": "300px"},
                    ),
                ],
            ),
            dcc.Graph(id="overlay-chart"),
        ])

    elif tab == "tab-phase-return":
        return _build_phase_return_tab()

    elif tab == "tab-correlation":
        return _build_correlation_tab()


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
    prevent_initial_call=False,
)
def update_overlay(stock_name):
    if stock_name:
        return create_overlay_chart(stock_name)
    return go.Figure()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
