"""Plotly chart construction."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .model import Prediction

UP, DOWN, AMBER = "#00d4a0", "#f87171", "#f59e0b"


def build_chart(data: pd.DataFrame, title: str, prediction: Prediction | None,
                pred_date: date) -> str:
    """Candlestick + volume chart with the next-day forecast marked. Returns HTML."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
    ), row=1, col=1)

    pred_x = [pd.Timestamp(pred_date)]
    if prediction:
        fig.add_trace(go.Scatter(
            x=pred_x, y=[prediction.close], mode="markers",
            name=f"Pred Close ₹{prediction.close:,.2f}",
            marker=dict(color=UP, size=16, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            error_y=dict(type="data", array=[prediction.band], color=UP, thickness=1.5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pred_x, y=[prediction.high], mode="markers",
            name=f"Pred High ₹{prediction.high:,.2f}",
            marker=dict(color=AMBER, size=14, symbol="triangle-up",
                        line=dict(color="white", width=1.5)),
        ), row=1, col=1)

    fig.add_vline(x=pd.Timestamp(pred_date).timestamp() * 1000,
                  line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))

    bar_colours = [UP if c >= o else DOWN for c, o in zip(data["Close"], data["Open"])]
    fig.add_trace(go.Bar(
        x=data.index, y=data["Volume"], name="Volume",
        marker_color=bar_colours, showlegend=False, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono, monospace", size=13, color="#94a3b8")),
        template="plotly_dark", paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
        hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=45, b=20), autosize=True,   # height comes from CSS
    )
    fig.update_xaxes(gridcolor="#1e2d45", zeroline=False,
                     rangebreaks=[dict(bounds=["sat", "mon"])])   # hide weekend gaps
    fig.update_yaxes(gridcolor="#1e2d45", zeroline=False)

    return fig.to_html(include_plotlyjs="cdn", full_html=False,
                       config={"responsive": True, "displayModeBar": False},
                       default_width="100%")
