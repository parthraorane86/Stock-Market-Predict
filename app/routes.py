"""HTTP routes. Kept thin: fetch -> compute -> render."""
from __future__ import annotations

from flask import Blueprint, abort, render_template, request

from .charts import build_chart
from .config import DEFAULT_PERIOD, PERIOD_BUTTONS, PERIODS, STOCKS, TRAIN_PERIOD
from .data import get_chart_data, get_history
from .market_hours import is_market_open, next_trading_day
from .services import build_snapshot, get_prediction

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    return render_template("home.html", stocks=STOCKS, market_open=is_market_open())


@bp.route("/stock/<symbol>")
def stock_detail(symbol: str):
    if symbol not in STOCKS:            # whitelist: never echo arbitrary input
        abort(404)

    period = request.args.get("period", DEFAULT_PERIOD)
    if period not in PERIODS:
        period = DEFAULT_PERIOD

    history = get_history(symbol, TRAIN_PERIOD)
    chart_data = get_chart_data(symbol, period)
    if history is None or chart_data is None:
        return render_template(
            "error.html", title="Data unavailable",
            message="Yahoo Finance didn't return data for this stock. Please try again in a minute.",
        ), 503

    prediction = get_prediction(symbol)
    pred_date = next_trading_day(history.index[-1].date())
    company = STOCKS[symbol]

    short_symbol = symbol.removesuffix(".NS")
    chart_html = build_chart(chart_data, f"{short_symbol} · {PERIODS[period]}", prediction, pred_date)

    return render_template(
        "stock.html",
        symbol=symbol,
        short_symbol=short_symbol,
        company=company,
        period=period,
        period_buttons=PERIOD_BUTTONS,
        snapshot=build_snapshot(history),
        prediction=prediction,
        pred_date=pred_date,
        chart_html=chart_html,
        market_open=is_market_open(),
    )
