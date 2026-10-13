# Stock Market Prediction

🔗 **Live app:** [https://stock-market-predictor-advance.onrender.com/](https://stock-market-predictor-advance.onrender.com/)

A Flask web app that shows live-ish charts for 40 NSE stocks and forecasts the next trading day's
**close** and **high** with a machine-learning model. Every forecast comes with an honest backtest
so you can see how much to trust it.

> Educational project. Predictions are often wrong and are **not financial advice**.

## Project structure

```
app/
├── __init__.py       create_app() – builds the Flask app
├── config.py         stock list, periods, model settings (edit things here)
├── routes.py         home page and stock page (thin – no business logic)
├── data.py           Yahoo Finance download + caching
├── cache.py          small thread-safe TTL cache
├── features.py       feature engineering and dataset preparation
├── model.py          ensemble model, walk-forward backtest, prediction
├── services.py       cached prediction + price snapshot
├── charts.py         Plotly candlestick/volume chart
├── market_hours.py   NSE open/closed, holidays, next trading day (IST)
├── templates/        base.html, home.html, stock.html, error.html
└── static/css/       style.css
tests/                pytest suite (runs offline with synthetic data)
wsgi.py               entry point
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python wsgi.py                     # http://localhost:5000
pytest                             # run the tests
```

## Deploy (Render / any Procfile host)

Start command: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
(already in the `Procfile`). Keep `--workers 1` so all requests share one in-memory cache.

## How the model works

- **Data:** 5 years of daily prices per stock, plus the Nifty 50 index for market context.
- **Features:** all stationary – recent returns, moving-average ratios, volatility, volume ratio,
  gap, RSI, MACD, Bollinger position, day of week, market returns.
- **Targets:** next-day *return* of the close and of the high (converted to prices afterwards).
- **Model:** average of XGBoost and Ridge regression.
- **Backtest:** 5-fold walk-forward validation (always train on the past, test on the future),
  compared against the naive "tomorrow = today" baseline. Results are shown on every stock page.
- **Corporate actions:** an overnight jump above 25% (demerger/split) is treated as a break in the
  price series, and only data after it is used for training.
- **Caching:** price data is cached 15 min and trained forecasts 30 min, so pages load quickly.

## Ideas for later

- Train once a day after market close and save the models with `joblib`.
- Tune hyper-parameters with `Optuna` using time-series cross-validation.
- Add India VIX, USD/INR or sector indices as extra features.
- Classify direction (up/down) with `XGBClassifier` alongside the price forecast.

## Known limitations

- Yahoo Finance data is free but unofficial and can be delayed or occasionally unavailable.
- The holiday calendar comes from the `holidays` package and may differ slightly from NSE's list.
- A stock with less than ~7 months of clean history (e.g. just after a demerger) gets no forecast.
