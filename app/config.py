"""Central configuration: stock list, constants and model settings."""
from __future__ import annotations

import os
from datetime import time

# ── Stocks (Yahoo Finance symbol -> company name) ────────────────────────────
STOCKS: dict[str, str] = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "SBIN.NS": "State Bank of India",
    "ITC.NS": "ITC",
    "BHARTIARTL.NS": "Bharti Airtel",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
    "BAJFINANCE.NS": "Bajaj Finance",
    "NESTLEIND.NS": "Nestle India",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports & SEZ",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS": "Eicher Motors",
    "ONGC.NS": "Oil & Natural Gas Corporation",
    "POWERGRID.NS": "Power Grid Corporation",
    "BPCL.NS": "Bharat Petroleum",
    "IOC.NS": "Indian Oil Corporation",
    "ETERNAL.NS": "Eternal (formerly Zomato)",      # was ZOMATO.NS
    "M&M.NS": "Mahindra & Mahindra",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "JSWSTEEL.NS": "JSW Steel",
    "COALINDIA.NS": "Coal India",
    "DIVISLAB.NS": "Divi's Laboratories",
    "GRASIM.NS": "Grasim Industries",
    "NTPC.NS": "NTPC",
    "BRITANNIA.NS": "Britannia Industries",
    "TMPV.NS": "Tata Motors Passenger Vehicles",    # was TATAMOTORS.NS
    "CIPLA.NS": "Cipla",
}

# ── Chart periods (Yahoo period -> label) ────────────────────────────────────
PERIODS: dict[str, str] = {
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
    "2y": "2 Years",
    "5y": "5 Years",
    "10y": "10 Years",
    "max": "All Time",
}
PERIOD_BUTTONS: dict[str, str] = {
    "1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y",
    "2y": "2Y", "5y": "5Y", "10y": "10Y", "max": "MAX",
}
DEFAULT_PERIOD = "3mo"

# ── Market hours (NSE, Indian Standard Time) ─────────────────────────────────
TIMEZONE = "Asia/Kolkata"
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_PERIOD = "5y"          # history used to train the model
MARKET_INDEX = "^NSEI"       # Nifty 50, used as market-context feature
HISTORY_TTL = 15 * 60        # seconds to cache price history
PREDICTION_TTL = 30 * 60     # seconds to cache a trained prediction

# ── Model ────────────────────────────────────────────────────────────────────
MIN_TRAIN_ROWS = 150         # minimum usable rows after feature warm-up
CV_SPLITS = 5                # walk-forward folds for the backtest
MAX_ABS_DAILY_RETURN = 0.25  # bigger one-day moves = corporate action (demerger, split)
RIDGE_ALPHA = 10.0
XGB_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1,
)

# ── Server ───────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 5000))
