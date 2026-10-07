"""NSE trading-calendar helpers. Everything is evaluated in Indian time (IST)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import holidays

from .config import MARKET_CLOSE, MARKET_OPEN, TIMEZONE

IST = ZoneInfo(TIMEZONE)

# Note: this is the general Indian holiday list. NSE's official list can differ
# slightly (e.g. special Muhurat sessions), so treat it as a good approximation.
_indian_holidays = holidays.India()


def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _indian_holidays


def is_market_open(now: datetime | None = None) -> bool:
    """True if the NSE is in its regular session at `now` (default: current IST time)."""
    now = now.astimezone(IST) if now else now_ist()
    return is_trading_day(now.date()) and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def next_trading_day(from_date: date) -> date:
    """First trading day strictly after `from_date` (skips weekends and holidays)."""
    day = from_date + timedelta(days=1)
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day
