from datetime import date, datetime

from app.market_hours import IST, is_market_open, is_trading_day, next_trading_day


def test_weekend_is_not_a_trading_day():
    assert not is_trading_day(date(2026, 10, 3))   # Saturday
    assert not is_trading_day(date(2026, 10, 4))   # Sunday
    assert is_trading_day(date(2026, 10, 5))       # Monday


def test_holiday_is_not_a_trading_day():
    assert not is_trading_day(date(2026, 1, 26))   # Republic Day


def test_next_trading_day_skips_weekend():
    assert next_trading_day(date(2026, 10, 2)) == date(2026, 10, 5)   # Fri -> Mon


def test_next_trading_day_skips_holiday():
    # 26 Jan 2026 is a Monday holiday; Friday 23 Jan -> Tuesday 27 Jan
    assert next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)


def test_market_open_uses_indian_time():
    assert is_market_open(datetime(2026, 10, 5, 10, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 10, 5, 16, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 10, 5, 9, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 10, 3, 10, 0, tzinfo=IST))   # Saturday
