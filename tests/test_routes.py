def test_home_lists_stocks(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"TCS" in response.data
    assert b"ETERNAL" in response.data and b"ZOMATO" not in response.data


def test_stock_page_renders(client):
    response = client.get("/stock/TCS.NS")
    assert response.status_code == 200
    assert b"Forecast for" in response.data
    assert b"How reliable is this?" in response.data


def test_stock_page_with_period(client):
    assert client.get("/stock/TCS.NS?period=1y").status_code == 200
    assert client.get("/stock/TCS.NS?period=10y").status_code == 200
    assert client.get("/stock/TCS.NS?period=nonsense").status_code == 200   # falls back to default


def test_symbol_with_ampersand(client):
    assert client.get("/stock/M&M.NS").status_code == 200


def test_unknown_symbol_is_404_and_not_reflected(client):
    payload = "<script>alert(1)</script>"
    response = client.get(f"/stock/{payload}")
    assert response.status_code == 404
    assert payload.encode() not in response.data


def test_data_failure_shows_friendly_error(client, monkeypatch):
    from app import data
    monkeypatch.setattr(data, "_download", lambda symbol, period: None)
    data.clear_cache()
    response = client.get("/stock/TCS.NS")
    assert response.status_code == 503
    assert b"Data unavailable" in response.data
