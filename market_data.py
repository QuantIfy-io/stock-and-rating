"""Finviz rating and earnings stamps, Nasdaq closes, bundled parquet fallback."""
from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
ACTIONS_PATH = DATA_DIR / "actions.parquet"
METADATA_PATH = DATA_DIR / "warehouse_metadata.json"

NY = ZoneInfo("America/New_York")
PRICE_START = "2023-06-01"
MARKET = "SPY"

# Liquid names plus the two stocks in the 3 Nov 2026 Schwab market update.
UNIVERSE: dict[str, str] = {
    "RIVN": "Rivian",
    "CRWD": "CrowdStrike",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "INTC": "Intel",
    "PLTR": "Palantir",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks",
    "SNOW": "Snowflake",
    "NFLX": "Netflix",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "C": "Citigroup",
    "V": "Visa",
    "MA": "Mastercard",
    "UNH": "UnitedHealth",
    "LLY": "Eli Lilly",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
    "MCD": "McDonald's",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "CAT": "Caterpillar",
    "BA": "Boeing",
    "GE": "GE Aerospace",
    "HON": "Honeywell",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "F": "Ford",
    "GM": "General Motors",
    "LCID": "Lucid",
    "UBER": "Uber",
    "DIS": "Disney",
    "COIN": "Coinbase",
    "SHOP": "Shopify",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str, *, accept_json: bool = False, attempts: int = 4) -> requests.Response:
    headers = dict(HEADERS)
    if accept_json:
        headers["Accept"] = "application/json"
        headers["Origin"] = "https://www.nasdaq.com"
        headers["Referer"] = "https://www.nasdaq.com/"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=headers, timeout=40)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}") from last_error


def _extract_array(page: str, key: str) -> list:
    marker = page.find(key)
    if marker < 0:
        return []
    start = page.find("[", marker)
    if start < 0:
        return []
    depth = 0
    for index in range(start, len(page)):
        char = page[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(page[start : index + 1])
    return []


def _stamp_to_day(timestamp: int) -> pd.Timestamp:
    day = datetime.fromtimestamp(int(timestamp), tz=NY).date()
    return pd.Timestamp(day)


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\u0026rarr;", " → ").replace("&rarr;", " → ").replace("→", " → ")
    text = html.unescape(text)
    return " ".join(text.split())


def _split_rating(rating: str) -> tuple[str, str]:
    if "→" in rating:
        left, right = rating.split("→", 1)
        return left.strip(), right.strip()
    return rating.strip(), ""


def fetch_finviz_actions(ticker: str) -> pd.DataFrame:
    """Rating changes and earnings stamps embedded in the Finviz quote page."""
    page = _get(f"https://finviz.com/quote.ashx?t={ticker}").text
    events = _extract_array(page, '"chartEvents":')
    rows: list[dict] = []
    for event in events:
        timestamp = event.get("dateTimestamp")
        if not timestamp:
            continue
        day = _stamp_to_day(timestamp)
        kind = event.get("eventType") or ""
        if kind == "chartEvent/earnings":
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "kind": "earnings",
                    "action": "Earnings",
                    "analyst": "",
                    "rating": "",
                    "from_grade": "",
                    "to_grade": "",
                    "target": "",
                    "fiscal_period": event.get("fiscalPeriod") or "",
                    "eps_actual": event.get("epsActual"),
                    "eps_estimate": event.get("epsEstimate"),
                }
            )
            continue
        if kind != "chartEvent/ratings":
            continue
        ratings = event.get("ratings") or []
        if isinstance(ratings, dict):
            ratings = [ratings]
        for rating in ratings:
            label = _clean_text(rating.get("rating"))
            from_grade, to_grade = _split_rating(label)
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "kind": "rating",
                    "action": _clean_text(rating.get("action")),
                    "analyst": _clean_text(rating.get("analyst")),
                    "rating": label,
                    "from_grade": from_grade,
                    "to_grade": to_grade,
                    "target": _clean_text(rating.get("targetPrice")),
                    "fiscal_period": "",
                    "eps_actual": None,
                    "eps_estimate": None,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        subset=["ticker", "date", "kind", "action", "analyst", "rating", "fiscal_period"]
    ).reset_index(drop=True)


def _parse_number(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).replace("$", "").replace(",", "").strip()
    if text in {"", "N/A", "--", "null"}:
        return float("nan")
    return float(text)


def fetch_nasdaq_prices(symbol: str, asset: str, start: str = PRICE_START) -> pd.DataFrame:
    """Daily closes. Nasdaq's historical rows are split-adjusted."""
    end = datetime.now(NY).date().isoformat()
    url = (
        "https://api.nasdaq.com/api/quote/"
        f"{symbol}/historical?assetclass={asset}&fromdate={start}&todate={end}&limit=10000"
    )
    payload = _get(url, accept_json=True).json()
    table = ((payload.get("data") or {}).get("tradesTable") or {})
    rows = table.get("rows") or []
    if not rows:
        raise RuntimeError(f"Nasdaq returned no rows for {symbol}.")
    frame = pd.DataFrame(
        {
            "ticker": symbol,
            "date": pd.to_datetime([row.get("date") for row in rows], format="%m/%d/%Y"),
            "close": [_parse_number(row.get("close")) for row in rows],
            "volume": [_parse_number(row.get("volume")) for row in rows],
        }
    )
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_bundle(tickers: list[str] | None = None, *, pause: float = 0.45) -> dict[str, pd.DataFrame]:
    names = tickers or list(UNIVERSE)
    price_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for ticker in names:
        try:
            action_frames.append(fetch_finviz_actions(ticker))
        except Exception as exc:  # noqa: BLE001 — keep the rest of the universe
            errors.append(f"{ticker} ratings: {exc}")
        time.sleep(pause)
        try:
            price_frames.append(fetch_nasdaq_prices(ticker, "stocks"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ticker} prices: {exc}")
        time.sleep(pause / 2)

    try:
        price_frames.append(fetch_nasdaq_prices(MARKET, "etf"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{MARKET} prices: {exc}")

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    if errors:
        print("Skipped:")
        for line in errors:
            print(" ", line)
    return {"prices": prices, "actions": actions}


def write_warehouse(bundle: dict[str, pd.DataFrame]) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices = bundle["prices"]
    actions = bundle["actions"]
    prices.to_parquet(PRICES_PATH, index=False)
    actions.to_parquet(ACTIONS_PATH, index=False)
    ratings = actions[actions["kind"] == "rating"] if not actions.empty else actions
    meta = {
        "written_at": datetime.now(NY).isoformat(timespec="seconds"),
        "price_start": str(prices["date"].min().date()) if not prices.empty else None,
        "price_end": str(prices["date"].max().date()) if not prices.empty else None,
        "n_price_rows": int(len(prices)),
        "n_tickers": int(prices.loc[prices["ticker"] != MARKET, "ticker"].nunique()) if not prices.empty else 0,
        "n_rating_rows": int(len(ratings)),
        "n_earnings_rows": int((actions["kind"] == "earnings").sum()) if not actions.empty else 0,
        "source": "Finviz chart events for ratings and earnings; Nasdaq historical closes; SPY is the market",
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def load_warehouse() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not PRICES_PATH.exists() or not ACTIONS_PATH.exists():
        raise FileNotFoundError(
            "Missing data/prices.parquet or data/actions.parquet. Run python download_data.py first."
        )
    prices = pd.read_parquet(PRICES_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    actions["date"] = pd.to_datetime(actions["date"])
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return prices, actions, meta
