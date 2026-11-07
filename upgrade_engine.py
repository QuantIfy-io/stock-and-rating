"""Event study: does an upgrade to Buy line up with a later abnormal rise?

The rating day is not a randomized treatment. The tests below are the
predictions a causal story would have to survive:

- the abnormal move should show up after the upgrade, not only before it
- it should remain on days with no earnings print nearby
- it should remain against a quiet session for the same stock
- a downgrade should not be required to look like the mirror image, but the
  upgrade day and the next week are separate questions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_data import MARKET, UNIVERSE

EST_LO = -120
EST_HI = -21
MIN_EST = 60
ISOLATED_PAD = 5
EARNINGS_PAD = 2
PLACEBO_NEAR = 15
PLACEBO_FAR = 80

WINDOWS: dict[str, tuple[int, int]] = {
    "pre5": (-5, -1),
    "day0": (0, 0),
    "post5": (1, 5),
    "post21": (1, 21),
}

WINDOW_LABELS = {
    "pre5": "Five sessions before",
    "day0": "Upgrade session",
    "post5": "Next five sessions",
    "post21": "Next 21 sessions",
}

SPECS = ("all", "isolated", "quiet")
SPEC_LABELS = {
    "all": "All upgrade days",
    "isolated": "Isolated upgrades",
    "quiet": "Quiet day",
}


def _complete_sum(values: np.ndarray) -> float:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(values.sum())


def _fit_market(stock: np.ndarray, market: np.ndarray, pos: int) -> tuple[float, float] | None:
    lo = pos + EST_LO
    hi = pos + EST_HI
    if lo < 0 or hi >= pos:
        return None
    y = stock[lo : hi + 1]
    x = market[lo : hi + 1]
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < MIN_EST:
        return None
    design = np.column_stack([np.ones(int(mask.sum())), x[mask]])
    coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
    return float(coef[0]), float(coef[1])


def _abnormal_path(
    stock: np.ndarray,
    market: np.ndarray,
    pos: int,
    alpha: float,
    beta: float,
    start: int,
    end: int,
) -> np.ndarray:
    out = np.full(end - start + 1, np.nan)
    for offset, cursor in enumerate(range(start, end + 1)):
        index = pos + cursor
        if index < 0 or index >= len(stock):
            continue
        if not np.isfinite(stock[index]) or not np.isfinite(market[index]):
            continue
        out[offset] = stock[index] - alpha - beta * market[index]
    return out


def _window_value(path: np.ndarray, start: int, end: int, origin: int) -> float:
    left = start - origin
    right = end - origin
    return _complete_sum(path[left : right + 1])


def _ols_cluster(
    y: np.ndarray, design: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """OLS with ticker-clustered standard errors. Returns coef, se, t, n, G."""
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    y = np.asarray(y, float)[mask]
    design = np.asarray(design, float)[mask]
    clusters = np.asarray(clusters)[mask]
    n, k = design.shape
    empty = np.full(k, np.nan)
    if n <= k + 1:
        return empty, empty, empty, int(n), 0
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return empty, empty, empty, int(n), 0
    meat = np.zeros((k, k))
    groups = pd.unique(clusters)
    g_count = int(len(groups))
    for group in groups:
        chosen = clusters == group
        score = design[chosen].T @ residual[chosen]
        meat += np.outer(score, score)
    scale = (g_count / (g_count - 1)) * ((n - 1) / (n - k)) if g_count > 1 else 1.0
    variance = scale * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(variance), 0, None))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return beta, se, t_stat, int(n), g_count


def _summarize(frame: pd.DataFrame, column: str, cluster: str = "ticker") -> dict:
    part = frame.loc[np.isfinite(frame[column]), [column, cluster]].copy()
    y = part[column].to_numpy(float)
    n = int(len(y))
    if n == 0:
        return {
            "n": 0,
            "n_tickers": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "hit": float("nan"),
        }
    beta, se, t_stat, n, g_count = _ols_cluster(y, np.ones((n, 1)), part[cluster].to_numpy())
    return {
        "n": n,
        "n_tickers": g_count,
        "mean": float(beta[0]),
        "median": float(np.median(y)),
        "se": float(se[0]),
        "t": float(t_stat[0]),
        "hit": float(np.mean(y > 0)),
    }


def _spec_mask(frame: pd.DataFrame, spec: str) -> pd.Series:
    if spec == "all":
        return pd.Series(True, index=frame.index)
    if spec == "isolated":
        return frame["isolated"].astype(bool)
    if spec == "quiet":
        return frame["isolated"].astype(bool) & ~frame["earnings_adjacent"].astype(bool)
    raise KeyError(spec)


def _stock_day_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (ticker, day), part in ratings.groupby(["ticker", "date"], sort=False):
        actions = part["action"].str.lower()
        n_up = int(actions.eq("upgrade").sum())
        n_down = int(actions.eq("downgrade").sum())
        if n_up and not n_down:
            direction = "upgrade"
        elif n_down and not n_up:
            direction = "downgrade"
        elif n_up and n_down:
            direction = "mixed"
        else:
            direction = "other"
        firms = ", ".join(sorted({name for name in part["analyst"] if name}))
        detail = "; ".join(
            f"{row.analyst}: {row.rating}".strip(": ") for row in part.itertuples(index=False)
        )
        rows.append(
            {
                "ticker": ticker,
                "name": UNIVERSE.get(ticker, ticker),
                "date": pd.Timestamp(day),
                "direction": direction,
                "n_up": n_up,
                "n_down": n_down,
                "firms": firms,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def _attach_paths(events: pd.DataFrame, prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if MARKET not in close.columns:
        raise RuntimeError("SPY prices are missing from the warehouse.")
    returns = close.pct_change()
    market = returns[MARKET]
    earnings_by_ticker = {
        ticker: set(pd.to_datetime(part["date"]))
        for ticker, part in earnings.groupby("ticker")
    }
    rating_dates = {
        ticker: set(pd.to_datetime(part["date"]))
        for ticker, part in events.groupby("ticker")
    }

    built: list[dict] = []
    for ticker, part in events.groupby("ticker"):
        if ticker not in returns.columns:
            continue
        both = returns[ticker].notna() & market.notna()
        calendar = returns.index[both]
        if len(calendar) < MIN_EST + 30:
            continue
        stock_r = returns.loc[calendar, ticker].to_numpy(float)
        mkt_r = market.loc[calendar].to_numpy(float)
        pos_of = {pd.Timestamp(day): index for index, day in enumerate(calendar)}

        def session_pos(day: pd.Timestamp) -> int | None:
            if day in pos_of:
                return pos_of[day]
            later = calendar[calendar >= day]
            if len(later) == 0:
                return None
            return pos_of[pd.Timestamp(later[0])]

        earn_pos = {
            session_pos(pd.Timestamp(day))
            for day in earnings_by_ticker.get(ticker, ())
        }
        earn_pos.discard(None)
        rate_pos = {pos_of[pd.Timestamp(day)] for day in rating_dates.get(ticker, ()) if pd.Timestamp(day) in pos_of}

        for row in part.itertuples(index=False):
            day = pd.Timestamp(row.date)
            if day not in pos_of:
                continue
            pos = pos_of[day]
            fit = _fit_market(stock_r, mkt_r, pos)
            if fit is None:
                continue
            alpha, beta = fit
            path = _abnormal_path(stock_r, mkt_r, pos, alpha, beta, -5, 21)
            record = row._asdict()
            record.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "raw": float(stock_r[pos]),
                    "market": float(mkt_r[pos]),
                    "ar0": _window_value(path, 0, 0, -5),
                    "car_pre5": _window_value(path, -5, -1, -5),
                    "car_post5": _window_value(path, 1, 5, -5),
                    "car_post21": _window_value(path, 1, 21, -5),
                    "earnings_adjacent": any(abs(pos - earn) <= EARNINGS_PAD for earn in earn_pos),
                    "isolated": not any(0 < abs(pos - other) <= ISOLATED_PAD for other in rate_pos),
                }
            )
            placebo = _placebo(stock_r, mkt_r, calendar, pos, rate_pos, earn_pos)
            record.update(placebo)
            built.append(record)
    if not built:
        return pd.DataFrame()
    return pd.DataFrame(built)


def _placebo(
    stock: np.ndarray,
    market: np.ndarray,
    calendar: pd.DatetimeIndex,
    pos: int,
    rate_pos: set[int],
    earn_pos: set[int],
) -> dict:
    blank = {
        "placebo_date": pd.NaT,
        "placebo_raw": float("nan"),
        "placebo_ar0": float("nan"),
        "placebo_pre5": float("nan"),
        "placebo_post5": float("nan"),
        "placebo_post21": float("nan"),
    }
    for back in range(PLACEBO_NEAR, PLACEBO_FAR + 1):
        cursor = pos - back
        if cursor + EST_LO < 0:
            break
        if any(abs(cursor - other) <= ISOLATED_PAD for other in rate_pos):
            continue
        if any(abs(cursor - earn) <= EARNINGS_PAD for earn in earn_pos):
            continue
        fit = _fit_market(stock, market, cursor)
        if fit is None:
            continue
        alpha, beta = fit
        path = _abnormal_path(stock, market, cursor, alpha, beta, -5, 21)
        blank.update(
            {
                "placebo_date": pd.Timestamp(calendar[cursor]),
                "placebo_raw": float(stock[cursor]),
                "placebo_ar0": _window_value(path, 0, 0, -5),
                "placebo_pre5": _window_value(path, -5, -1, -5),
                "placebo_post5": _window_value(path, 1, 5, -5),
                "placebo_post21": _window_value(path, 1, 21, -5),
            }
        )
        return blank
    return blank


def _summary_table(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    upgrades = events[events["direction"] == "upgrade"]
    downgrades = events[events["direction"] == "downgrade"]
    outcome = {
        "pre5": ("car_pre5", "placebo_pre5"),
        "day0": ("ar0", "placebo_ar0"),
        "post5": ("car_post5", "placebo_post5"),
        "post21": ("car_post21", "placebo_post21"),
    }
    for spec in SPECS:
        up = upgrades.loc[_spec_mask(upgrades, spec)]
        down = downgrades.loc[_spec_mask(downgrades, spec)]
        for window, (col, placebo_col) in outcome.items():
            for direction, frame, use_col in (
                ("upgrade", up, col),
                ("downgrade", down, col),
                ("placebo", up, placebo_col),
            ):
                stats = _summarize(frame, use_col)
                rows.append({"spec": spec, "direction": direction, "window": window, **stats})
    return pd.DataFrame(rows)


def _paired_frame(upgrades: pd.DataFrame, outcome: str, placebo_outcome: str) -> pd.DataFrame:
    need = list(
        dict.fromkeys(
            [
                "ticker",
                "earnings_adjacent",
                "car_pre5",
                "ar0",
                outcome,
                "placebo_pre5",
                "placebo_ar0",
                placebo_outcome,
            ]
        )
    )
    part = upgrades.dropna(subset=[outcome, placebo_outcome, "car_pre5", "placebo_pre5"]).copy()
    if part.empty:
        return part
    treated = part[need].copy()
    treated["upgrade"] = 1.0
    treated["earnings"] = treated["earnings_adjacent"].astype(float)
    treated["pre"] = treated["car_pre5"]
    treated["y"] = treated[outcome]
    control = part[need].copy()
    control["upgrade"] = 0.0
    control["earnings"] = 0.0
    control["pre"] = control["placebo_pre5"]
    control["ar0"] = control["placebo_ar0"]
    control["y"] = control[placebo_outcome]
    stacked = pd.concat([treated, control], ignore_index=True)
    return stacked


def _regression_table(events: pd.DataFrame) -> pd.DataFrame:
    upgrades = events[events["direction"] == "upgrade"]
    specs_out: list[dict] = []
    designs = {
        "session": ("ar0", "placebo_ar0", ["upgrade", "earnings", "pre"]),
        "next5": ("car_post5", "placebo_post5", ["upgrade", "earnings", "pre"]),
        "next5_after_session": ("car_post5", "placebo_post5", ["upgrade", "earnings", "pre", "ar0"]),
    }
    term_names = {
        "upgrade": "Upgrade",
        "earnings": "Earnings within two sessions",
        "pre": "Prior five sessions",
        "ar0": "Upgrade session",
    }
    outcome_names = {
        "session": "Upgrade session",
        "next5": "Next five sessions",
        "next5_after_session": "Next five sessions, after the session move",
    }
    for spec in SPECS:
        sample = upgrades.loc[_spec_mask(upgrades, spec)]
        for key, (col, placebo_col, terms) in designs.items():
            paired = _paired_frame(sample, col, placebo_col)
            if paired.empty:
                continue
            # A quiet-tape sample has no earnings variation, so that column is dropped.
            used = [term for term in terms if paired[term].nunique(dropna=True) > 1]
            design = np.column_stack(
                [np.ones(len(paired)), *[paired[term].to_numpy(float) for term in used]]
            )
            beta, se, t_stat, n, g_count = _ols_cluster(
                paired["y"].to_numpy(float), design, paired["ticker"].to_numpy()
            )
            labels = ["Intercept", *[term_names[term] for term in used]]
            for index, label in enumerate(labels):
                specs_out.append(
                    {
                        "spec": spec,
                        "outcome_key": key,
                        "outcome": outcome_names[key],
                        "term": label,
                        "coef": float(beta[index]),
                        "se": float(se[index]),
                        "t": float(t_stat[index]),
                        "n": n,
                        "n_tickers": g_count,
                    }
                )
    return pd.DataFrame(specs_out)


def _lookup_case(events: pd.DataFrame, ticker: str, day: str, firm: str) -> pd.Series | None:
    part = events[(events["ticker"] == ticker) & (events["date"] == pd.Timestamp(day))]
    if part.empty:
        return None
    hit = part[part["firms"].str.contains(firm, case=False, na=False)]
    row = hit.iloc[0] if not hit.empty else part.iloc[0]
    return row


def _session_return(prices: pd.DataFrame, ticker: str, day: str) -> float:
    part = prices[prices["ticker"] == ticker].sort_values("date")
    part = part.set_index("date")["close"]
    day_ts = pd.Timestamp(day)
    if day_ts not in part.index:
        return float("nan")
    pos = part.index.get_loc(day_ts)
    if isinstance(pos, slice) or pos == 0:
        return float("nan")
    prev = float(part.iloc[pos - 1])
    if prev == 0:
        return float("nan")
    return float(part.iloc[pos]) / prev - 1.0


def _close_on(prices: pd.DataFrame, ticker: str, day: str) -> float:
    part = prices[(prices["ticker"] == ticker) & (prices["date"] == pd.Timestamp(day))]
    if part.empty:
        return float("nan")
    return float(part["close"].iloc[0])


def build_study(prices: pd.DataFrame, actions: pd.DataFrame) -> dict:
    ratings = actions[actions["kind"] == "rating"].copy()
    earnings = actions[actions["kind"] == "earnings"].copy()
    days = _stock_day_ratings(ratings)
    events = _attach_paths(days, prices, earnings)
    summary = _summary_table(events) if not events.empty else pd.DataFrame()
    regression = _regression_table(events) if not events.empty else pd.DataFrame()
    case = {
        "rivn": _lookup_case(events, "RIVN", "2026-11-02", "UBS"),
        "rivn_prior": _lookup_case(events, "RIVN", "2026-10-30", "Jefferies"),
        "crwd": _lookup_case(events, "CRWD", "2026-11-02", "Deutsche"),
    }
    rivn_earn = earnings[
        (earnings["ticker"] == "RIVN")
        & (earnings["date"] >= "2026-10-01")
        & (earnings["date"] <= "2026-11-02")
    ].sort_values("date")
    case_tape = {
        "rivn_close_mon": _close_on(prices, "RIVN", "2026-11-02"),
        "rivn_close_fri": _close_on(prices, "RIVN", "2026-11-06"),
        "rivn_close_prior": _close_on(prices, "RIVN", "2026-10-30"),
        "crwd_close_mon": _close_on(prices, "CRWD", "2026-11-02"),
        "spy_mon": _session_return(prices, MARKET, "2026-11-02"),
        "rivn_mon": _session_return(prices, "RIVN", "2026-11-02"),
        "crwd_mon": _session_return(prices, "CRWD", "2026-11-02"),
        "rivn_earnings_date": None if rivn_earn.empty else pd.Timestamp(rivn_earn.iloc[-1]["date"]),
        "rivn_earnings_period": "" if rivn_earn.empty else str(rivn_earn.iloc[-1]["fiscal_period"]),
        "rivn_earnings_raw": (
            float("nan")
            if rivn_earn.empty
            else _session_return(prices, "RIVN", str(pd.Timestamp(rivn_earn.iloc[-1]["date"]).date()))
        ),
        "rivn_fri_from_mon": (
            _close_on(prices, "RIVN", "2026-11-06") / _close_on(prices, "RIVN", "2026-11-02") - 1.0
        ),
    }
    return {
        "events": events,
        "summary": summary,
        "regression": regression,
        "case": case,
        "case_tape": case_tape,
        "prices": prices,
        "actions": actions,
    }


def cell(summary: pd.DataFrame, spec: str, direction: str, window: str) -> pd.Series:
    hit = summary[
        (summary["spec"] == spec)
        & (summary["direction"] == direction)
        & (summary["window"] == window)
    ]
    if hit.empty:
        return pd.Series(
            {"n": 0, "n_tickers": 0, "mean": np.nan, "median": np.nan, "se": np.nan, "t": np.nan, "hit": np.nan}
        )
    return hit.iloc[0]


def coef(regression: pd.DataFrame, spec: str, outcome_key: str, term: str) -> pd.Series | None:
    hit = regression[
        (regression["spec"] == spec)
        & (regression["outcome_key"] == outcome_key)
        & (regression["term"] == term)
    ]
    if hit.empty:
        return None
    return hit.iloc[0]


def pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:+.{digits}f}%"


def describe_mean(stats: pd.Series) -> str:
    return (
        f"{pct(stats['mean'])} across {int(stats['n'])} events "
        f"in {int(stats['n_tickers'])} tickers (t {stats['t']:+.2f})"
    )
