"""Streamlit: an upgrade to Buy, the session, and the week after."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import MARKET, UNIVERSE, load_warehouse
from upgrade_engine import (
    WINDOW_LABELS,
    _summarize,
    build_study,
    cell,
    coef,
    pct,
)

SCHWAB_UPDATE = "https://www.schwab.com/learn/story/tuesdays-schwab-market-update-podcast"
CASE_START = "2026-08-03"
CASE_END = "2026-11-06"

AMBER = "#e07a3d"
BLUE = "#4c8bf5"
TEAL = "#38a8a4"
RED = "#d45d79"
MUTED = "#8a93a3"
TEXT = "#d7dbe3"

PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.28,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(0,0,0,0)",
)


def _style(fig: go.Figure, *, title: str, height: int, ytitle: str, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        showlegend=showlegend,
        legend=PLOTLY_LEGEND if showlegend else dict(),
        margin=dict(l=56, r=16, t=52, b=110 if showlegend else 48),
        height=height,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        yaxis_title=ytitle,
        barmode="group",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def _mark(
    fig: go.Figure,
    day: pd.Timestamp,
    label: str,
    color: str,
    *,
    y: float = 0.96,
    anchor: str = "left",
) -> None:
    x = pd.Timestamp(day).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(width=1, dash="dot", color=color),
    )
    if not label:
        return
    fig.add_annotation(
        x=x,
        y=y,
        yref="paper",
        text=label,
        showarrow=False,
        xanchor=anchor,
        yanchor="bottom",
        font=dict(size=11, color=color),
    )


@st.cache_data(show_spinner=False)
def cached_study() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    prices, actions, meta = load_warehouse()
    study = build_study(prices, actions)
    case = {
        key: None if row is None else row.to_dict()
        for key, row in study["case"].items()
    }
    tape = dict(study["case_tape"])
    if tape.get("rivn_earnings_date") is not None:
        tape["rivn_earnings_date"] = pd.Timestamp(tape["rivn_earnings_date"])
    return study["events"], study["summary"], study["regression"], case, tape, meta


def _indexed(prices: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    close = (
        prices[prices["ticker"].isin([ticker, MARKET])]
        .pivot(index="date", columns="ticker", values="close")
        .sort_index()
    )
    window = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if window.empty or ticker not in window.columns:
        return pd.DataFrame()
    base = window.iloc[0]
    return window.divide(base).mul(100.0)


def _price_chart(
    indexed: pd.DataFrame,
    ticker: str,
    color: str,
    title: str,
    marks: list[tuple[pd.Timestamp, str, str, float, str]],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indexed.index,
            y=indexed[ticker],
            name=ticker,
            mode="lines",
            line=dict(color=color, width=2.4),
            hovertemplate="%{y:.1f}<extra>" + ticker + "</extra>",
        )
    )
    if MARKET in indexed.columns:
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=indexed[MARKET],
                name="SPY",
                mode="lines",
                line=dict(color=MUTED, width=1.6),
                hovertemplate="%{y:.1f}<extra>SPY</extra>",
            )
        )
    fig = _style(fig, title=title, height=440, ytitle="Index, first day = 100")
    for day, label, mark_color, y, anchor in marks:
        _mark(fig, day, label, mark_color, y=y, anchor=anchor)
    return fig


def _bar_value(summary: pd.DataFrame, spec: str, direction: str, window: str) -> tuple[float, float]:
    stats = cell(summary, spec, direction, window)
    mean = float(stats["mean"]) * 100.0
    se = float(stats["se"]) * 100.0
    return mean, se


def _panel_bars(summary: pd.DataFrame) -> go.Figure:
    windows = ["pre5", "day0", "post5"]
    labels = [WINDOW_LABELS[key].replace("Upgrade session", "Rating session") for key in windows]
    series = (
        ("Quiet-tape upgrades", "quiet", "upgrade", AMBER),
        ("Matched quiet days", "quiet", "placebo", TEAL),
        ("Quiet-tape downgrades", "quiet", "downgrade", RED),
    )
    fig = go.Figure()
    for name, spec, direction, color in series:
        means, errors = [], []
        for window in windows:
            mean, se = _bar_value(summary, spec, direction, window)
            means.append(mean)
            errors.append(1.96 * se if np.isfinite(se) else 0.0)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=means,
                name=name,
                marker_color=color,
                error_y=dict(type="data", array=errors, thickness=1.2, width=4, color=TEXT),
            )
        )
    return _style(
        fig,
        title="Abnormal return around the rating, quiet tape",
        height=460,
        ytitle="Percent",
    )


def _earnings_bars(events: pd.DataFrame) -> go.Figure:
    upgrades = events[events["direction"] == "upgrade"]
    groups = (
        ("Quiet tape", upgrades["isolated"] & ~upgrades["earnings_adjacent"], AMBER),
        ("Earnings within two sessions", upgrades["earnings_adjacent"], BLUE),
    )
    fig = go.Figure()
    for name, mask, color in groups:
        stats = _summarize(upgrades.loc[mask], "ar0")
        se = 1.96 * float(stats["se"]) * 100.0 if np.isfinite(stats["se"]) else 0.0
        fig.add_trace(
            go.Bar(
                x=["Rating session"],
                y=[float(stats["mean"]) * 100.0],
                name=f"{name} (n={int(stats['n'])})",
                marker_color=color,
                error_y=dict(type="data", array=[se], thickness=1.2, width=4, color=TEXT),
            )
        )
    return _style(
        fig,
        title="Earnings weeks and a quiet tape",
        height=420,
        ytitle="Percent",
    )


def _histogram(events: pd.DataFrame, rivian_ar: float) -> go.Figure:
    quiet = events[
        (events["direction"] == "upgrade")
        & events["isolated"]
        & ~events["earnings_adjacent"]
    ]
    values = quiet["ar0"].dropna().mul(100.0)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            name="Quiet-tape upgrades",
            marker_color=AMBER,
            nbinsx=28,
            hovertemplate="%{x:.1f}%<extra></extra>",
        )
    )
    fig = _style(
        fig,
        title="Quiet-tape upgrade sessions",
        height=420,
        ytitle="Upgrades",
        showlegend=False,
    )
    if np.isfinite(rivian_ar):
        x = 100.0 * rivian_ar
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(width=1.5, dash="dot", color=BLUE),
        )
        fig.add_annotation(
            x=x,
            y=1,
            yref="paper",
            text="Rivian 2 Nov",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11, color=BLUE),
        )
    fig.update_xaxes(title="Abnormal return, percent")
    return fig


def _lookup_chart(prices: pd.DataFrame, events: pd.DataFrame, actions: pd.DataFrame, ticker: str) -> go.Figure:
    indexed = _indexed(prices, ticker, "2025-01-02", CASE_END)
    fig = go.Figure()
    if indexed.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=indexed.index,
            y=indexed[ticker],
            name=ticker,
            mode="lines",
            line=dict(color=AMBER, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indexed.index,
            y=indexed[MARKET],
            name="SPY",
            mode="lines",
            line=dict(color=MUTED, width=1.5),
        )
    )
    part = events[(events["ticker"] == ticker) & (events["date"] >= indexed.index[0])]
    for direction, color, symbol, name in (
        ("upgrade", AMBER, "triangle-up", "Upgrade"),
        ("downgrade", RED, "triangle-down", "Downgrade"),
    ):
        days = part.loc[part["direction"] == direction, "date"]
        days = [day for day in days if day in indexed.index]
        if not days:
            continue
        fig.add_trace(
            go.Scatter(
                x=days,
                y=indexed.loc[days, ticker],
                name=name,
                mode="markers",
                marker=dict(color=color, symbol=symbol, size=10, line=dict(width=1, color=TEXT)),
            )
        )
    earnings = actions[
        (actions["ticker"] == ticker)
        & (actions["kind"] == "earnings")
        & (actions["date"] >= indexed.index[0])
        & (actions["date"] <= indexed.index[-1])
    ]
    for day in earnings["date"]:
        if pd.Timestamp(day) in indexed.index:
            _mark(fig, pd.Timestamp(day), "", TEAL)
    return _style(fig, title=f"{UNIVERSE.get(ticker, ticker)} since 2025, indexed to 100", height=460, ytitle="Index, first day = 100")


def _event_table(events: pd.DataFrame) -> pd.DataFrame:
    show = events[events["direction"].isin(["upgrade", "downgrade"])].copy()
    show = show.sort_values(["date", "ticker"], ascending=[False, True])
    out = pd.DataFrame(
        {
            "Date": show["date"].dt.strftime("%Y-%m-%d"),
            "Ticker": show["ticker"],
            "Name": show["name"],
            "Direction": show["direction"],
            "Firms": show["firms"],
            "Session": show["raw"].map(lambda v: pct(v)),
            "Abnormal": show["ar0"].map(lambda v: pct(v)),
            "Prior 5 sessions": show["car_pre5"].map(lambda v: pct(v)),
            "Next 5 sessions": show["car_post5"].map(lambda v: pct(v)),
            "Earnings nearby": show["earnings_adjacent"].map(lambda v: "Yes" if v else "No"),
            "Isolated": show["isolated"].map(lambda v: "Yes" if v else "No"),
        }
    )
    return out


def _metric_caption(stats: pd.Series) -> str:
    if int(stats["n"]) == 0:
        return "No events in this window."
    return (
        f"Mean {pct(stats['mean'])}. Median {pct(stats['median'])}. "
        f"{stats['hit']:.0%} finished positive. "
        f"t {stats['t']:+.1f} across {int(stats['n'])} events in {int(stats['n_tickers'])} names."
    )


def main() -> None:
    st.set_page_config(page_title="Upgrade and the next week", page_icon=":material/show_chart:", layout="wide")
    try:
        events, summary, regression, case, tape, meta = cached_study()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    prices, actions, _meta = load_warehouse()
    rivn = case.get("rivn") or {}
    crwd = case.get("crwd") or {}
    quiet_day = cell(summary, "quiet", "upgrade", "day0")
    quiet_pre = cell(summary, "quiet", "upgrade", "pre5")
    quiet_post = cell(summary, "quiet", "upgrade", "post5")
    quiet_month = cell(summary, "quiet", "upgrade", "post21")
    down_day = cell(summary, "quiet", "downgrade", "day0")
    down_post = cell(summary, "quiet", "downgrade", "post5")
    session_gap = coef(regression, "quiet", "session", "Upgrade")
    week_gap = coef(regression, "quiet", "next5", "Upgrade")
    clustered_pre = _summarize(events[(events["direction"] == "upgrade") & ~events["isolated"]], "car_pre5")

    st.title("The upgrade and the next week")
    st.markdown(
        "Schwab's market update on **3 Nov 2026** put two upgrades in one breath: "
        "Rivian climbed 8% on an upgrade to Buy from UBS, and CrowdStrike fell 3.5% "
        "despite an upgrade to Buy from Deutsche Bank. "
        f"[Listen to the update]({SCHWAB_UPDATE})."
    )
    st.markdown(
        "The question is the path **after** the rating. A cause would still be visible "
        "once the session is over, on days with no earnings print nearby and no second note "
        "in the same week, and against a quiet session in the same stock."
    )

    st.subheader("Monday, 2 Nov")
    left, right = st.columns(2)
    rivn_index = _indexed(prices, "RIVN", CASE_START, CASE_END)
    crwd_index = _indexed(prices, "CRWD", CASE_START, CASE_END)
    earn_day = tape.get("rivn_earnings_date")
    with left:
        marks = []
        if earn_day is not None:
            marks.append((pd.Timestamp(earn_day), "Q3", TEAL, 0.96, "right"))
        marks.append((pd.Timestamp("2026-10-30"), "Jefferies", AMBER, 0.62, "right"))
        marks.append((pd.Timestamp("2026-11-02"), "UBS", RED, 0.30, "right"))
        st.plotly_chart(
            _price_chart(rivn_index, "RIVN", AMBER, "Rivian and SPY, indexed to 100", marks),
            width="stretch",
        )
    with right:
        crwd_marks = [
            (pd.Timestamp("2026-08-26"), "Earnings", TEAL, 0.96, "left"),
            (pd.Timestamp("2026-11-02"), "Deutsche Bank", RED, 0.55, "right"),
        ]
        st.plotly_chart(
            _price_chart(crwd_index, "CRWD", BLUE, "CrowdStrike and SPY, indexed to 100", crwd_marks),
            width="stretch",
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rivian session", pct(tape["rivn_mon"]))
    m2.metric("Rivian abnormal", pct(rivn.get("ar0", np.nan)))
    m3.metric("CrowdStrike session", pct(tape["crwd_mon"]))
    m4.metric("CrowdStrike abnormal", pct(crwd.get("ar0", np.nan)))
    st.caption(
        f"Close to close, with SPY {pct(tape['spy_mon'])} that session. "
        f"Schwab's figures were 8% and 3.5%. "
        f"Rivian Q3 {tape['rivn_earnings_period']} printed on "
        f"{pd.Timestamp(earn_day).date() if earn_day is not None else 'n/a'}, "
        f"and the stock finished {pct(tape['rivn_earnings_raw'])}. "
        f"Jefferies upgraded Rivian the next session "
        f"({pct((case.get('rivn_prior') or {}).get('raw', np.nan))} close to close, "
        f"abnormal {pct((case.get('rivn_prior') or {}).get('ar0', np.nan))}). "
        "Those two upgrades are one session apart, and earnings sits inside two sessions of both, "
        "so Monday is not a quiet-tape event. "
        f"By Friday 6 Nov Rivian closed at ${tape['rivn_close_fri']:.2f}, "
        f"{pct(tape['rivn_fri_from_mon'])} from Monday's close. "
        "The five-session window after Monday is not complete yet."
    )
    if rivn:
        st.caption(
            f"Rivian's market-model beta into that session was {rivn['beta']:.2f}. "
            "The abnormal return is the session return minus that beta times SPY, "
            "with the fit taken from sessions 120 through 21 before the rating."
        )

    st.subheader("The other upgrades")
    st.markdown(
        f"A **quiet tape** is an upgrade with no other rating within five sessions and no earnings "
        f"within two. That sample is **{int(quiet_day['n'])} upgrades in {int(quiet_day['n_tickers'])} names**, "
        f"from the Finviz stamps on {meta.get('n_tickers', 50)} liquid stocks."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Five sessions before", pct(quiet_pre["mean"]))
    c2.metric("Upgrade session", pct(quiet_day["mean"]))
    c3.metric("Next five sessions", pct(quiet_post["mean"]))
    st.caption(
        "Five sessions before — " + _metric_caption(quiet_pre)
        + " Upgrade session — " + _metric_caption(quiet_day)
        + " Next five sessions — " + _metric_caption(quiet_post)
        + " Next 21 sessions — " + _metric_caption(quiet_month)
    )
    if session_gap is not None and week_gap is not None:
        st.markdown(
            f"Set beside a quiet session in the **same stock**, and holding the prior five sessions fixed, "
            f"the upgrade gap is **{pct(session_gap['coef'])}** on the rating day "
            f"(t {session_gap['t']:+.1f}) and **{pct(week_gap['coef'])}** over the next five sessions "
            f"(t {week_gap['t']:+.1f})."
        )
    st.markdown(
        f"When another rating lands in the same week, the five sessions before the upgrade "
        f"are already **{pct(clustered_pre['mean'])}** abnormal "
        f"(t {clustered_pre['t']:+.1f}, {int(clustered_pre['n'])} events). "
        f"Downgrade sessions on a quiet tape average **{pct(down_day['mean'])}** "
        f"(t {down_day['t']:+.1f}). The next five sessions average **{pct(down_post['mean'])}** "
        f"(t {down_post['t']:+.1f})."
    )

    st.plotly_chart(_panel_bars(summary), width="stretch")
    st.caption(
        "Bars are the equal-weighted mean abnormal return. "
        "Whiskers are a 95% range from a ticker-clustered standard error. "
        "A whisker through zero is a mean that sits inside a flat range. "
        "Matched quiet days are a session about three weeks earlier in the same name, "
        "with no rating within five sessions and no earnings within two."
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        earn_day_stats = _summarize(events[(events["direction"] == "upgrade") & events["earnings_adjacent"]], "ar0")
        earn_post = _summarize(events[(events["direction"] == "upgrade") & events["earnings_adjacent"]], "car_post5")
        st.plotly_chart(_earnings_bars(events), width="stretch")
        st.caption(
            f"Earnings weeks: session mean {pct(earn_day_stats['mean'])}, "
            f"median {pct(earn_day_stats['median'])}, t {earn_day_stats['t']:+.1f}. "
            f"Quiet tape: session mean {pct(quiet_day['mean'])}, median {pct(quiet_day['median'])}. "
            f"The next five sessions are {pct(earn_post['mean'])} after earnings weeks "
            f"(t {earn_post['t']:+.1f}) and {pct(quiet_post['mean'])} on the quiet tape."
        )
    with chart_right:
        st.plotly_chart(_histogram(events, float(rivn.get("ar0", np.nan))), width="stretch")
        st.caption(
            "The blue line is Rivian's 2 Nov abnormal return. "
            "That day is outside the quiet-tape sample, because Jefferies had upgraded it the session before "
            "and the Q3 print was two sessions before."
        )

    st.subheader("Look up a name")
    names = sorted(events["ticker"].unique())
    default = names.index("RIVN") if "RIVN" in names else 0
    ticker = st.selectbox(
        "Ticker",
        names,
        index=default,
        format_func=lambda symbol: f"{symbol} — {UNIVERSE.get(symbol, symbol)}",
    )
    st.plotly_chart(_lookup_chart(prices, events, actions, ticker), width="stretch")
    st.caption("Triangles are rating days. Teal lines are earnings stamps. The index starts at 100 on the first session of 2025.")

    table = _event_table(events)
    scope = st.radio("Table", ("This name", "All names"), horizontal=True)
    direction = st.radio("Direction", ("Upgrades", "Downgrades", "Both"), horizontal=True)
    view = table if scope == "All names" else table[table["Ticker"] == ticker]
    if direction == "Upgrades":
        view = view[view["Direction"] == "upgrade"]
    elif direction == "Downgrades":
        view = view[view["Direction"] == "downgrade"]
    st.dataframe(view, hide_index=True, use_container_width=True)

    with st.expander("How the abnormal return is built"):
        st.markdown(
            "Each rating day is one row, even when several firms move on the same session. "
            "Mixed days, with an upgrade and a downgrade together, stay out of the means. "
            "Initiations and target changes stay out too. Only Finviz's Upgrade and Downgrade stamps count."
        )
        st.markdown("Session return:")
        st.latex(r"R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1")
        st.markdown(
            "Alpha and beta come from an ordinary regression of the stock on SPY "
            "over sessions −120 through −21. The rating window is left out of the fit."
        )
        st.latex(r"AR_{i,t} = R_{i,t} - \hat\alpha_i - \hat\beta_i R_{m,t}")
        st.latex(r"CAR_i(t_1,t_2) = \sum_{t=t_1}^{t_2} AR_{i,t}")
        st.markdown(
            "The next-week test uses $CAR_i(1,5)$. A window that is not fully printed yet, "
            "including Rivian's week after 2 Nov, is left out of that mean."
        )
        st.markdown("The controlled comparison stacks each upgrade with its matched quiet day:")
        st.latex(
            r"CAR_i(1,5) = a + b\,\mathrm{Upgrade}_i + c\,\mathrm{Earnings}_i + d\,CAR_i(-5,-1) + e_i"
        )
        st.markdown(
            "$b$ is the extra abnormal return lined up with the upgrade after the prior week "
            "and an earnings flag. On the quiet tape the earnings flag does not vary, so it drops out. "
            "Standard errors are clustered by ticker. "
            "Finviz's chart stamps are the ratings it attaches to the price chart, not a full broker file. "
            "The universe is today's liquid names, so the history is not a point-in-time index."
        )
        st.caption(
            f"Warehouse {meta.get('price_start')} to {meta.get('price_end')}. "
            f"{meta.get('n_rating_rows')} rating rows, {meta.get('n_earnings_rows')} earnings stamps. "
            f"{meta.get('source')}."
        )

    st.caption(
        "Educational study of a public market update. Full credit to Charles Schwab, early market update, 3 Nov 2026. "
        "Rating and earnings dates from Finviz. Closes from Nasdaq. The market is SPY."
    )


if __name__ == "__main__":
    main()
