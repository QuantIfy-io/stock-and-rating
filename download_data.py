"""Seed the bundled rating, earnings, and price warehouse."""
from __future__ import annotations

import sys

from market_data import UNIVERSE, fetch_bundle, write_warehouse
from upgrade_engine import build_study, cell, coef, describe_mean, pct


def _print_case(study: dict) -> None:
    tape = study["case_tape"]
    print(
        f"2 Nov close-to-close  RIVN {pct(tape['rivn_mon'])}  "
        f"CRWD {pct(tape['crwd_mon'])}  SPY {pct(tape['spy_mon'])}"
    )
    print(
        f"RIVN closes  30 Oct {tape['rivn_close_prior']:.2f}  "
        f"2 Nov {tape['rivn_close_mon']:.2f}  6 Nov {tape['rivn_close_fri']:.2f}"
    )
    print(f"RIVN earnings {tape['rivn_earnings_date']} {tape['rivn_earnings_period']}")
    for label, row in study["case"].items():
        if row is None:
            print(label, "MISSING")
            continue
        print(
            f"{label} {row['date'].date()} {row['direction']} {row['firms']} "
            f"raw {pct(row['raw'])} AR {pct(row['ar0'])} "
            f"pre {pct(row['car_pre5'])} post5 {pct(row['car_post5'])} "
            f"beta {row['beta']:.2f} earnings {bool(row['earnings_adjacent'])} "
            f"isolated {bool(row['isolated'])}"
        )


def _print_summary(study: dict) -> None:
    summary = study["summary"]
    events = study["events"]
    print(
        f"Events {len(events)}  upgrades {(events.direction=='upgrade').sum()}  "
        f"downgrades {(events.direction=='downgrade').sum()}  "
        f"mixed {(events.direction=='mixed').sum()}"
    )
    for spec in ("all", "isolated", "quiet"):
        print(f"--- {spec}")
        for window in ("pre5", "day0", "post5", "post21"):
            for direction in ("upgrade", "downgrade", "placebo"):
                stats = cell(summary, spec, direction, window)
                print(f"  {direction:10} {window:6} {describe_mean(stats)}  median {pct(stats['median'])}  hit {stats['hit']:.0%}" if stats["n"] else f"  {direction:10} {window:6} empty")
    regression = study["regression"]
    print("--- regression, upgrade coefficient")
    for spec in ("all", "isolated", "quiet"):
        for outcome in ("session", "next5", "next5_after_session"):
            row = coef(regression, spec, outcome, "Upgrade")
            if row is None:
                print(f"  {spec} {outcome}: none")
                continue
            print(
                f"  {spec} {outcome}: {pct(row['coef'])} t {row['t']:+.2f} "
                f"n {int(row['n'])} tickers {int(row['n_tickers'])}"
            )


def main() -> None:
    tickers = sys.argv[1:] or list(UNIVERSE)
    bundle = fetch_bundle(tickers)
    meta = write_warehouse(bundle)
    print(
        f"Prices {meta['price_start']} → {meta['price_end']} · "
        f"{meta['n_tickers']} tickers · {meta['n_rating_rows']} rating rows · "
        f"{meta['n_earnings_rows']} earnings"
    )
    study = build_study(bundle["prices"], bundle["actions"])
    _print_case(study)
    _print_summary(study)


if __name__ == "__main__":
    main()
