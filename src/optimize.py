"""Parameter search and variant comparison."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .strategy import StrategyParams, compute_metrics, run_backtest

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def score_result(metrics: dict[str, float]) -> float:
    """
    Primary objective: expectancy in R * sqrt(trades) with drawdown and PF penalties.
    Prefer robust edges over lucky few trades.
    """
    trades = metrics.get("trades", 0)
    if trades < 8:
        return -999.0
    exp_r = metrics.get("expectancy_r", 0.0)
    pf = min(metrics.get("profit_factor", 0.0), 5.0)
    dd = abs(metrics.get("max_dd_pct", 0.0))
    wr = metrics.get("win_rate", 0.0)
    # Soft constraints
    if pf < 0.9:
        return exp_r * (trades ** 0.5) - 5.0
    return exp_r * (trades ** 0.5) + 0.3 * pf + 0.2 * wr - 1.5 * dd


def baseline_variants() -> list[tuple[str, StrategyParams]]:
    """Named research variants to compare against original defaults."""
    base = StrategyParams()
    return [
        ("v1_original", base),
        (
            "v2_volume",
            replace(base, use_volume=True, vol_mult=1.2, cancel_retest=True),
        ),
        (
            "v2_htf_bias",
            replace(base, use_htf_bias=True, htf_rule="4h", htf_ema=50),
        ),
        (
            "v2_retest",
            replace(base, retest_entry=True, cancel_retest=False, break_buf=0.15),
        ),
        (
            "v2_quality",
            replace(base, min_quality=0.45, min_height_atr=1.8, min_retrace=0.55),
        ),
        (
            "v2_momentum_strict",
            replace(base, spread_atr=0.4, min_score=3, adx_min=22, need_cross=False),
        ),
        (
            "v2_momentum_loose",
            replace(base, spread_atr=0.2, min_score=1, adx_min=15, slope_atr=0.15),
        ),
        (
            "v2_rr15_no_trail",
            replace(base, rr_target=1.5, use_trail_after_1r=False),
        ),
        (
            "v2_rr25_trail",
            replace(base, rr_target=2.5, use_trail_after_1r=True, trail_atr=1.2),
        ),
        (
            "v2_combo_best_guess",
            replace(
                base,
                use_volume=True,
                vol_mult=1.15,
                use_htf_bias=True,
                cancel_retest=False,
                retest_entry=False,
                min_quality=0.4,
                min_height_atr=1.6,
                spread_atr=0.35,
                min_score=2,
                adx_min=18,
                rr_target=2.0,
                use_trail_after_1r=True,
                cooldown=8,
                opp_cancel=True,
            ),
        ),
        (
            "v2_combo_retest_vol",
            replace(
                base,
                use_volume=True,
                vol_mult=1.1,
                use_htf_bias=True,
                retest_entry=True,
                cancel_retest=False,
                min_quality=0.4,
                break_buf=0.15,
                rr_target=2.0,
                cooldown=5,
                opp_cancel=True,
            ),
        ),
    ]


def grid_search(df: pd.DataFrame, max_combos: int = 120) -> pd.DataFrame:
    """Coarse grid around the most sensitive knobs."""
    grid = {
        "min_retrace": [0.45, 0.55, 0.65],
        "min_height_atr": [1.2, 1.6, 2.0],
        "break_buf": [0.1, 0.2, 0.35],
        "spread_atr": [0.2, 0.3, 0.45],
        "min_score": [1, 2, 3],
        "adx_min": [15.0, 20.0, 25.0],
        "rr_target": [1.5, 2.0, 2.5],
        "cancel_retest": [True, False],
        "use_volume": [False, True],
        "use_htf_bias": [False, True],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    # Subsample deterministically for speed
    step = max(1, len(combos) // max_combos)
    combos = combos[::step][:max_combos]

    rows = []
    for values in combos:
        kwargs = dict(zip(keys, values))
        params = StrategyParams(**kwargs)
        # Keep research extras sensible with volume
        if kwargs["use_volume"]:
            params = replace(params, vol_mult=1.2)
        res = run_backtest(df, params)
        m = res.metrics
        rows.append(
            {
                **kwargs,
                **m,
                "score": score_result(m),
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out


def walk_forward(
    df: pd.DataFrame,
    params: StrategyParams,
    train_frac: float = 0.6,
) -> dict[str, Any]:
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split]
    test = df.iloc[split:]
    tr = run_backtest(train, params).metrics
    te = run_backtest(test, params).metrics
    return {"train": tr, "test": te, "split": split, "n": n}


def compare_variants(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, params in baseline_variants():
        for ds_name, df in datasets.items():
            res = run_backtest(df, params)
            m = res.metrics
            rows.append(
                {
                    "variant": name,
                    "dataset": ds_name,
                    **m,
                    "score": score_result(m),
                }
            )
    return pd.DataFrame(rows)
