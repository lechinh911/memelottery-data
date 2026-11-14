#!/usr/bin/env python3
"""Run full research pipeline: download → variants → grid → walk-forward → report."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_UNIVERSE, load_symbol
from src.optimize import (
    RESULTS,
    compare_variants,
    grid_search,
    score_result,
    walk_forward,
)
from src.strategy import StrategyParams, run_backtest


def load_universe(limit: int | None = None) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    universe = DEFAULT_UNIVERSE if limit is None else DEFAULT_UNIVERSE[:limit]
    for symbol, interval, period in universe:
        key = f"{symbol}|{interval}"
        try:
            print(f"loading {key} ({period}) ...", flush=True)
            df = load_symbol(symbol, interval=interval, period=period)
            print(f"  -> {len(df)} bars", flush=True)
            datasets[key] = df
        except Exception as e:
            print(f"  FAIL {key}: {e}", flush=True)
    return datasets


def aggregate_variant_scores(cmp: pd.DataFrame) -> pd.DataFrame:
    """Average score across datasets; require enough total trades."""
    g = (
        cmp.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            median_pf=("profit_factor", "median"),
            median_wr=("win_rate", "median"),
            median_exp_r=("expectancy_r", "median"),
            median_dd=("max_dd_pct", "median"),
            total_trades=("trades", "sum"),
            datasets=("dataset", "nunique"),
        )
        .sort_values("mean_score", ascending=False)
    )
    return g


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    datasets = load_universe()
    if not datasets:
        raise SystemExit("No datasets loaded")

    print("\n=== Variant comparison ===", flush=True)
    cmp = compare_variants(datasets)
    cmp.to_csv(RESULTS / "variant_comparison.csv", index=False)
    agg = aggregate_variant_scores(cmp)
    agg.to_csv(RESULTS / "variant_aggregate.csv", index=False)
    print(agg.to_string(index=False), flush=True)

    # Grid search on primary liquid crypto 1h + daily BTC
    primary_keys = [k for k in ("BTC-USD|1h", "BTC-USD|1d", "ETH-USD|1h") if k in datasets]
    grid_frames = []
    best_params_by_ds: dict[str, StrategyParams] = {}
    for key in primary_keys:
        print(f"\n=== Grid search on {key} ===", flush=True)
        grid = grid_search(datasets[key], max_combos=100)
        grid.to_csv(RESULTS / f"grid_{key.replace('|', '_').replace('=', '')}.csv", index=False)
        grid_frames.append(grid.assign(dataset=key))
        top = grid.iloc[0].to_dict()
        print(
            f"top score={top['score']:.3f} trades={top['trades']:.0f} "
            f"WR={top['win_rate']:.1%} PF={top['profit_factor']:.2f} ExpR={top['expectancy_r']:.3f}",
            flush=True,
        )
        best_params_by_ds[key] = StrategyParams(
            min_retrace=float(top["min_retrace"]),
            min_height_atr=float(top["min_height_atr"]),
            break_buf=float(top["break_buf"]),
            spread_atr=float(top["spread_atr"]),
            min_score=int(top["min_score"]),
            adx_min=float(top["adx_min"]),
            rr_target=float(top["rr_target"]),
            cancel_retest=bool(top["cancel_retest"]),
            use_volume=bool(top["use_volume"]),
            use_htf_bias=bool(top["use_htf_bias"]),
            vol_mult=1.2 if top["use_volume"] else 1.0,
            opp_cancel=True,
            cooldown=5,
            use_trail_after_1r=True,
        )

    if grid_frames:
        all_grid = pd.concat(grid_frames, ignore_index=True)
        # Consensus: average rank of param hashes across datasets is hard;
        # instead pick params that appear most in top-10 of each dataset by score.
        tops = []
        for key in primary_keys:
            g = all_grid[all_grid["dataset"] == key].nlargest(15, "score")
            tops.append(g)
        consensus = pd.concat(tops, ignore_index=True)
        param_cols = [
            "min_retrace",
            "min_height_atr",
            "break_buf",
            "spread_atr",
            "min_score",
            "adx_min",
            "rr_target",
            "cancel_retest",
            "use_volume",
            "use_htf_bias",
        ]
        # Mode of each column among top rows
        consensus_params = {}
        for c in param_cols:
            consensus_params[c] = consensus[c].mode().iloc[0]
        print("\n=== Consensus top params ===", flush=True)
        print(json.dumps(consensus_params, indent=2, default=str), flush=True)

        best = StrategyParams(
            min_retrace=float(consensus_params["min_retrace"]),
            min_height_atr=float(consensus_params["min_height_atr"]),
            break_buf=float(consensus_params["break_buf"]),
            spread_atr=float(consensus_params["spread_atr"]),
            min_score=int(consensus_params["min_score"]),
            adx_min=float(consensus_params["adx_min"]),
            rr_target=float(consensus_params["rr_target"]),
            cancel_retest=bool(consensus_params["cancel_retest"]),
            use_volume=bool(consensus_params["use_volume"]),
            use_htf_bias=bool(consensus_params["use_htf_bias"]),
            vol_mult=1.2 if consensus_params["use_volume"] else 1.0,
            opp_cancel=True,
            cooldown=8,
            min_quality=0.35,
            use_trail_after_1r=True,
            trail_atr=1.3,
        )
    else:
        best = StrategyParams(use_volume=True, use_htf_bias=True, opp_cancel=True)

    # Also evaluate best named variant from aggregate
    best_variant_name = agg.iloc[0]["variant"]
    from src.optimize import baseline_variants

    named = dict(baseline_variants())[best_variant_name]

    print(f"\n=== Walk-forward: consensus grid + best variant ({best_variant_name}) ===", flush=True)
    wf_rows = []
    final_candidates = {
        "consensus_grid": best,
        best_variant_name: named,
        "v1_original": StrategyParams(),
    }
    # Blend: take named best and apply consensus numeric knobs lightly
    blended = replace(
        named,
        min_retrace=best.min_retrace,
        min_height_atr=best.min_height_atr,
        break_buf=best.break_buf,
        spread_atr=best.spread_atr,
        min_score=best.min_score,
        adx_min=best.adx_min,
        rr_target=best.rr_target,
        cancel_retest=best.cancel_retest,
    )
    final_candidates["blended"] = blended

    summary_oos = []
    for cname, params in final_candidates.items():
        oos_scores = []
        for ds_name, df in datasets.items():
            wf = walk_forward(df, params, train_frac=0.6)
            wf_rows.append(
                {
                    "candidate": cname,
                    "dataset": ds_name,
                    **{f"train_{k}": v for k, v in wf["train"].items()},
                    **{f"test_{k}": v for k, v in wf["test"].items()},
                    "test_score": score_result(wf["test"]),
                }
            )
            oos_scores.append(score_result(wf["test"]))
            full = run_backtest(df, params).metrics
            summary_oos.append({"candidate": cname, "dataset": ds_name, "scope": "full", **full})
        print(
            f"{cname}: mean OOS score={sum(oos_scores)/len(oos_scores):.3f}",
            flush=True,
        )

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(RESULTS / "walk_forward.csv", index=False)
    pd.DataFrame(summary_oos).to_csv(RESULTS / "full_sample.csv", index=False)

    oos_agg = (
        wf_df.groupby("candidate", as_index=False)
        .agg(
            mean_test_score=("test_score", "mean"),
            median_test_pf=("test_profit_factor", "median"),
            median_test_wr=("test_win_rate", "median"),
            median_test_exp=("test_expectancy_r", "median"),
            median_test_dd=("test_max_dd_pct", "median"),
            total_test_trades=("test_trades", "sum"),
        )
        .sort_values("mean_test_score", ascending=False)
    )
    oos_agg.to_csv(RESULTS / "oos_aggregate.csv", index=False)
    print("\n=== OOS aggregate ===", flush=True)
    print(oos_agg.to_string(index=False), flush=True)

    winner_name = oos_agg.iloc[0]["candidate"]
    winner = final_candidates[winner_name]

    # Persist winner params
    from dataclasses import asdict

    with open(RESULTS / "best_params.json", "w") as f:
        json.dump({"winner": winner_name, "params": asdict(winner)}, f, indent=2)

    # Per-dataset full metrics for winner
    winner_rows = []
    for ds_name, df in datasets.items():
        m = run_backtest(df, winner).metrics
        winner_rows.append({"dataset": ds_name, **m})
    pd.DataFrame(winner_rows).to_csv(RESULTS / "winner_full.csv", index=False)

    elapsed = time.time() - t0
    report = {
        "elapsed_sec": elapsed,
        "datasets": list(datasets.keys()),
        "best_variant_isample": best_variant_name,
        "oos_winner": winner_name,
        "oos_table": oos_agg.to_dict(orient="records"),
        "winner_params": asdict(winner),
        "winner_full": winner_rows,
    }
    with open(RESULTS / "research_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {elapsed:.1f}s. Winner={winner_name}", flush=True)
    print(f"Artifacts in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
