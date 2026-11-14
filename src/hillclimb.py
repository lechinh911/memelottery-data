#!/usr/bin/env python3
"""
Iterative research loop:
  diagnose losses → propose filters → backtest → keep wins → repeat
Objective: raise winrate + expectancy without collapsing trade count.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_symbol
from src.optimize import score_result, walk_forward
from src.strategy import StrategyParams, run_backtest

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FOCUS = ["BTC-USD|1h", "ETH-USD|1h", "GC=F|1h", "EURUSD=X|1h"]


def load_focus() -> dict[str, pd.DataFrame]:
    out = {}
    mapping = {
        "BTC-USD|1h": ("BTC-USD", "1h", "730d"),
        "ETH-USD|1h": ("ETH-USD", "1h", "730d"),
        "GC=F|1h": ("GC=F", "1h", "730d"),
        "EURUSD=X|1h": ("EURUSD=X", "1h", "730d"),
    }
    for k, args in mapping.items():
        out[k] = load_symbol(*args)
    return out


def quality_score(metrics: dict[str, float]) -> float:
    """Prefer high WR + positive expectancy with enough trades."""
    trades = metrics.get("trades", 0)
    if trades < 10:
        return -999.0
    wr = metrics.get("win_rate", 0.0)
    exp_r = metrics.get("expectancy_r", 0.0)
    pf = min(metrics.get("profit_factor", 0.0), 4.0)
    dd = abs(metrics.get("max_dd_pct", 0.0))
    # Heavily weight winrate while still requiring edge
    return (2.5 * wr + 1.8 * exp_r + 0.4 * pf) * (trades ** 0.35) - 2.0 * dd


def eval_config(name: str, params: StrategyParams, datasets: dict[str, pd.DataFrame]) -> dict:
    rows = []
    oos_scores, full_wrs, oos_wrs = [], [], []
    for ds, df in datasets.items():
        full = run_backtest(df, params)
        wf = walk_forward(df, params, 0.6)
        fm, tm = full.metrics, wf["test"]
        rows.append({"config": name, "dataset": ds, "scope": "full", **fm, "qscore": quality_score(fm)})
        rows.append({"config": name, "dataset": ds, "scope": "oos", **tm, "qscore": quality_score(tm)})
        oos_scores.append(quality_score(tm))
        full_wrs.append(fm["win_rate"])
        oos_wrs.append(tm["win_rate"] if tm["trades"] >= 5 else np.nan)
    return {
        "name": name,
        "params": params,
        "rows": rows,
        "mean_oos_q": float(np.mean(oos_scores)),
        "median_oos_wr": float(np.nanmedian(oos_wrs)),
        "median_full_wr": float(np.median(full_wrs)),
        "min_oos_q": float(np.min(oos_scores)),
    }


def diagnose(params: StrategyParams, datasets: dict[str, pd.DataFrame]) -> dict:
    """Analyze what separates wins vs losses."""
    all_trades = []
    for ds, df in datasets.items():
        res = run_backtest(df, params)
        for t in res.trades:
            all_trades.append(
                {
                    "dataset": ds,
                    "side": t.side,
                    "win": t.pnl_r > 0,
                    "pnl_r": t.pnl_r,
                    "reason": t.reason,
                    "quality": t.quality,
                    "adx": t.adx_at_entry,
                    "spread_atr": t.spread_atr_at_entry,
                    "extension": t.extension_atr,
                    "bars_after_break": t.bars_after_break,
                    "stop_atr": t.stop_atr,
                }
            )
    td = pd.DataFrame(all_trades)
    if td.empty:
        return {"n": 0}
    summary = {
        "n": int(len(td)),
        "wr": float(td["win"].mean()),
        "by_reason": td.groupby("reason")["win"].agg(["count", "mean"]).to_dict(),
        "by_side": td.groupby("side")["win"].agg(["count", "mean"]).to_dict(),
        "win_vs_loss_means": {
            "quality": {
                "win": float(td.loc[td.win, "quality"].mean()),
                "loss": float(td.loc[~td.win, "quality"].mean()),
            },
            "adx": {
                "win": float(td.loc[td.win, "adx"].mean()),
                "loss": float(td.loc[~td.win, "adx"].mean()),
            },
            "extension": {
                "win": float(td.loc[td.win, "extension"].mean()),
                "loss": float(td.loc[~td.win, "extension"].mean()),
            },
            "bars_after_break": {
                "win": float(td.loc[td.win, "bars_after_break"].mean()),
                "loss": float(td.loc[~td.win, "bars_after_break"].mean()),
            },
            "stop_atr": {
                "win": float(td.loc[td.win, "stop_atr"].mean()),
                "loss": float(td.loc[~td.win, "stop_atr"].mean()),
            },
            "spread_atr": {
                "win": float(td.loc[td.win, "spread_atr"].mean()),
                "loss": float(td.loc[~td.win, "spread_atr"].mean()),
            },
        },
    }
    # Threshold probes
    probes = {}
    for col, qs in [
        ("quality", [0.4, 0.5, 0.55, 0.6]),
        ("adx", [18, 22, 25, 28]),
        ("extension", [1.0, 1.5, 2.0, 2.5]),
        ("bars_after_break", [3, 5, 8, 12]),
        ("stop_atr", [2.0, 2.5, 3.0, 3.5]),
    ]:
        best = None
        for thr in qs:
            if col in ("extension", "bars_after_break", "stop_atr"):
                sub = td[td[col] <= thr]
            else:
                sub = td[td[col] >= thr]
            if len(sub) < 20:
                continue
            wr = float(sub["win"].mean())
            exp = float(sub["pnl_r"].mean())
            score = wr * 2 + exp
            if best is None or score > best["score"]:
                best = {"thr": thr, "n": len(sub), "wr": wr, "exp": exp, "score": score}
        probes[col] = best
    summary["threshold_probes"] = probes
    td.to_csv(RESULTS / "trade_diagnosis.csv", index=False)
    return summary


def base_configs() -> dict[str, StrategyParams]:
    strict = StrategyParams(
        min_retrace=0.60,
        min_height_atr=1.8,
        break_buf=0.25,
        cancel_retest=True,
        use_volume=True,
        vol_mult=1.2,
        use_htf_bias=True,
        opp_cancel=True,
        min_quality=0.50,
        spread_atr=0.35,
        min_score=3,
        adx_min=22,
        rr_target=1.5,
        use_trail_after_1r=True,
        trail_atr=1.3,
        cooldown=10,
    )
    core = StrategyParams(
        min_retrace=0.55,
        min_height_atr=1.6,
        break_buf=0.15,
        cancel_retest=False,
        use_volume=True,
        vol_mult=1.15,
        use_htf_bias=True,
        opp_cancel=True,
        min_quality=0.40,
        spread_atr=0.30,
        min_score=2,
        adx_min=20,
        rr_target=1.5,
        use_trail_after_1r=True,
        trail_atr=1.3,
        cooldown=8,
    )
    return {"v2_strict": strict, "v2_core": core}


def mutate_from_diagnosis(base: StrategyParams, diag: dict, round_id: int) -> list[tuple[str, StrategyParams]]:
    """Generate candidates from diagnosis probes + known good levers."""
    cands: list[tuple[str, StrategyParams]] = []
    probes = diag.get("threshold_probes", {})

    def add(name: str, **kwargs):
        cands.append((name, replace(base, **kwargs)))

    # Round-agnostic quality levers
    add("q55", min_quality=0.55)
    add("q60", min_quality=0.60)
    add("adx25", adx_min=25)
    add("adx28", adx_min=28)
    add("score3_spread40", min_score=3, spread_atr=0.40)
    add("retrace65", min_retrace=0.65)
    add("height20", min_height_atr=2.0)
    add("break35", break_buf=0.35)
    add("vol13", vol_mult=1.3)
    add("ext15", max_extension_atr=1.5)
    add("ext20", max_extension_atr=2.0)
    add("ext25", max_extension_atr=2.5)
    add("stop_max30", max_stop_atr=3.0)
    add("stop_max25", max_stop_atr=2.5)
    add("stop_min08", min_stop_atr=0.8)
    add("breakwin5", break_cross_win=5)
    add("breakwin8", break_cross_win=8)
    add("breakwin12", break_cross_win=12)
    add("close_dir", require_close_dir=True)
    add("rsi", use_rsi=True, rsi_long_max=65, rsi_short_min=35)
    add("rsi_tight", use_rsi=True, rsi_long_max=60, rsi_short_min=40)
    add("tp1_be", tp1_r=1.0, rr_target=2.0, use_trail_after_1r=True)
    add("rr12_trail", rr_target=1.2, use_trail_after_1r=True, trail_atr=1.0)
    add("rr15_notrail", rr_target=1.5, use_trail_after_1r=False)
    add("cancel_off", cancel_retest=False)
    add("cancel_on", cancel_retest=True)

    # Combine strongest probes
    if probes.get("quality"):
        add("probe_q", min_quality=float(probes["quality"]["thr"]))
    if probes.get("adx"):
        add("probe_adx", adx_min=float(probes["adx"]["thr"]))
    if probes.get("extension"):
        add("probe_ext", max_extension_atr=float(probes["extension"]["thr"]))
    if probes.get("bars_after_break"):
        add("probe_win", break_cross_win=int(probes["bars_after_break"]["thr"]))
    if probes.get("stop_atr"):
        add("probe_stop", max_stop_atr=float(probes["stop_atr"]["thr"]))

    if round_id >= 2:
        # Stacked combos for winrate
        add(
            "stack_wr1",
            min_quality=0.55,
            adx_min=25,
            max_extension_atr=2.0,
            break_cross_win=8,
            require_close_dir=True,
            min_score=3,
            spread_atr=0.35,
            cancel_retest=True,
            rr_target=1.5,
            tp1_r=1.0,
        )
        add(
            "stack_wr2",
            min_quality=0.55,
            adx_min=25,
            max_extension_atr=1.5,
            max_stop_atr=3.0,
            break_cross_win=5,
            use_rsi=True,
            rsi_long_max=65,
            rsi_short_min=35,
            require_close_dir=True,
            min_score=3,
            vol_mult=1.25,
            rr_target=1.5,
            use_trail_after_1r=True,
        )
        add(
            "stack_wr3",
            min_quality=0.60,
            adx_min=28,
            max_extension_atr=2.0,
            break_cross_win=8,
            min_height_atr=2.0,
            min_retrace=0.65,
            break_buf=0.30,
            min_score=3,
            spread_atr=0.40,
            cancel_retest=True,
            rr_target=1.5,
            tp1_r=1.0,
            trail_atr=1.2,
        )
        add(
            "stack_gold_bias",
            min_quality=0.50,
            adx_min=22,
            max_extension_atr=2.0,
            break_cross_win=10,
            require_close_dir=True,
            use_rsi=True,
            rsi_long_max=70,
            rsi_short_min=30,
            cancel_retest=False,
            rr_target=2.0,
            tp1_r=1.0,
        )

    if round_id >= 3:
        add(
            "final_hq",
            min_quality=0.58,
            adx_min=26,
            max_extension_atr=1.8,
            max_stop_atr=3.2,
            min_stop_atr=0.7,
            break_cross_win=6,
            require_close_dir=True,
            use_rsi=True,
            rsi_long_max=62,
            rsi_short_min=38,
            min_score=3,
            spread_atr=0.38,
            vol_mult=1.25,
            min_height_atr=1.9,
            min_retrace=0.62,
            break_buf=0.28,
            cancel_retest=True,
            opp_cancel=True,
            use_volume=True,
            use_htf_bias=True,
            rr_target=1.5,
            tp1_r=1.0,
            use_trail_after_1r=True,
            trail_atr=1.2,
            cooldown=10,
        )
        add(
            "final_balanced",
            min_quality=0.52,
            adx_min=24,
            max_extension_atr=2.2,
            max_stop_atr=3.5,
            break_cross_win=8,
            require_close_dir=True,
            use_rsi=True,
            rsi_long_max=65,
            rsi_short_min=35,
            min_score=3,
            spread_atr=0.35,
            vol_mult=1.2,
            min_height_atr=1.8,
            min_retrace=0.60,
            break_buf=0.25,
            cancel_retest=True,
            rr_target=1.5,
            tp1_r=1.0,
            use_trail_after_1r=True,
            trail_atr=1.25,
            cooldown=8,
        )

    # Deduplicate by param dict
    seen = set()
    uniq = []
    for name, params in cands:
        key = json.dumps(asdict(params), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, params))
    return uniq


def main() -> None:
    datasets = load_focus()
    history = []
    all_rows = []

    bases = base_configs()
    # Start from best prior (strict)
    champion_name = "v2_strict"
    champion = bases[champion_name]
    champ_eval = eval_config(champion_name, champion, datasets)
    history.append(
        {
            "round": 0,
            "name": champion_name,
            "mean_oos_q": champ_eval["mean_oos_q"],
            "median_oos_wr": champ_eval["median_oos_wr"],
            "min_oos_q": champ_eval["min_oos_q"],
        }
    )
    all_rows.extend(champ_eval["rows"])
    print(
        f"[R0] champion={champion_name} OOS_Q={champ_eval['mean_oos_q']:.3f} "
        f"OOS_WR={champ_eval['median_oos_wr']:.1%} minQ={champ_eval['min_oos_q']:.3f}",
        flush=True,
    )

    for rnd in range(1, 4):
        print(f"\n===== ROUND {rnd}: diagnose =====", flush=True)
        diag = diagnose(champion, datasets)
        print(
            f"trades={diag.get('n')} WR={diag.get('wr', 0):.1%} "
            f"probes={json.dumps(diag.get('threshold_probes', {}), default=str)[:500]}",
            flush=True,
        )
        with open(RESULTS / f"diagnosis_r{rnd}.json", "w") as f:
            json.dump(diag, f, indent=2, default=str)

        print(f"===== ROUND {rnd}: mutate + backtest =====", flush=True)
        cands = mutate_from_diagnosis(champion, diag, rnd)
        # Always include current champion and the other base
        cands = [(champion_name, champion)] + cands
        if rnd == 1:
            cands.append(("v2_core", bases["v2_core"]))

        best = champ_eval
        for name, params in cands:
            ev = eval_config(f"r{rnd}_{name}", params, datasets)
            all_rows.extend(ev["rows"])
            # Selection: better mean OOS quality, prefer higher WR, reject if any market catastrophic
            better = False
            if ev["mean_oos_q"] > best["mean_oos_q"] + 0.02:
                better = True
            elif abs(ev["mean_oos_q"] - best["mean_oos_q"]) <= 0.02 and ev["median_oos_wr"] > best["median_oos_wr"] + 0.02:
                better = True
            # Robustness: don't accept if min OOS collapses vs champion unless WR jump huge
            if better and ev["min_oos_q"] < best["min_oos_q"] - 1.0 and ev["median_oos_wr"] < best["median_oos_wr"] + 0.05:
                better = False
            if better:
                best = ev
                print(
                    f"  ↑ {ev['name']}: OOS_Q={ev['mean_oos_q']:.3f} OOS_WR={ev['median_oos_wr']:.1%} "
                    f"minQ={ev['min_oos_q']:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"  · {ev['name']}: OOS_Q={ev['mean_oos_q']:.3f} OOS_WR={ev['median_oos_wr']:.1%}",
                    flush=True,
                )

        champion = best["params"]
        champion_name = best["name"]
        champ_eval = best
        history.append(
            {
                "round": rnd,
                "name": champion_name,
                "mean_oos_q": champ_eval["mean_oos_q"],
                "median_oos_wr": champ_eval["median_oos_wr"],
                "min_oos_q": champ_eval["min_oos_q"],
                "params": asdict(champion),
            }
        )
        print(
            f"[R{rnd}] NEW champion={champion_name} OOS_Q={champ_eval['mean_oos_q']:.3f} "
            f"OOS_WR={champ_eval['median_oos_wr']:.1%}",
            flush=True,
        )

    # Final report vs v1 and v2_strict
    print("\n===== FINAL COMPARISON =====", flush=True)
    finals = {
        "v1_original": StrategyParams(),
        "v2_strict": bases["v2_strict"],
        "v3_champion": champion,
    }
    final_rows = []
    summary = []
    for name, params in finals.items():
        ev = eval_config(name, params, datasets)
        final_rows.extend(ev["rows"])
        summary.append(
            {
                "config": name,
                "mean_oos_q": ev["mean_oos_q"],
                "median_oos_wr": ev["median_oos_wr"],
                "median_full_wr": ev["median_full_wr"],
                "min_oos_q": ev["min_oos_q"],
            }
        )
        print(
            f"{name}: OOS_Q={ev['mean_oos_q']:.3f} OOS_WR={ev['median_oos_wr']:.1%} "
            f"FULL_WR={ev['median_full_wr']:.1%} minQ={ev['min_oos_q']:.3f}",
            flush=True,
        )
        # Per dataset OOS detail
        for r in ev["rows"]:
            if r["scope"] == "oos":
                print(
                    f"  {r['dataset']}: trades={r['trades']:.0f} WR={r['win_rate']:.1%} "
                    f"PF={r['profit_factor']:.2f} ExpR={r['expectancy_r']:.3f}",
                    flush=True,
                )

    pd.DataFrame(all_rows + final_rows).to_csv(RESULTS / "hillclimb_all.csv", index=False)
    pd.DataFrame(final_rows).to_csv(RESULTS / "hillclimb_final.csv", index=False)
    with open(RESULTS / "best_params.json", "w") as f:
        json.dump(
            {
                "winner": "v3_champion",
                "params": asdict(champion),
                "history": history,
                "summary": summary,
            },
            f,
            indent=2,
        )
    with open(RESULTS / "hillclimb_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone. Champion params saved to results/best_params.json", flush=True)


if __name__ == "__main__":
    main()
