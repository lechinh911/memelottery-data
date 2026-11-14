"""
Port of Pine Script: 2 Đáy/2 Đỉnh (cụm nến) → Gãy → EMA20/50 Momentum.

Adds research features toggled via StrategyParams:
- volume confirmation on breakout
- higher-timeframe EMA bias (resampled)
- retest entry after breakout
- R-multiple exits (pattern height)
- cooldown between trades
- pattern quality score filter
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema, pivot_high, pivot_low, rsi, volume_sma


@dataclass
class StrategyParams:
    # EMA
    ema_fast: int = 20
    ema_slow: int = 50
    # Swing
    piv_left: int = 5
    piv_right: int = 5
    atr_len: int = 14
    # Cluster
    cluster_atr: float = 0.5
    min_cluster: int = 3
    spike_atr: float = 1.0
    # Double pattern
    min_retrace: float = 0.5
    tol_atr: float = 0.75
    min_height_atr: float = 1.5
    min_bars: int = 5
    max_gap: int = 100
    max_bars: int = 50
    opp_cancel: bool = False
    # Break
    break_buf: float = 0.2
    cancel_retest: bool = True
    cross_win: int = 0
    # Momentum
    need_cross: bool = False
    confirm_win: int = 0
    spread_atr: float = 0.3
    slope_len: int = 3
    slope_atr: float = 0.25
    min_score: int = 2
    adx_len: int = 14
    adx_min: float = 20.0
    # Risk / exits
    rr_target: float = 2.0
    use_trail_after_1r: bool = True
    trail_atr: float = 1.5
    # Research filters (v2)
    use_volume: bool = False
    vol_mult: float = 1.2
    vol_len: int = 20
    use_htf_bias: bool = False
    htf_rule: str = "4h"  # pandas resample rule; ignored on daily
    htf_ema: int = 50
    retest_entry: bool = False
    retest_buf_atr: float = 0.15
    cooldown: int = 5
    min_quality: float = 0.0  # 0-1 pattern quality
    long_only: bool = False
    short_only: bool = False
    # v3 quality / winrate filters
    max_extension_atr: float = 0.0  # skip if close already > N*ATR past neckline (0=off)
    max_stop_atr: float = 0.0  # skip if stop distance > N*ATR (0=off)
    min_stop_atr: float = 0.0  # skip if stop too tight (0=off)
    require_close_dir: bool = False  # entry bar must close in trade direction
    use_rsi: bool = False
    rsi_len: int = 14
    rsi_long_max: float = 65.0
    rsi_short_min: float = 35.0
    break_cross_win: int = 0  # max bars after break to allow entry (0=off); alias focus window
    tp1_r: float = 0.0  # if >0, move stop to BE after tp1_r and keep runner to rr_target
    atr_stop_cap: float = 0.0  # if >0, tighten pattern stop to at most N*ATR from entry
    # Costs
    fee_bps: float = 4.0  # 0.04% per side
    slip_bps: float = 2.0


@dataclass
class Trade:
    side: str
    entry_idx: int
    entry_price: float
    stop: float
    target: float
    stop_init: float = 0.0
    exit_idx: int | None = None
    exit_price: float | None = None
    reason: str | None = None
    pattern_height: float = 0.0
    quality: float = 0.0
    pnl_r: float = 0.0
    pnl_pct: float = 0.0
    adx_at_entry: float = 0.0
    spread_atr_at_entry: float = 0.0
    extension_atr: float = 0.0
    bars_after_break: int = 0
    stop_atr: float = 0.0


@dataclass
class BacktestResult:
    params: dict[str, Any]
    trades: list[Trade]
    equity: pd.Series
    metrics: dict[str, float]


def _cluster_high(
    high: np.ndarray,
    atr_arr: np.ndarray,
    i: int,
    ph: float,
    piv_left: int,
    piv_right: int,
    cluster_atr: float,
    min_cluster: int,
    spike_atr: float,
) -> tuple[bool, float, int, int]:
    """Validate high cluster at pivot center index (i - piv_right). Returns valid, bot, left_ext, right_ext."""
    center = i - piv_right
    tol = atr_arr[i - piv_right] * cluster_atr if not np.isnan(atr_arr[i - piv_right]) else np.nan
    if np.isnan(tol):
        return False, np.nan, 0, 0
    cnt = 1
    z_bot = ph
    li = ri = 0
    for offset in range(1, piv_left + 1):
        o = center - offset
        if o < 0:
            break
        if high[o] >= ph - tol:
            cnt += 1
            z_bot = min(z_bot, high[o])
            li += 1
        else:
            break
    for offset in range(1, piv_right + 1):
        o = center + offset
        if o >= len(high):
            break
        if high[o] >= ph - tol:
            cnt += 1
            z_bot = min(z_bot, high[o])
            ri += 1
        else:
            break
    nb_max = -np.inf
    for offset in range(1, piv_left + 1):
        o = center - offset
        if o >= 0:
            nb_max = max(nb_max, high[o])
    for offset in range(1, piv_right + 1):
        o = center + offset
        if o < len(high):
            nb_max = max(nb_max, high[o])
    is_spike = spike_atr > 0 and nb_max > -np.inf and (ph - nb_max) >= spike_atr * atr_arr[i - piv_right]
    valid = cnt >= min_cluster or is_spike
    return valid, z_bot, li, ri


def _cluster_low(
    low: np.ndarray,
    atr_arr: np.ndarray,
    i: int,
    pl: float,
    piv_left: int,
    piv_right: int,
    cluster_atr: float,
    min_cluster: int,
    spike_atr: float,
) -> tuple[bool, float, int, int]:
    center = i - piv_right
    tol = atr_arr[i - piv_right] * cluster_atr if not np.isnan(atr_arr[i - piv_right]) else np.nan
    if np.isnan(tol):
        return False, np.nan, 0, 0
    cnt = 1
    z_top = pl
    li = ri = 0
    for offset in range(1, piv_left + 1):
        o = center - offset
        if o < 0:
            break
        if low[o] <= pl + tol:
            cnt += 1
            z_top = max(z_top, low[o])
            li += 1
        else:
            break
    for offset in range(1, piv_right + 1):
        o = center + offset
        if o >= len(low):
            break
        if low[o] <= pl + tol:
            cnt += 1
            z_top = max(z_top, low[o])
            ri += 1
        else:
            break
    nb_min = np.inf
    for offset in range(1, piv_left + 1):
        o = center - offset
        if o >= 0:
            nb_min = min(nb_min, low[o])
    for offset in range(1, piv_right + 1):
        o = center + offset
        if o < len(low):
            nb_min = min(nb_min, low[o])
    is_spike = spike_atr > 0 and nb_min < np.inf and (nb_min - pl) >= spike_atr * atr_arr[i - piv_right]
    valid = cnt >= min_cluster or is_spike
    return valid, z_top, li, ri


def _pattern_quality(
    p1: float,
    p2: float,
    neck: float,
    height: float,
    gap_bars: int,
    min_bars: int,
    max_gap: int,
    atr_v: float,
) -> float:
    """0-1 score: symmetry of extremes + spacing + height relative to ATR."""
    if atr_v <= 0 or height <= 0:
        return 0.0
    symmetry = 1.0 - min(1.0, abs(p1 - p2) / (atr_v + 1e-12))
    span = max_gap - min_bars
    spacing = 1.0 - abs((gap_bars - (min_bars + span / 2)) / (span / 2 + 1e-12))
    spacing = float(np.clip(spacing, 0, 1))
    height_score = float(np.clip(height / (2.5 * atr_v), 0, 1))
    return 0.4 * symmetry + 0.3 * spacing + 0.3 * height_score


def _htf_bias(df: pd.DataFrame, rule: str, ema_len: int) -> pd.Series:
    """Return +1 / -1 / 0 bias aligned to df index using closed HTF bars."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(0, index=df.index)
    # Infer if already coarse (daily+)
    if len(df) > 2:
        median_delta = pd.Series(df.index).diff().median()
        if median_delta is not pd.NaT and median_delta >= pd.Timedelta("1D"):
            # On daily, use own close vs EMA as soft bias
            e = ema(df["close"], ema_len)
            return pd.Series(np.where(df["close"] > e, 1, -1), index=df.index)

    ohlc = df["close"].resample(rule).last().dropna()
    e = ema(ohlc, ema_len)
    bias = pd.Series(np.where(ohlc > e, 1, -1), index=ohlc.index)
    # Shift 1 so only closed HTF bar is used
    bias = bias.shift(1)
    return bias.reindex(df.index, method="ffill").fillna(0).astype(int)


def run_backtest(df: pd.DataFrame, params: StrategyParams | None = None) -> BacktestResult:
    p = params or StrategyParams()
    data = df.copy()
    cols = {c.lower(): c for c in data.columns}
    for need in ("open", "high", "low", "close", "volume"):
        if need not in cols and need.title() in data.columns:
            pass
    data.columns = [c.lower() for c in data.columns]
    for need in ("open", "high", "low", "close"):
        if need not in data.columns:
            raise ValueError(f"missing column {need}")
    if "volume" not in data.columns:
        data["volume"] = 0.0

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    volume = data["volume"].to_numpy(dtype=float)
    n = len(data)

    ema_f = ema(data["close"], p.ema_fast).to_numpy()
    ema_s = ema(data["close"], p.ema_slow).to_numpy()
    atr_arr = atr(data, p.atr_len).to_numpy()
    adx_arr = adx(data, p.adx_len).to_numpy()
    vol_ma = volume_sma(data["volume"], p.vol_len).to_numpy()
    rsi_arr = rsi(data["close"], p.rsi_len).to_numpy() if p.use_rsi else np.full(n, 50.0)
    ph_s = pivot_high(data["high"], p.piv_left, p.piv_right).to_numpy()
    pl_s = pivot_low(data["low"], p.piv_left, p.piv_right).to_numpy()
    htf = _htf_bias(data, p.htf_rule, p.htf_ema).to_numpy() if p.use_htf_bias else np.zeros(n)
    open_ = data["open"].to_numpy(dtype=float)

    spread = ema_f - ema_s
    slope_f = np.full(n, np.nan)
    for i in range(p.slope_len, n):
        if atr_arr[i] and atr_arr[i] > 0:
            slope_f[i] = (ema_f[i] - ema_f[i - p.slope_len]) / atr_arr[i]

    last_up_idx = -1
    last_dn_idx = -1

    low_idx: list[int] = []
    low_px: list[float] = []
    low_top: list[float] = []
    high_idx: list[int] = []
    high_px: list[float] = []
    high_bot: list[float] = []

    bull_armed = bear_armed = False
    bull_broken = bear_broken = False
    bull_level = bear_level = np.nan
    bull_sl = bear_sl = np.nan
    bull_exp = bear_exp = -1
    bull_break_idx = bear_break_idx = -1
    bull_height = bear_height = 0.0
    bull_quality = bear_quality = 0.0
    bull_retest_ok = bear_retest_ok = (not p.retest_entry)

    trades: list[Trade] = []
    position: Trade | None = None
    equity = np.ones(n)
    eq = 1.0
    last_trade_exit = -10_000
    peak_since_entry = 0.0
    trough_since_entry = 0.0

    cost = (p.fee_bps + p.slip_bps) / 10_000.0

    for i in range(n):
        equity[i] = eq

        # EMA crosses
        if i > 0:
            if ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]:
                last_up_idx = i
            if ema_f[i - 1] >= ema_s[i - 1] and ema_f[i] < ema_s[i]:
                last_dn_idx = i

        # Manage open trade exits first (on this bar's OHLC)
        if position is not None:
            hit = False
            exit_px = None
            reason = None
            risk0 = abs(position.entry_price - (position.stop_init or position.stop))
            if position.side == "long":
                peak_since_entry = max(peak_since_entry, high[i])
                # Move stop to BE (+tiny) after tp1_r
                if p.tp1_r > 0 and risk0 > 0 and (peak_since_entry - position.entry_price) >= p.tp1_r * risk0:
                    be = position.entry_price * (1 + cost)
                    if be > position.stop:
                        position.stop = be
                # stop first (conservative)
                if low[i] <= position.stop:
                    exit_px = position.stop
                    reason = "stop"
                    hit = True
                elif high[i] >= position.target:
                    exit_px = position.target
                    reason = "target"
                    hit = True
                elif p.use_trail_after_1r and risk0 > 0 and (peak_since_entry - position.entry_price) >= risk0:
                    trail = close[i] - p.trail_atr * atr_arr[i]
                    if trail > position.stop:
                        position.stop = trail
            else:
                trough_since_entry = min(trough_since_entry, low[i])
                if p.tp1_r > 0 and risk0 > 0 and (position.entry_price - trough_since_entry) >= p.tp1_r * risk0:
                    be = position.entry_price * (1 - cost)
                    if be < position.stop:
                        position.stop = be
                if high[i] >= position.stop:
                    exit_px = position.stop
                    reason = "stop"
                    hit = True
                elif low[i] <= position.target:
                    exit_px = position.target
                    reason = "target"
                    hit = True
                elif p.use_trail_after_1r and risk0 > 0 and (position.entry_price - trough_since_entry) >= risk0:
                    trail = close[i] + p.trail_atr * atr_arr[i]
                    if trail < position.stop:
                        position.stop = trail

            if hit and exit_px is not None:
                if position.side == "long":
                    raw = (exit_px / position.entry_price) - 1.0
                else:
                    raw = (position.entry_price / exit_px) - 1.0
                raw -= 2 * cost
                risk_px = abs(position.entry_price - (position.stop_init or position.stop))
                position.exit_idx = i
                position.exit_price = exit_px
                position.reason = reason
                position.pnl_pct = raw
                if risk_px > 0:
                    if position.side == "long":
                        position.pnl_r = (exit_px - position.entry_price) / risk_px
                    else:
                        position.pnl_r = (position.entry_price - exit_px) / risk_px
                eq *= 1.0 + position.pnl_pct
                equity[i] = eq
                trades.append(position)
                last_trade_exit = i
                position = None

        atr_i = atr_arr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # ---- Detect clusters / patterns when pivot confirms ----
        if not np.isnan(ph_s[i]):
            ph = float(ph_s[i])
            valid_h, h_bot, _, _ = _cluster_high(
                high, atr_arr, i, ph, p.piv_left, p.piv_right, p.cluster_atr, p.min_cluster, p.spike_atr
            )
            h_idx = i - p.piv_right
            if valid_h:
                tol_v = atr_arr[i - p.piv_right] * p.tol_atr
                min_h = atr_arr[i - p.piv_right] * p.min_height_atr
                for j in range(len(high_idx) - 1, -1, -1):
                    i1 = high_idx[j]
                    p1 = high_px[j]
                    b1 = high_bot[j]
                    gap_bars = h_idx - i1
                    if gap_bars > p.max_gap:
                        break
                    if gap_bars >= p.min_bars and ph <= p1 + tol_v:
                        hh = np.max(high[i1 : h_idx + 1])
                        ll = np.min(low[i1 : h_idx + 1])
                        retrace_ok = ph >= ll + p.min_retrace * (p1 - ll)
                        height = min(h_bot, b1) - ll
                        if retrace_ok and height >= min_h and hh <= max(ph, p1) + tol_v:
                            if not (bear_armed and bear_broken):
                                q = _pattern_quality(p1, ph, ll, height, gap_bars, p.min_bars, p.max_gap, atr_arr[i - p.piv_right])
                                if q >= p.min_quality:
                                    bear_armed = True
                                    bear_broken = False
                                    bear_break_idx = -1
                                    bear_level = float(ll)
                                    bear_sl = float(max(ph, p1))
                                    bear_exp = i + p.max_bars
                                    bear_height = float(height)
                                    bear_quality = q
                                    bear_retest_ok = not p.retest_entry
                                    if p.opp_cancel:
                                        bull_armed = bull_broken = False
                            break
                high_idx.append(h_idx)
                high_px.append(ph)
                high_bot.append(h_bot)
                if len(high_idx) > 20:
                    high_idx.pop(0)
                    high_px.pop(0)
                    high_bot.pop(0)

        if not np.isnan(pl_s[i]):
            pl = float(pl_s[i])
            valid_l, l_top, _, _ = _cluster_low(
                low, atr_arr, i, pl, p.piv_left, p.piv_right, p.cluster_atr, p.min_cluster, p.spike_atr
            )
            l_idx = i - p.piv_right
            if valid_l:
                tol_v = atr_arr[i - p.piv_right] * p.tol_atr
                min_h = atr_arr[i - p.piv_right] * p.min_height_atr
                for j in range(len(low_idx) - 1, -1, -1):
                    i1 = low_idx[j]
                    p1 = low_px[j]
                    t1 = low_top[j]
                    gap_bars = l_idx - i1
                    if gap_bars > p.max_gap:
                        break
                    if gap_bars >= p.min_bars and pl >= p1 - tol_v:
                        hh = np.max(high[i1 : l_idx + 1])
                        ll = np.min(low[i1 : l_idx + 1])
                        retrace_ok = pl <= hh - p.min_retrace * (hh - p1)
                        height = hh - max(l_top, t1)
                        if retrace_ok and height >= min_h and ll >= min(pl, p1) - tol_v:
                            if not (bull_armed and bull_broken):
                                q = _pattern_quality(p1, pl, hh, height, gap_bars, p.min_bars, p.max_gap, atr_arr[i - p.piv_right])
                                if q >= p.min_quality:
                                    bull_armed = True
                                    bull_broken = False
                                    bull_break_idx = -1
                                    bull_level = float(hh)
                                    bull_sl = float(min(pl, p1))
                                    bull_exp = i + p.max_bars
                                    bull_height = float(height)
                                    bull_quality = q
                                    bull_retest_ok = not p.retest_entry
                                    if p.opp_cancel:
                                        bear_armed = bear_broken = False
                            break
                low_idx.append(l_idx)
                low_px.append(pl)
                low_top.append(l_top)
                if len(low_idx) > 20:
                    low_idx.pop(0)
                    low_px.pop(0)
                    low_top.pop(0)

        # ---- Update break / cancel ----
        if bull_armed:
            if close[i] < bull_sl:
                bull_armed = bull_broken = False
            elif bull_broken:
                if p.cross_win > 0 and i - bull_break_idx > p.cross_win:
                    bull_armed = bull_broken = False
                elif p.cancel_retest and not p.retest_entry and close[i] < bull_level:
                    bull_armed = bull_broken = False
                elif p.retest_entry and not bull_retest_ok:
                    # need close back near neckline then continue
                    if abs(close[i] - bull_level) <= p.retest_buf_atr * atr_i or close[i] < bull_level:
                        if close[i] >= bull_level - p.retest_buf_atr * atr_i:
                            bull_retest_ok = True
            else:
                if i > bull_exp:
                    bull_armed = False
                elif close[i] > bull_level + p.break_buf * atr_i:
                    vol_ok = (not p.use_volume) or (volume[i] >= p.vol_mult * vol_ma[i] if not np.isnan(vol_ma[i]) else False)
                    if vol_ok:
                        bull_broken = True
                        bull_break_idx = i
                        bull_retest_ok = not p.retest_entry

        if bear_armed:
            if close[i] > bear_sl:
                bear_armed = bear_broken = False
            elif bear_broken:
                if p.cross_win > 0 and i - bear_break_idx > p.cross_win:
                    bear_armed = bear_broken = False
                elif p.cancel_retest and not p.retest_entry and close[i] > bear_level:
                    bear_armed = bear_broken = False
                elif p.retest_entry and not bear_retest_ok:
                    if abs(close[i] - bear_level) <= p.retest_buf_atr * atr_i or close[i] > bear_level:
                        if close[i] <= bear_level + p.retest_buf_atr * atr_i:
                            bear_retest_ok = True
            else:
                if i > bear_exp:
                    bear_armed = False
                elif close[i] < bear_level - p.break_buf * atr_i:
                    vol_ok = (not p.use_volume) or (volume[i] >= p.vol_mult * vol_ma[i] if not np.isnan(vol_ma[i]) else False)
                    if vol_ok:
                        bear_broken = True
                        bear_break_idx = i
                        bear_retest_ok = not p.retest_entry

        # ---- Momentum gates ----
        spread_ok_up = spread[i] >= p.spread_atr * atr_i
        spread_ok_dn = -spread[i] >= p.spread_atr * atr_i
        score_up = (
            (1 if (not np.isnan(slope_f[i]) and slope_f[i] >= p.slope_atr) else 0)
            + (1 if i > 0 and spread[i] > spread[i - 1] else 0)
            + (1 if close[i] > max(ema_f[i], ema_s[i]) else 0)
        )
        score_dn = (
            (1 if (not np.isnan(slope_f[i]) and slope_f[i] <= -p.slope_atr) else 0)
            + (1 if i > 0 and spread[i] < spread[i - 1] else 0)
            + (1 if close[i] < min(ema_f[i], ema_s[i]) else 0)
        )
        adx_ok = p.adx_min <= 0 or adx_arr[i] >= p.adx_min

        win_up = p.confirm_win == 0 or (last_up_idx >= 0 and i - last_up_idx <= p.confirm_win)
        win_dn = p.confirm_win == 0 or (last_dn_idx >= 0 and i - last_dn_idx <= p.confirm_win)
        cross_ok_up = (not p.need_cross) or (last_up_idx > bull_break_idx and win_up)
        cross_ok_dn = (not p.need_cross) or (last_dn_idx > bear_break_idx and win_dn)

        htf_long_ok = (not p.use_htf_bias) or htf[i] >= 0
        htf_short_ok = (not p.use_htf_bias) or htf[i] <= 0

        bars_after_bull = i - bull_break_idx if bull_break_idx >= 0 else 9999
        bars_after_bear = i - bear_break_idx if bear_break_idx >= 0 else 9999
        break_win_ok_up = p.break_cross_win <= 0 or bars_after_bull <= p.break_cross_win
        break_win_ok_dn = p.break_cross_win <= 0 or bars_after_bear <= p.break_cross_win

        ext_up = (close[i] - bull_level) / atr_i if bull_broken and not np.isnan(bull_level) else 0.0
        ext_dn = (bear_level - close[i]) / atr_i if bear_broken and not np.isnan(bear_level) else 0.0
        ext_ok_up = p.max_extension_atr <= 0 or ext_up <= p.max_extension_atr
        ext_ok_dn = p.max_extension_atr <= 0 or ext_dn <= p.max_extension_atr

        stop_atr_up = (close[i] - bull_sl) / atr_i if bull_armed and not np.isnan(bull_sl) else 0.0
        stop_atr_dn = (bear_sl - close[i]) / atr_i if bear_armed and not np.isnan(bear_sl) else 0.0
        stop_ok_up = True
        stop_ok_dn = True
        if p.max_stop_atr > 0:
            stop_ok_up = stop_ok_up and stop_atr_up <= p.max_stop_atr
            stop_ok_dn = stop_ok_dn and stop_atr_dn <= p.max_stop_atr
        if p.min_stop_atr > 0:
            stop_ok_up = stop_ok_up and stop_atr_up >= p.min_stop_atr
            stop_ok_dn = stop_ok_dn and stop_atr_dn >= p.min_stop_atr

        dir_ok_up = (not p.require_close_dir) or (close[i] > open_[i])
        dir_ok_dn = (not p.require_close_dir) or (close[i] < open_[i])

        rsi_ok_up = (not p.use_rsi) or (not np.isnan(rsi_arr[i]) and rsi_arr[i] <= p.rsi_long_max)
        rsi_ok_dn = (not p.use_rsi) or (not np.isnan(rsi_arr[i]) and rsi_arr[i] >= p.rsi_short_min)

        long_signal = (
            bull_armed
            and bull_broken
            and bull_retest_ok
            and i >= bull_break_idx
            and ema_f[i] > ema_s[i]
            and spread_ok_up
            and score_up >= p.min_score
            and adx_ok
            and cross_ok_up
            and htf_long_ok
            and break_win_ok_up
            and ext_ok_up
            and stop_ok_up
            and dir_ok_up
            and rsi_ok_up
            and not p.short_only
        )
        short_signal = (
            bear_armed
            and bear_broken
            and bear_retest_ok
            and i >= bear_break_idx
            and ema_f[i] < ema_s[i]
            and spread_ok_dn
            and score_dn >= p.min_score
            and adx_ok
            and cross_ok_dn
            and htf_short_ok
            and break_win_ok_dn
            and ext_ok_dn
            and stop_ok_dn
            and dir_ok_dn
            and rsi_ok_dn
            and not p.long_only
        )

        can_enter = position is None and (i - last_trade_exit) >= p.cooldown

        if can_enter and long_signal:
            entry = close[i] * (1 + cost)
            stop = bull_sl
            if p.atr_stop_cap > 0:
                stop = max(stop, entry - p.atr_stop_cap * atr_i)
            risk = entry - stop
            if risk > 0:
                target = entry + p.rr_target * risk
                measured = bull_level + bull_height
                if measured > entry:
                    target = max(target, entry + (measured - entry) * min(1.0, p.rr_target / 2))
                position = Trade(
                    side="long",
                    entry_idx=i,
                    entry_price=entry,
                    stop=stop,
                    target=target,
                    stop_init=stop,
                    pattern_height=bull_height,
                    quality=bull_quality,
                    adx_at_entry=float(adx_arr[i]) if not np.isnan(adx_arr[i]) else 0.0,
                    spread_atr_at_entry=float(spread[i] / atr_i),
                    extension_atr=float(ext_up),
                    bars_after_break=int(bars_after_bull),
                    stop_atr=float(risk / atr_i),
                )
                peak_since_entry = high[i]
                bull_armed = bull_broken = False
        elif can_enter and short_signal:
            entry = close[i] * (1 - cost)
            stop = bear_sl
            if p.atr_stop_cap > 0:
                stop = min(stop, entry + p.atr_stop_cap * atr_i)
            risk = stop - entry
            if risk > 0:
                target = entry - p.rr_target * risk
                measured = bear_level - bear_height
                if measured < entry:
                    target = min(target, entry - (entry - measured) * min(1.0, p.rr_target / 2))
                position = Trade(
                    side="short",
                    entry_idx=i,
                    entry_price=entry,
                    stop=stop,
                    target=target,
                    stop_init=stop,
                    pattern_height=bear_height,
                    quality=bear_quality,
                    adx_at_entry=float(adx_arr[i]) if not np.isnan(adx_arr[i]) else 0.0,
                    spread_atr_at_entry=float(-spread[i] / atr_i),
                    extension_atr=float(ext_dn),
                    bars_after_break=int(bars_after_bear),
                    stop_atr=float(risk / atr_i),
                )
                trough_since_entry = low[i]
                bear_armed = bear_broken = False

    # Close any open position at last close
    if position is not None:
        exit_px = close[-1]
        risk_px = abs(position.entry_price - (position.stop_init or position.stop))
        position.exit_idx = n - 1
        position.exit_price = exit_px
        position.reason = "eod"
        if position.side == "long":
            position.pnl_pct = (exit_px / position.entry_price) - 1.0 - 2 * cost
            position.pnl_r = (exit_px - position.entry_price) / risk_px if risk_px > 0 else 0.0
        else:
            position.pnl_pct = (position.entry_price / exit_px) - 1.0 - 2 * cost
            position.pnl_r = (position.entry_price - exit_px) / risk_px if risk_px > 0 else 0.0
        eq *= 1.0 + position.pnl_pct
        equity[-1] = eq
        trades.append(position)

    metrics = compute_metrics(trades, pd.Series(equity, index=data.index))
    return BacktestResult(params=asdict(p), trades=trades, equity=pd.Series(equity, index=data.index), metrics=metrics)


def compute_metrics(trades: list[Trade], equity: pd.Series) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0,
            "net_pnl_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    rs = np.array([t.pnl_r for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    compound = float(np.prod([1 + t.pnl_pct for t in trades]) - 1.0)
    return {
        "trades": float(len(trades)),
        "win_rate": float((rs > 0).mean()),
        "profit_factor": float(pf),
        "avg_r": float(rs.mean()),
        "expectancy_r": float(rs.mean()),
        "net_pnl_pct": compound,
        "max_dd_pct": float(dd),
        "sharpe": sharpe,
        "long_trades": float(sum(1 for t in trades if t.side == "long")),
        "short_trades": float(sum(1 for t in trades if t.side == "short")),
    }
