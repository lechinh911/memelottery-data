"""Market data download helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    # yfinance sometimes returns 'adj close'
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].copy()
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def load_symbol(
    symbol: str,
    interval: str = "1h",
    period: str = "2y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV via yfinance.
    Note: yfinance hour data max ~730d; daily can go longer with period='max'.
    """
    safe = symbol.replace("/", "-").replace("=", "")
    path = CACHE / f"{safe}_{interval}_{period}.csv"
    if use_cache and path.exists():
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        return _normalize(df)

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {symbol} {interval} {period}")
    df = _normalize(df)
    df.index.name = "date"
    df.to_csv(path)
    return df


# Universe for research (liquid, diverse regimes)
DEFAULT_UNIVERSE = [
    # Crypto
    ("BTC-USD", "1h", "730d"),
    ("ETH-USD", "1h", "730d"),
    ("BTC-USD", "1d", "5y"),
    ("ETH-USD", "1d", "5y"),
    # FX / indices / commodities via Yahoo
    ("EURUSD=X", "1h", "730d"),
    ("GC=F", "1h", "730d"),  # Gold
    ("ES=F", "1h", "730d"),  # S&P futures
    ("AAPL", "1d", "5y"),
]
