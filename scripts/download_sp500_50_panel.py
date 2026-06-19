"""Download and process a local-only 50-stock S&P500 daily panel."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch

UNIVERSE_ID = "sp500_50_liquid_sector_v0"
YFINANCE_DOC_URL = "https://ranaroussi.github.io/yfinance/"
YAHOO_TERMS_URL = "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html"

SECTOR_TICKERS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("Information Technology", ("AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE")),
        ("Health Care", ("UNH", "JNJ", "LLY", "MRK", "ABBV", "TMO")),
        ("Financials", ("JPM", "BAC", "WFC", "GS", "MS", "V")),
        ("Consumer Discretionary", ("AMZN", "TSLA", "HD", "MCD", "NKE")),
        ("Communication Services", ("GOOGL", "META", "NFLX", "DIS", "VZ")),
        ("Industrials", ("CAT", "GE", "HON", "UNP", "RTX")),
        ("Consumer Staples", ("PG", "KO", "PEP", "WMT")),
        ("Energy", ("XOM", "CVX", "COP", "SLB")),
        ("Utilities", ("NEE", "DUK", "SO")),
        ("Materials", ("LIN", "APD", "SHW")),
        ("Real Estate", ("AMT", "PLD", "EQIX")),
    ]
)

SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
}

ConditionMode = Literal["spy_vix_level", "spy_vix_change", "vix_only"]
RawFormat = Literal["csv", "parquet"]


@dataclass(frozen=True)
class ProcessedPanel:
    """Processed tensors and metadata for the local panel."""

    data: torch.Tensor
    standardized_data: torch.Tensor
    labels: torch.Tensor
    metadata: dict[str, Any]
    aligned_prices: pd.DataFrame
    stock_returns: pd.DataFrame
    raw_conditions: pd.DataFrame
    sector_etf_returns: pd.DataFrame | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download a local-only yfinance S&P500 50-stock panel.",
    )
    parser.add_argument("--start", required=True, help="Inclusive download start date.")
    parser.add_argument("--end", required=True, help="Exclusive download end date.")
    parser.add_argument(
        "--output-root", default="data", help="Root containing raw/ and processed/."
    )
    parser.add_argument(
        "--include-sector-etfs",
        action="store_true",
        help="Also download sector ETF adjusted-close series for later ablations.",
    )
    parser.add_argument(
        "--raw-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Raw local file format. Parquet requires an installed pandas parquet engine.",
    )
    parser.add_argument("--window-length", type=int, default=60, help="Return-window length.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Leading fraction of windows used for label standardisation and train split.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=("spy_vix_level", "spy_vix_change", "vix_only"),
        default="spy_vix_level",
        help="Model-visible condition convention.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable yfinance progress output.",
    )
    return parser


def main() -> int:
    """Download raw prices, process return windows, and write local artefacts."""
    args = build_parser().parse_args()
    if args.window_length <= 1:
        raise SystemExit("--window-length must be greater than one.")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must lie in (0, 1).")

    yf = import_yfinance()
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw" / "sp500_50_panel"
    processed_dir = output_root / "processed" / "sp500_50_panel"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tickers = panel_tickers()
    required_symbols = [*tickers, "SPY", "^VIX"]
    prices = download_adjusted_close(
        yf=yf,
        symbols=required_symbols,
        start=args.start,
        end=args.end,
        progress=not args.no_progress,
    )
    write_frame(prices, raw_dir / "adjusted_close", raw_format=cast(RawFormat, args.raw_format))

    sector_etf_prices = None
    if args.include_sector_etfs:
        sector_etf_symbols = list(SECTOR_ETFS.values())
        sector_etf_prices = download_adjusted_close(
            yf=yf,
            symbols=sector_etf_symbols,
            start=args.start,
            end=args.end,
            progress=not args.no_progress,
        )
        write_frame(
            sector_etf_prices,
            raw_dir / "sector_etf_adjusted_close",
            raw_format=cast(RawFormat, args.raw_format),
        )

    processed = process_panel(
        prices=prices,
        sector_etf_prices=sector_etf_prices,
        start=args.start,
        end=args.end,
        yfinance_version=yfinance_version(),
        window_length=args.window_length,
        train_fraction=args.train_fraction,
        condition_mode=cast(ConditionMode, args.condition_mode),
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    save_processed_panel(processed, processed_dir=processed_dir)

    print("S&P500 50-stock panel download complete.")
    print(f"raw_dir={raw_dir}")
    print(f"processed_dir={processed_dir}")
    print(f"data_shape={tuple(processed.data.shape)}")
    print(f"labels_shape={tuple(processed.labels.shape)}")
    print(f"aligned_price_dates={len(processed.aligned_prices)}")
    print(f"window_start={processed.metadata['date_range']['first_window_start_date']}")
    print(f"window_end={processed.metadata['date_range']['last_window_end_date']}")
    print(f"condition_names={processed.metadata['condition_names']}")
    return 0


def import_yfinance() -> Any:
    """Import yfinance or raise a clear optional-dependency message."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is required for this downloader. Install the optional data group with "
            "`poetry install --with data`, then rerun the command."
        ) from exc
    return yf


def yfinance_version() -> str:
    """Return the installed yfinance version."""
    try:
        return importlib_metadata.version("yfinance")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def panel_tickers() -> list[str]:
    """Return the fixed 50-stock ticker order."""
    return [ticker for tickers in SECTOR_TICKERS.values() for ticker in tickers]


def sector_names_for_tickers() -> list[str]:
    """Return sector names in ticker order."""
    return [sector for sector, tickers in SECTOR_TICKERS.items() for _ticker in tickers]


def sector_label_ids() -> list[int]:
    """Return integer sector labels in ticker order."""
    labels: list[int] = []
    for sector_index, tickers in enumerate(SECTOR_TICKERS.values()):
        labels.extend([sector_index] * len(tickers))
    return labels


def download_adjusted_close(
    *,
    yf: Any,
    symbols: list[str],
    start: str,
    end: str,
    progress: bool,
) -> pd.DataFrame:
    """Download adjusted-close prices for a fixed symbol list."""
    frame = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=progress,
        threads=True,
        group_by="column",
    )
    if frame.empty:
        raise SystemExit(f"yfinance returned no rows for symbols={symbols}.")
    prices = extract_price_field(frame)
    missing_symbols = sorted(set(symbols) - {str(column) for column in prices.columns})
    if missing_symbols:
        raise SystemExit(f"Missing adjusted-close columns for: {missing_symbols}.")
    prices = prices.reindex(columns=symbols)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    if prices.empty:
        raise SystemExit("Adjusted-close panel is empty after column alignment.")
    return prices


def extract_price_field(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted close or close prices from a yfinance result frame."""
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = {str(value) for value in frame.columns.get_level_values(0)}
        level1 = {str(value) for value in frame.columns.get_level_values(1)}
        for field in ("Adj Close", "Close"):
            if field in level0:
                return cast(pd.DataFrame, frame[field])
            if field in level1:
                return cast(pd.DataFrame, frame.xs(field, level=1, axis=1))
        raise SystemExit("Downloaded frame does not contain 'Adj Close' or 'Close'.")
    return frame.to_frame() if isinstance(frame, pd.Series) else frame


def write_frame(frame: pd.DataFrame, path_prefix: Path, *, raw_format: RawFormat) -> Path:
    """Write a raw or processed frame in the requested local format."""
    if raw_format == "csv":
        path = path_prefix.with_suffix(".csv")
        frame.to_csv(path, index_label="Date")
        return path
    path = path_prefix.with_suffix(".parquet")
    try:
        frame.to_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Parquet output requires a pandas parquet engine such as pyarrow. "
            "Rerun with --raw-format csv or install a parquet engine outside this task."
        ) from exc
    return path


def process_panel(
    *,
    prices: pd.DataFrame,
    sector_etf_prices: pd.DataFrame | None,
    start: str,
    end: str,
    yfinance_version: str,
    window_length: int,
    train_fraction: float,
    condition_mode: ConditionMode,
    raw_dir: Path,
    processed_dir: Path,
) -> ProcessedPanel:
    """Convert adjusted prices into return windows, labels, and metadata."""
    tickers = panel_tickers()
    required_symbols = [*tickers, "SPY", "^VIX"]
    price_panel = prices.reindex(columns=required_symbols)
    missing_before = price_panel.isna().sum().astype(int).to_dict()
    aligned_prices = price_panel.dropna(axis=0, how="any")
    if len(aligned_prices) <= window_length:
        raise SystemExit(
            "Aligned panel is too short for the requested window length: "
            f"{len(aligned_prices)} price rows, window_length={window_length}."
        )

    log_prices = np.log(aligned_prices)
    log_returns = log_prices.diff().dropna(axis=0, how="any")
    stock_returns = log_returns[tickers]
    spy_returns = log_returns["SPY"]
    vix_log_level = np.log(aligned_prices["^VIX"]).loc[log_returns.index]
    vix_log_change = log_returns["^VIX"]

    raw_conditions = build_raw_conditions(
        spy_returns=spy_returns,
        vix_log_level=vix_log_level,
        vix_log_change=vix_log_change,
        condition_mode=condition_mode,
    )
    data, labels, window_start_dates, window_end_dates, label_stats = build_windows_and_labels(
        stock_returns=stock_returns,
        raw_conditions=raw_conditions,
        window_length=window_length,
        train_fraction=train_fraction,
    )
    train_count = int(label_stats["train_window_count"])
    return_stats = fit_return_standardization(data, train_count=train_count)
    standardized_data = standardize_return_windows(data, return_stats)
    sector_etf_returns = None
    if sector_etf_prices is not None:
        sector_etf_aligned = sector_etf_prices.dropna(axis=0, how="any")
        sector_etf_returns = np.log(sector_etf_aligned).diff().dropna(axis=0, how="any")
        sector_etf_returns = sector_etf_returns.loc[
            sector_etf_returns.index.intersection(stock_returns.index)
        ]

    metadata = build_metadata(
        start=start,
        end=end,
        yfinance_version=yfinance_version,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prices=prices,
        aligned_prices=aligned_prices,
        stock_returns=stock_returns,
        data=data,
        standardized_data=standardized_data,
        labels=labels,
        missing_before=missing_before,
        condition_mode=condition_mode,
        condition_names=list(raw_conditions.columns),
        label_stats=label_stats,
        return_stats=return_stats,
        train_count=train_count,
        train_fraction=train_fraction,
        window_start_dates=window_start_dates,
        window_end_dates=window_end_dates,
        include_sector_etfs=sector_etf_prices is not None,
        sector_etf_returns=sector_etf_returns,
    )
    return ProcessedPanel(
        data=data,
        standardized_data=standardized_data,
        labels=labels,
        metadata=metadata,
        aligned_prices=aligned_prices,
        stock_returns=stock_returns,
        raw_conditions=raw_conditions,
        sector_etf_returns=sector_etf_returns,
    )


def fit_return_standardization(
    data: torch.Tensor,
    *,
    train_count: int,
    epsilon: float = 1e-6,
) -> dict[str, list[float]]:
    """Fit per-asset return standardisation on train windows only."""
    if train_count <= 0:
        raise SystemExit("Cannot fit return standardisation with an empty train split.")
    train_data = data[:train_count].float()
    flat_returns = train_data.reshape(-1, train_data.shape[-1])
    mean = flat_returns.mean(dim=0)
    std = flat_returns.std(dim=0, unbiased=False).clamp_min(float(epsilon))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": float(epsilon),
    }


def standardize_return_windows(
    data: torch.Tensor,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    """Apply per-asset return standardisation to raw window tensors."""
    mean = torch.as_tensor(stats["mean"], dtype=data.dtype).view(1, 1, -1)
    std = torch.as_tensor(stats["std"], dtype=data.dtype).view(1, 1, -1)
    return (data - mean) / std


def build_raw_conditions(
    *,
    spy_returns: pd.Series,
    vix_log_level: pd.Series,
    vix_log_change: pd.Series,
    condition_mode: ConditionMode,
) -> pd.DataFrame:
    """Build unstandardised model-visible conditions."""
    if condition_mode == "spy_vix_level":
        return pd.DataFrame(
            {
                "spy_log_return_start": spy_returns,
                "log_vix_level_start": vix_log_level,
            }
        )
    if condition_mode == "spy_vix_change":
        return pd.DataFrame(
            {
                "spy_log_return_start": spy_returns,
                "vix_log_change_start": vix_log_change,
            }
        )
    return pd.DataFrame({"log_vix_level_start": vix_log_level})


def build_windows_and_labels(
    *,
    stock_returns: pd.DataFrame,
    raw_conditions: pd.DataFrame,
    window_length: int,
    train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], dict[str, Any]]:
    """Build return windows and train-standardised labels."""
    n_windows = len(stock_returns) - window_length + 1
    if n_windows <= 0:
        raise SystemExit("Not enough return rows to build one window.")

    data_array = np.stack(
        [
            stock_returns.iloc[start_index : start_index + window_length].to_numpy(
                dtype=np.float32,
            )
            for start_index in range(n_windows)
        ],
        axis=0,
    )
    label_array = raw_conditions.iloc[:n_windows].to_numpy(dtype=np.float32)
    train_count = int(np.floor(n_windows * train_fraction))
    train_count = min(max(train_count, 1), n_windows - 1)
    label_mean = label_array[:train_count].mean(axis=0, keepdims=True)
    label_std = label_array[:train_count].std(axis=0, keepdims=True)
    label_std = np.where(label_std < 1e-8, 1.0, label_std)
    labels = (label_array - label_mean) / label_std

    return_dates = [str(index.date()) for index in stock_returns.index]
    window_start_dates = return_dates[:n_windows]
    window_end_dates = return_dates[window_length - 1 :]
    label_stats = {
        "train_window_count": train_count,
        "label_mean": label_mean.reshape(-1).astype(float).tolist(),
        "label_std": label_std.reshape(-1).astype(float).tolist(),
    }
    return (
        torch.from_numpy(data_array).float(),
        torch.from_numpy(labels.astype(np.float32)).float(),
        window_start_dates,
        window_end_dates,
        label_stats,
    )


def build_metadata(
    *,
    start: str,
    end: str,
    yfinance_version: str,
    raw_dir: Path,
    processed_dir: Path,
    prices: pd.DataFrame,
    aligned_prices: pd.DataFrame,
    stock_returns: pd.DataFrame,
    data: torch.Tensor,
    standardized_data: torch.Tensor,
    labels: torch.Tensor,
    missing_before: dict[str, int],
    condition_mode: ConditionMode,
    condition_names: list[str],
    label_stats: dict[str, Any],
    return_stats: dict[str, list[float]],
    train_count: int,
    train_fraction: float,
    window_start_dates: list[str],
    window_end_dates: list[str],
    include_sector_etfs: bool,
    sector_etf_returns: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build JSON-safe local metadata."""
    tickers = panel_tickers()
    sectors = sector_names_for_tickers()
    return {
        "universe_id": UNIVERSE_ID,
        "tickers": tickers,
        "sectors": sectors,
        "sector_names": list(SECTOR_TICKERS.keys()),
        "sector_label_ids": sector_label_ids(),
        "ticker_sector_map": dict(zip(tickers, sectors, strict=True)),
        "condition_mode": condition_mode,
        "condition_names": condition_names,
        "shape": {"data": list(data.shape), "labels": list(labels.shape)},
        "date_range": {
            "requested_start": start,
            "requested_end": end,
            "first_aligned_price_date": str(aligned_prices.index[0].date()),
            "last_aligned_price_date": str(aligned_prices.index[-1].date()),
            "first_return_date": str(stock_returns.index[0].date()),
            "last_return_date": str(stock_returns.index[-1].date()),
            "first_window_start_date": window_start_dates[0],
            "last_window_start_date": window_start_dates[-1],
            "first_window_end_date": window_end_dates[0],
            "last_window_end_date": window_end_dates[-1],
        },
        "window_start_dates": window_start_dates,
        "window_end_dates": window_end_dates,
        "missing_data": {
            "handling": "inner_join_drop_any_missing_after_adjusted_close_alignment",
            "forward_fill_used": False,
            "raw_price_rows": len(prices),
            "aligned_price_rows": len(aligned_prices),
            "dropped_price_rows": int(len(prices) - len(aligned_prices)),
            "missing_adjusted_close_by_symbol": {
                str(symbol): int(count) for symbol, count in missing_before.items()
            },
        },
        "split": {
            "train_fraction": float(train_fraction),
            "train_window_count": int(train_count),
            "eval_window_count": int(data.shape[0] - train_count),
        },
        "label_standardisation": {
            "fit_split": "train",
            "condition_names": condition_names,
            "mean": label_stats["label_mean"],
            "std": label_stats["label_std"],
        },
        "return_standardisation": {
            "fit_split": "train",
            "asset_names": tickers,
            "mean": return_stats["mean"],
            "std": return_stats["std"],
            "epsilon": return_stats["epsilon"],
            "raw_tensor": str(processed_dir / "raw_data.pt"),
            "standardized_tensor": str(processed_dir / "standardized_data.pt"),
            "standardized_shape": list(standardized_data.shape),
        },
        "files": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "data_tensor": str(processed_dir / "data.pt"),
            "raw_data_tensor": str(processed_dir / "raw_data.pt"),
            "standardized_data_tensor": str(processed_dir / "standardized_data.pt"),
            "labels_tensor": str(processed_dir / "labels.pt"),
            "metadata": str(processed_dir / "metadata.json"),
        },
        "sector_etfs": {
            "included": include_sector_etfs,
            "mapping": SECTOR_ETFS,
            "return_rows": 0 if sector_etf_returns is None else len(sector_etf_returns),
        },
        "yfinance": {
            "version": yfinance_version,
            "documentation": YFINANCE_DOC_URL,
        },
        "data_use_caveat": {
            "summary": (
                "Yahoo-backed data is for local research use only in this benchmark. "
                "Do not commit or redistribute downloaded data."
            ),
            "yahoo_terms": YAHOO_TERMS_URL,
            "no_redistribution": True,
        },
    }


def save_processed_panel(processed: ProcessedPanel, *, processed_dir: Path) -> None:
    """Write processed tensors, CSV summaries, and metadata."""
    torch.save(processed.data, processed_dir / "data.pt")
    torch.save(processed.data, processed_dir / "raw_data.pt")
    torch.save(processed.standardized_data, processed_dir / "standardized_data.pt")
    torch.save(processed.labels, processed_dir / "labels.pt")
    processed.stock_returns.to_csv(processed_dir / "stock_log_returns.csv", index_label="Date")
    processed.raw_conditions.to_csv(processed_dir / "raw_conditions.csv", index_label="Date")
    if processed.sector_etf_returns is not None:
        processed.sector_etf_returns.to_csv(
            processed_dir / "sector_etf_log_returns.csv",
            index_label="Date",
        )
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(processed.metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    """Return a JSON-safe value."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclass_is_instance(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def dataclass_is_instance(value: Any) -> bool:
    """Return whether ``value`` is a dataclass instance."""
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


if __name__ == "__main__":
    raise SystemExit(main())
