import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import urllib3
import yfinance as yf

from stock_forecaster.config import FEATURE_COLUMNS, ModelConfig
from stock_forecaster.schemas import CompanyProfile, StockBundle


class DataServiceError(Exception):
    pass


def _is_rate_limit_message(message: str) -> bool:
    lower = message.lower()
    return "429" in lower or "too many requests" in lower or "rate limit" in lower


def _is_yahoo_non_retryable_message(message: str) -> bool:
    lower = message.lower()
    tokens = [
        "yftzmissingerror",
        "no timezone found",
        "possibly delisted",
        "expecting value: line 1 column 1",
    ]
    return any(token in lower for token in tokens)


def _sanitize_ticker(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ticker.upper())


def _market_cache_path(ticker: str, config: ModelConfig) -> Path:
    base = Path(__file__).resolve().parents[1] / config.artifacts_dir / "market_data"
    base.mkdir(parents=True, exist_ok=True)
    safe_ticker = _sanitize_ticker(ticker)
    return base / f"{safe_ticker}.csv"


def _is_cache_fresh(cache_path: Path, config: ModelConfig) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds <= (config.market_cache_ttl_hours * 3600)


def _normalize_downloaded_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        raise DataServiceError(
            f"No market data returned for {ticker}. It may be invalid or temporarily unavailable."
        )

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    # Normalize title casing from external providers.
    rename_map = {str(col).lower(): col for col in normalized.columns}
    for expected in FEATURE_COLUMNS:
        lower_expected = expected.lower()
        if lower_expected in rename_map and rename_map[lower_expected] != expected:
            normalized = normalized.rename(columns={rename_map[lower_expected]: expected})

    missing = [col for col in FEATURE_COLUMNS if col not in normalized.columns]
    if missing:
        raise DataServiceError(
            f"Downloaded data for {ticker} is missing required columns: {', '.join(missing)}"
        )

    normalized = normalized[FEATURE_COLUMNS].dropna().sort_index()

    if normalized.empty:
        raise DataServiceError(f"Historical data for {ticker} is empty after cleanup.")

    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)

    return normalized


def _read_cached_market_data(cache_path: Path) -> pd.DataFrame:
    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return _normalize_downloaded_frame(data, ticker=cache_path.stem)


def _write_cached_market_data(cache_path: Path, data: pd.DataFrame) -> None:
    data.to_csv(cache_path)


def _download_from_yahoo(ticker: str, config: ModelConfig) -> pd.DataFrame:
    last_error_message = ""

    # Silence noisy provider logs; we surface cleaner app-level errors.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    for attempt in range(1, config.max_download_retries + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            raw = ticker_obj.history(
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            return _normalize_downloaded_frame(raw, ticker=ticker)
        except Exception as exc:  # pragma: no cover - network/provider behavior is non-deterministic
            last_error_message = str(exc)

            if _is_yahoo_non_retryable_message(last_error_message):
                break

            if attempt == config.max_download_retries:
                break

            sleep_seconds = config.retry_backoff_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    if _is_rate_limit_message(last_error_message):
        raise DataServiceError(
            "Yahoo Finance is rate limiting requests right now (HTTP 429)."
        )

    raise DataServiceError(
        f"Yahoo provider failed for {ticker}. Reason: {last_error_message or 'unknown error'}"
    )


def _stooq_symbol(ticker: str) -> str:
    raw = ticker.strip().lower()

    if "." in raw or raw.startswith("^"):
        return raw

    # Common US-stock convenience mapping.
    return f"{raw}.us"


def _download_from_stooq(ticker: str) -> pd.DataFrame:
    symbol = _stooq_symbol(ticker)
    https_url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    http_url = f"http://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockForecaster/1.0)"}

    errors = []
    raw = None

    try:
        response = requests.get(https_url, timeout=20, headers=headers, verify=True)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
    except requests.exceptions.SSLError as exc:
        errors.append(f"HTTPS SSL verify failed: {exc}")
    except Exception as exc:  # pragma: no cover
        errors.append(f"HTTPS fetch failed: {exc}")

    if raw is None:
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(https_url, timeout=20, headers=headers, verify=False)
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
        except Exception as exc:  # pragma: no cover
            errors.append(f"HTTPS insecure fallback failed: {exc}")

    if raw is None:
        try:
            response = requests.get(http_url, timeout=20, headers=headers)
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
        except Exception as exc:  # pragma: no cover
            errors.append(f"HTTP fallback failed: {exc}")

    if raw is None:
        raise DataServiceError(
            f"Stooq provider failed for {ticker}. Reason: {' | '.join(errors)}"
        )

    if raw.empty:
        raise DataServiceError(f"Stooq returned empty data for {ticker} ({symbol}).")

    raw.columns = [str(col).strip().title() for col in raw.columns]
    if "Date" not in raw.columns:
        raise DataServiceError(f"Stooq response for {ticker} is missing Date column.")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")

    return _normalize_downloaded_frame(raw, ticker=ticker)


def _download_market_data(ticker: str, config: ModelConfig) -> pd.DataFrame:
    errors = []

    try:
        return _download_from_yahoo(ticker, config)
    except DataServiceError as exc:
        errors.append(str(exc))

    try:
        return _download_from_stooq(ticker)
    except DataServiceError as exc:
        errors.append(str(exc))

    joined = " | ".join(errors)
    raise DataServiceError(
        f"Unable to fetch market data for {ticker}. Tried Yahoo and Stooq. {joined}"
    )


def _build_company_profile(ticker: str) -> CompanyProfile:
    # Keep profile lookup lightweight to avoid rate-limited quoteSummary endpoints.
    display_name = ticker.upper()
    sector = "N/A"

    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = getattr(ticker_obj, "fast_info", None)
        exchange = None
        if fast_info:
            exchange = fast_info.get("exchange")
        if exchange:
            sector = str(exchange)
    except Exception:
        pass

    return CompanyProfile(name=display_name, sector=sector, logo_url="")


def load_stock_bundle(ticker: str, sequence_length: int = ModelConfig().sequence_length) -> StockBundle:
    config = ModelConfig()
    cache_path = _market_cache_path(ticker, config)

    if _is_cache_fresh(cache_path, config):
        data = _read_cached_market_data(cache_path)
    else:
        try:
            data = _download_market_data(ticker, config)
            _write_cached_market_data(cache_path, data)
        except DataServiceError as fetch_error:
            if cache_path.exists():
                data = _read_cached_market_data(cache_path)
            else:
                raise DataServiceError(
                    f"{fetch_error} Also, no local cached data is available for fallback."
                ) from fetch_error

    if len(data) < sequence_length:
        raise DataServiceError(
            f"Not enough past information for {ticker} to make a forecast."
        )

    profile = _build_company_profile(ticker)
    last_window = data.tail(sequence_length)

    return StockBundle(data=data, last_window=last_window, profile=profile)
