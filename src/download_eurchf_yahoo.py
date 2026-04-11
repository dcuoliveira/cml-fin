import argparse
import os
import sys
import pandas as pd


def _validate_dates(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
    end_ts = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")

    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date.")

    return start_ts, end_ts


def download_eurchf_data(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required. Install it with: pip install yfinance"
        ) from exc

    start_ts, end_ts = _validate_dates(start_date, end_date)

    # Yahoo's end parameter is exclusive in yfinance, so include next day.
    data = yf.download(
        "EURCHF=X",
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data returned for EURCHF=X in the requested date range."
        )

    data.index.name = "Date"
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download daily EUR/CHF Yahoo Finance data."
    )
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "data", "inputs", "eurchf_yahoo_daily.csv"
        ),
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    try:
        data = download_eurchf_data(args.start_date, args.end_date)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data.to_csv(args.output)
    print(
        f"Saved {len(data)} rows of EURCHF=X daily data to: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
