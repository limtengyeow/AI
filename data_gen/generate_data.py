import os
import json
import requests
import pandas as pd
import time
import argparse
from datetime import datetime
from pathlib import Path

# === Load config.json ===
with open("config.json") as f:
    cfg = json.load(f)

API_KEY = cfg["env"]["POLYGON_API_KEY"]
DEFAULT_DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DEFAULT_DATA_DIR).mkdir(exist_ok=True)

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date, save_path):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )
    params = {
        "adjusted": "true",
        "limit": 50000,
        "apiKey": API_KEY
    }

    all_data = []
    print(f"Fetching {ticker} from {from_date} to {to_date}...")
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed for {ticker}:", response.text)
            break

        data = response.json().get("results", [])
        if not data:
            break

        all_data.extend(data)

        if len(data) < 50000:
            break

        last_timestamp = data[-1]['t']
        next_from = datetime.utcfromtimestamp(last_timestamp / 1000.0).strftime('%Y-%m-%d')
        params["from"] = next_from
        time.sleep(1)

    df = pd.DataFrame(all_data)
    if df.empty:
        print(f"No data for {ticker}.")
        return

    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    }, inplace=True)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.set_index("timestamp", inplace=True)
    df.to_csv(save_path)
    print(f"Saved {ticker} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data from Polygon.io")
    parser.add_argument("--ticker", help="Single stock ticker (e.g., NVDA)")
    parser.add_argument("--tickers-file", help="Path to file with one ticker per line")
    parser.add_argument("--interval", required=True, help="Interval (e.g., 5min)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", help="Output CSV path for single ticker only")

    args = parser.parse_args()
    interval = args.interval
    multiplier = int(interval.replace("min", "")) if "min" in interval else 1
    timespan = "minute" if "min" in interval else interval

    if args.tickers_file:
        with open(args.tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        for ticker in tickers:
            out_path = os.path.join(DEFAULT_DATA_DIR, f"{ticker}_{interval}.csv")
            fetch_polygon_ohlcv(ticker, multiplier, timespan, args.start, args.end, out_path)
    elif args.ticker:
        out_path = args.out or os.path.join(DEFAULT_DATA_DIR, f"{args.ticker}_{interval}.csv")
        fetch_polygon_ohlcv(args.ticker, multiplier, timespan, args.start, args.end, out_path)
    else:
        print("Error: Provide either --ticker or --tickers-file")