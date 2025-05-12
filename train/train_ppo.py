import os
import sys

# === ADD THIS BLOCK ===
# Compute project root (one level up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Prepend env/ so Python can find trading_env.py there
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))
# ======================

# Now regular imports work:
from trading_env import TradingEnv


import glob
import json
import argparse
from stable_baselines3 import PPO
from pathlib import Path

# === Load config.json ===
with open("config.json") as f:
    cfg = json.load(f)

# Extract PPO params and environment settings
PPO_PARAMS = cfg["training"]["PPO_PARAMS"]
DEFAULT_DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")

# TODO: import your actual trading environment
# from trading_env import TradingEnv

def train_for_ticker(ticker, data_dir, total_timesteps, model_dir, stats_dir):
    """
    Train a PPO model for a single ticker using all matching CSVs in data_dir.
    """
    # discover all CSVs for this ticker
    pattern = os.path.join(data_dir, f"{ticker}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No data files found for {ticker} in {data_dir}")
        return

    print(f"Training {ticker} on files: {files}")
    # initialize your trading environment with the CSVs
    # env = TradingEnv(csv_files=files, config=cfg)
    env = None  # replace with actual env initialization

    # ensure output directories exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    # train the PPO model
    model = PPO("MlpPolicy", env, **PPO_PARAMS)
    model.learn(total_timesteps=total_timesteps)

    # save the trained model
    model_path = os.path.join(model_dir, f"{ticker}_ppo.zip")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # if your env supports saving normalization stats:
    # stats_path = os.path.join(stats_dir, f"{ticker}_stats.pkl")
    # env.save_stats(stats_path)
    # print(f"Saved stats: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on one or multiple tickers using generated CSV data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker",
        help="Single stock ticker to train (e.g., NVDA)"
    )
    group.add_argument(
        "--tickers-file",
        help="Path to file with one ticker symbol per line"
    )
    parser.add_argument(
        "--total-timesteps", type=int, required=True,
        help="Total timesteps to train each model"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--stats-dir", default="stats",
        help="Directory to save normalization stats"
    )
    args = parser.parse_args()

    # Determine tickers to process
    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = [args.ticker]

    # Train for each ticker
    for ticker in tickers:
        train_for_ticker(
            ticker=ticker,
            data_dir=DEFAULT_DATA_DIR,
            total_timesteps=args.total_timesteps,
            model_dir=args.model_dir,
            stats_dir=args.stats_dir
        )
