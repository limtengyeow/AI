import argparse
import json
import pandas as pd
import sys

# Ensure Gym environments work with SB3
try:
    import shimmy  # Shimmy bridges OpenAI Gym to Gymnasium
except ImportError:
    print(
        "Warning: shimmy not installed; install via 'pip install shimmy>=2.0' if you encounter env errors",
        file=sys.stderr,
    )

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ← Updated import to point into the env/ package
from env.trading_env import TradingEnv

# Load config
with open("config.json") as f:
    cfg = json.load(f)["training"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        help="CSV files for training",
    )
    parser.add_argument(
        "--model-out",
        default="ppo_trend.zip",
        help="Output model file",
    )
    parser.add_argument(
        "--stats-out",
        default="vecnorm_trend.pkl",
        help="Output normalization stats",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total timesteps to train",
    )
    args = parser.parse_args()

    # Load dataframes
    dfs = []
    for path in args.data_files:
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        dfs.append(df)

    # Create vectorized envs
    env_fns = [lambda df=df: TradingEnv(df) for df in dfs]
    vec_env = DummyVecEnv(env_fns)
    env = VecNormalize(
        vec_env,
        norm_obs=cfg["NORM_OBS"],
        norm_reward=cfg["NORM_REWARD"],
    )

    # Initialize and train model
    model = PPO("MlpPolicy", env, verbose=1, **cfg["PPO_PARAMS"])
    model.learn(total_timesteps=args.total_timesteps)

    # Save model and normalization stats
    model.save(args.model_out)
    env.save(args.stats_out)

    print("✅ Training complete.")
