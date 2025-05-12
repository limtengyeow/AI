import argparse
import json
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

def run_inference(env, model, output_file):
    obs = env.reset()
    done = False
    logs = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        info = {
            "step": env.current_step,
            "action": action,
            "price": env.data.loc[env.current_step, "close"],
            "date": env.data.loc[env.current_step, "Date"]
        }
        logs.append(info)
        obs, _, done, _ = env.step(action)
    df = pd.DataFrame(logs)
    df.to_csv(output_file, index=False)
    print(f"Inference complete, actions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--input_csv",   required=True)
    parser.add_argument("--output_csv",  required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    env = TradingEnv(data_files=[args.input_csv], cfg=cfg)
    model = PPO.load(args.model)
    run_inference(env, model, args.output_csv)
