import argparse
import json
import numpy as np
from stable_baselines3 import PPO
from trading_env import TradingEnv

def evaluate(env, model, n_episodes=5, deterministic=True):
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model",  required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    env = TradingEnv(data_files=cfg["evaluation"]["data_files"], cfg=cfg)
    model = PPO.load(args.model)
    rewards = evaluate(env, model, cfg["evaluation"]["n_eval_episodes"], cfg["evaluation"].get("deterministic", True))
    for i, r in enumerate(rewards):
        print(f"Episode {i+1}: Reward {r:.2f}")
    print(f"Mean Reward: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
