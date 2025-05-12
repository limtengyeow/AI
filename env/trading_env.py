import json, gym, numpy as np, pandas as pd

# Load config
with open('config.json') as f:
    cfg = json.load(f)['training']

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df); self.step_i = 0; self.pos = 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def _get_obs(self):
        r = self.df.loc[self.step_i]
        return np.array([
            r['open'], r['high'], r['low'], r['close'], r['volume'],
            r['ribbon_w'], r['ribbon_vel'],
            r['close']/self.df['close'].shift(1).iloc[self.step_i]-1,
            r['close'].pct_change(5).fillna(0)
        ], dtype=np.float32)

    def reset(self):
        self.step_i = 0; self.pos = 1
        return self._get_obs()

    def step(self, action):
        r = self.df['close'].pct_change().fillna(0).iloc[self.step_i] * (action-1)
        raw = np.clip(r, *cfg['CLIP_REWARD']) * cfg['REWARD_SCALE']
        self.pos = action; self.step_i += 1
        done = self.step_i >= self.n_steps-1
        return self._get_obs(), raw, done, {}