import json, gym, numpy as np, pandas as pd

# Load config
with open('config.json') as f:
    cfg = json.load(f)['training']

class TradingEnv(gym.Env):
    def __init__(self, data, features=None, reward_components=None):
        super(TradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.features = features if features is not None else []
        self.reward_components = reward_components if reward_components is not None else []

        self.current_step = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.position = 0  # -1 = short, 0 = flat, 1 = long
        self.portfolio_value = self.cash

        # Action space: [Hold, Buy, Sell]
        self.action_space = gym.spaces.Discrete(3)

        # Observation space (assumes price + simple indicators)
        obs_dim = 5  # open, high, low, close, volume
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.portfolio_value = self.cash
        return self._get_observation()

    def step(self, action):
        done = False
        reward = 0

        if self.current_step >= len(self.data) - 1:
            done = True

        price = self.data.loc[self.current_step, 'close']
        next_price = self.data.loc[self.current_step + 1, 'close']

        # Simulate trade
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
        elif action == 2:  # Sell
            if self.position == 1:
                reward += next_price - price
                self.cash += reward
                self.position = 0

        # Calculate basic portfolio value
        self.portfolio_value = self.cash
        if self.position == 1:
            self.portfolio_value += next_price

        # Apply reward components
        for component in self.reward_components:
            if component["type"] == "pnl":
                reward += (next_price - price) * self.position * component.get("scale", 1.0)

        self.current_step += 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        row = self.data.loc[self.current_step]
        obs = np.array([row['open'], row['high'], row['low'], row['close'], row['volume']], dtype=np.float32)
        return obs