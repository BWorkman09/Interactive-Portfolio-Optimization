import gym
import numpy as np
import pandas as pd

class CustomPortfolioEnv(gym.Env):
    def __init__(self, data, risk_profile="Moderate"):
        super(CustomPortfolioEnv, self).__init__()

        self.data = data
        self.assets = data.columns.tolist()
        self.num_assets = len(self.assets)
        self.current_step = 0

        self.returns = self.data.pct_change().dropna().values

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)

        self.risk_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_profile]

        self.portfolio_value = 1_000.0

        # NEW: Track full history for charting
        self.history = [self.portfolio_value]
        self.dates = self.data.index.tolist()

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1_000.0
        self.history = [self.portfolio_value]
        return self.returns[self.current_step]

    def step(self, action):
        weights = np.clip(action, 0, 1)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= np.sum(weights)

        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(weights, asset_returns)

        # Diversification bonus
        concentration_penalty = np.sum(np.square(weights))
        diversification_bonus = 1 - concentration_penalty

        adjusted_return = (portfolio_return * self.risk_multiplier) + (0.001 * diversification_bonus)

        self.portfolio_value *= (1 + adjusted_return)

        self.current_step += 1

        terminated = self.current_step >= len(self.returns) - 1
        truncated = False

        if not terminated:
            obs = self.returns[self.current_step]
        else:
            obs = np.zeros(self.num_assets)

        self.history.append(self.portfolio_value)

        info = {
            "portfolio_value": self.portfolio_value,
            "history": self.history,
            "dates": self.dates[1:self.current_step+2]
        }

        return obs, adjusted_return, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}: Portfolio Value ${self.portfolio_value:.2f}")
