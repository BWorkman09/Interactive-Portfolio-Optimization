import gym
import numpy as np

class CustomPortfolioEnv(gym.Env):
    def __init__(self, data, risk_profile="Moderate"):
        super(CustomPortfolioEnv, self).__init__()

        self.data = data
        self.assets = data.columns.tolist()
        self.num_assets = len(self.assets)

        # Normalize to returns
        self.returns = self.data.pct_change().dropna().values

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)

        self.risk_multiplier = {
            "Conservative": 0.5,
            "Moderate": 1.0,
            "Aggressive": 1.5
        }.get(risk_profile, 1.0)

        self.portfolio_value = 1000.0
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1000.0

        obs = self.returns[self.current_step]
        obs = np.nan_to_num(obs, nan=0.0)  # 👈 ensure no NaNs
        return obs

    def step(self, action):
        weights = np.clip(action, 0, 1)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / self.num_assets
        else:
            weights /= np.sum(weights)

        asset_returns = self.returns[self.current_step]
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)  # 👈 ensure no NaNs

        portfolio_return = np.dot(weights, asset_returns)
        portfolio_return = np.clip(portfolio_return, -1.0, 1.0)

        adjusted_return = portfolio_return * self.risk_multiplier
        self.portfolio_value *= (1 + adjusted_return)

        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1

        if not done:
            obs = self.returns[self.current_step]
            obs = np.nan_to_num(obs, nan=0.0)  # 👈 again sanitize NaNs
        else:
            obs = np.zeros(self.num_assets)

        reward = adjusted_return
        info = {"portfolio_value": self.portfolio_value}

        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}: Portfolio Value ${self.portfolio_value:.2f}")
