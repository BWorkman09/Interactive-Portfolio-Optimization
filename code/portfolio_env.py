import gym
import numpy as np

class CustomPortfolioEnv(gym.Env):
    def __init__(self, data, risk_profile="Moderate"):
        super(CustomPortfolioEnv, self).__init__()
        
        self.data = data  # historical price data (DataFrame)
        self.assets = data.columns.tolist()
        self.num_assets = len(self.assets)
        self.current_step = 0

        # Normalize price changes into returns
        self.returns = self.data.pct_change().dropna().values

        # Action: portfolio weights [0,1] per asset
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation: recent returns
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)

        # Risk adjustment
        self.risk_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_profile]

        # Track portfolio
        self.portfolio_value = 1_000.0  # starting $1000

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1_000.0
        return self.returns[self.current_step]
    
    def step(self, action):
        weights = np.clip(action, 0, 1)

        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= np.sum(weights)

        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(weights, asset_returns)
        adjusted_return = portfolio_return * self.risk_multiplier
        self.portfolio_value *= (1 + adjusted_return)

        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1

        if not done:
            obs = self.returns[self.current_step]
        else:
            obs = np.zeros(self.num_assets)

        reward = adjusted_return
        info = {"portfolio_value": self.portfolio_value}

        return obs, reward, done, info



    def render(self, mode="human"):
        print(f"Step {self.current_step}: Portfolio Value ${self.portfolio_value:.2f}")
