import gymnasium as gym
import numpy as np

class CustomPortfolioEnv(gym.Env):
    def __init__(self, data, risk_profile="Moderate", initial_investment=1000):
        super(CustomPortfolioEnv, self).__init__()

        self.assets = data.columns.tolist()
        self.num_assets = len(self.assets)
        self.data = data

        self.dates = data.index
        self.returns = data.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna().values

        self.current_step = 0
        self.initial_investment = initial_investment
        self.portfolio_value = self.initial_investment
        self.history = [self.portfolio_value]

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)

        self.risk_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}.get(risk_profile, 1.0)


    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.initial_investment  # ✅ dynamic
        self.history = [self.portfolio_value]
        return self.returns[self.current_step], {}  # if using gymnasium or SB3 compatibility



    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(weights, asset_returns)
        concentration_penalty = np.sum(np.square(weights))
        diversification_bonus = 1 - concentration_penalty

        adjusted_return = (portfolio_return * self.risk_multiplier) + (0.001 * diversification_bonus)
        adjusted_return = np.clip(adjusted_return, -0.99, 0.99)

        self.portfolio_value *= (1 + adjusted_return)

        self.current_step += 1

        # REQUIRED: Separate 'terminated' and 'truncated'
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False  # we are not using time limits here

        obs = self.returns[self.current_step] if not terminated else np.zeros(self.num_assets)
        self.history.append(self.portfolio_value)

        info = {
            "portfolio_value": self.portfolio_value,
            "history": self.history,
            "dates": self.dates[1:self.current_step+2]
        }

        return obs, adjusted_return, terminated, truncated, info


    def render(self, mode="human"):
        print(f"Step {self.current_step}: Portfolio Value ${self.portfolio_value:.2f}")
