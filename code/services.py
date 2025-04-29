import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO
from portfolio_env import CustomPortfolioEnv
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(assets, start="2019-01-01", end="2024-01-01"):
    data = yf.download(assets, start=start, end=end)["Close"]
    return data.dropna()


def train_rl_model(data, risk_profile):
    model = PPO("MlpPolicy", data, verbose=0)
    model.learn(total_timesteps=10000)
    return model


def optimize_portfolio(risk_tolerance, investment_amount, assets):
    # Step 1: Fetch data
    data = fetch_market_data(assets)

    # Step 2: Calculate returns
    returns = data.pct_change().dropna()

    # Step 3: Set up environment
    # (pass risk_tolerance STRING directly)
    env = CustomPortfolioEnv(returns, risk_tolerance)

    # Step 4: Train PPO model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 5: Run the trained agent
    obs = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    # Step 6: Build allocation chart
    weights = np.clip(action, 0, 1)
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= np.sum(weights)

    allocation_chart = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4
    )])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Step 7: Build performance chart
    performance_chart = go.Figure()
    performance_chart.add_trace(go.Scatter(
        y=portfolio_values,
        mode="lines",
        name="Portfolio Value"
    ))
    performance_chart.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Step",
        yaxis_title="Portfolio Value ($)"
    )

    # Step 8: Prepare Allocations CSV
    allocations_df = pd.DataFrame({
        "Asset": assets,
        "Allocation (%)": np.round(weights * 100, 2)
    })
    allocations_csv = allocations_df.to_csv(index=False)

    # Step 9: Return everything
    return {
        "allocation_chart": allocation_chart,
        "performance_chart": performance_chart,
        "allocations_csv": allocations_csv
    }
