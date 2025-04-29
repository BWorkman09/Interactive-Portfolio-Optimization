import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from portfolio_env import CustomPortfolioEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go
import gymnasium as gym


def fetch_market_data(assets):
    import yfinance as yf
    import pandas as pd
    import datetime

    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)

    data = yf.download(assets, start=start, end=end, group_by='ticker', auto_adjust=True)

    # Handle single vs multi asset requests
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = pd.DataFrame({ticker: data[ticker]["Close"] for ticker in assets})
    else:
        adj_close = pd.DataFrame(data["Close"])
        adj_close.columns = assets  # give it the same structure for consistency

    return adj_close.dropna()

def optimize_portfolio(risk_tolerance, investment_amount, assets):
    data = fetch_market_data(assets)
    returns = data.pct_change().dropna()

    env = CustomPortfolioEnv(returns, risk_profile=risk_tolerance)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # 👈 critical fix

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 2: Calculate returns
    returns = data.pct_change().dropna()

    # Step 3: Set risk multiplier
    risk_levels = {
        "Conservative": 0.5,
        "Moderate": 1.0,
        "Aggressive": 1.5
    }
    risk_multiplier = risk_levels.get(risk_tolerance, 1.0)

    # Step 4: Set up environment
    env = CustomPortfolioEnv(returns, risk_profile=risk_tolerance)

    # Step 5: Train PPO model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 6: Run the trained agent
    obs, _ = env.reset()
    terminated = False
    truncated = False
    portfolio_values = [env.portfolio_value]

    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)  # or obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    # Step 7: Build allocation chart
    weights = np.clip(action, 0, 1)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

    allocation_chart = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4
    )])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Step 8: Build performance chart (with real dates)
    performance_chart = go.Figure()
    performance_chart.add_trace(go.Scatter(
        x=info["dates"],
        y=info["history"],
        mode="lines+markers",
        name="Portfolio Value"
    ))
    performance_chart.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)"
    )

    # Step 9: Prepare Allocations CSV
    allocations_df = pd.DataFrame({
        "Asset": assets,
        "Allocation (%)": np.round(weights * 100, 2)
    })
    allocations_csv = allocations_df.to_csv(index=False)

    # Step 10: Return everything
    return {
        "allocation_chart": allocation_chart,
        "performance_chart": performance_chart,
        "allocations_csv": allocations_csv
    }
