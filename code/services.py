import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from portfolio_env import CustomPortfolioEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go


def fetch_market_data(assets, time_range):
    end = datetime.datetime.today()
    if time_range == "7 Days":
        start = end - datetime.timedelta(days=7)
    elif time_range == "1 Month":
        start = end - datetime.timedelta(days=30)
    elif time_range == "6 Months":
        start = end - datetime.timedelta(days=182)
    else:  # "1 Year"
        start = end - datetime.timedelta(days=365)

    data = yf.download(assets, start=start, end=end, group_by='ticker', auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        adj_close = pd.DataFrame({ticker: data[ticker]["Close"] for ticker in assets})
    else:
        adj_close = pd.DataFrame(data["Close"])
        adj_close.columns = assets

    return adj_close.dropna()


def optimize_portfolio(risk_tolerance, investment_amount, assets, time_range):
    data = fetch_market_data(assets, time_range)
    returns = data.pct_change().dropna()

    env = CustomPortfolioEnv(returns, risk_profile=risk_tolerance, initial_investment=investment_amount)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(env.portfolio_value)

    # Build allocation chart
    weights = np.clip(action, 0, 1)
    weights /= np.sum(weights) if np.sum(weights) > 0 else len(weights)

    allocation_chart = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4
    )])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Passive baseline performance
    equal_weights = np.ones(len(assets)) / len(assets)
    passive_values = (data.pct_change().dropna() + 1).cumprod().dot(equal_weights) * investment_amount

    # RL Portfolio Performance chart
    performance_chart = go.Figure()
    performance_chart.add_trace(go.Scatter(
        x=env.dates[1:len(portfolio_values)+1],
        y=portfolio_values,
        mode="lines+markers",
        name="RL Portfolio Value"
    ))
    performance_chart.add_trace(go.Scatter(
        x=passive_values.index,
        y=passive_values.values,
        mode="lines",
        name="Passive Equal-Weight Portfolio",
        line=dict(dash='dot')
    ))
    performance_chart.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)"
    )

    # Prepare CSV
    allocations_df = pd.DataFrame({
        "Asset": assets,
        "Allocation (%)": np.round(weights * 100, 2)
    })
    allocations_csv = allocations_df.to_csv(index=False)

    return {
        "allocation_chart": allocation_chart,
        "performance_chart": performance_chart,
        "allocations_csv": allocations_csv
    }
