import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from portfolio_env import CustomPortfolioEnv
from stable_baselines3 import PPO
import plotly.graph_objects as go


def fetch_market_data(assets, time_range):
    import yfinance as yf
    import pandas as pd
    import datetime

    end = datetime.datetime.today()

    # Map time_range to start date
    if time_range == "7 Days":
        start = end - datetime.timedelta(days=7)
    elif time_range == "1 Month":
        start = end - datetime.timedelta(days=30)
    elif time_range == "6 Months":
        start = end - datetime.timedelta(days=182)
    else:  # "1 Year"
        start = end - datetime.timedelta(days=365)

    data = yf.download(assets, start=start, end=end, group_by='ticker', auto_adjust=True)

    # Handle single vs multi-asset format
    if isinstance(data.columns, pd.MultiIndex):
        close_data = pd.DataFrame({ticker: data[ticker]["Close"] for ticker in assets})
    else:
        close_data = pd.DataFrame(data["Close"])
        close_data.columns = assets

    return close_data.dropna()


def optimize_portfolio(risk_tolerance, investment_amount, assets, time_range):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from stable_baselines3 import PPO
    from portfolio_env import CustomPortfolioEnv
    from services import fetch_market_data

    # Step 1: Fetch historical data based on time_range
    data = fetch_market_data(assets, time_range)
    returns = data.pct_change().dropna()

    # Step 2: Initialize environment with investment amount
    env = CustomPortfolioEnv(data, risk_profile=risk_tolerance, initial_investment=investment_amount)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 3: Run RL agent
    obs, info = env.reset()
    rl_values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        rl_values.append(env.portfolio_value)

    # Step 4: Generate allocation pie chart
    weights = np.clip(action, 0, 1)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
    allocation_chart = go.Figure(data=[go.Pie(labels=assets, values=weights, hole=0.4)])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Step 5: Passive equal-weight benchmark (same initial investment)
    equal_weights = np.ones(len(assets)) / len(assets)
    passive_returns = returns.dot(equal_weights)
    passive_growth = (1 + passive_returns).cumprod() * investment_amount
    passive_dates = returns.index

    # Step 6: Build performance comparison chart
    performance_chart = go.Figure()
    performance_chart.add_trace(go.Scatter(
        x=info["dates"],
        y=rl_values,
        mode="lines+markers",
        name="RL Portfolio Value"
    ))
    performance_chart.add_trace(go.Scatter(
        x=passive_dates,
        y=passive_growth,
        mode="lines",
        name="Passive Equal-Weight Portfolio",
        line=dict(dash='dot')
    ))
    performance_chart.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)"
    )

    # Step 7: Allocation CSV export
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
