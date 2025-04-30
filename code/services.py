import numpy as np
import pandas as pd
import plotly.graph_objects as go
from stable_baselines3 import PPO
from portfolio_env import CustomPortfolioEnv

import yfinance as yf
import datetime

def fetch_market_data(assets, time_range):
    end = datetime.datetime.today()
    days_lookup = {
        "7 Days": 7,
        "1 Month": 30,
        "6 Months": 180,
        "1 Year": 365
    }
    days = days_lookup.get(time_range, 365)
    start = end - datetime.timedelta(days=days)

    data = yf.download(assets, start=start, end=end, auto_adjust=True, group_by='ticker')

    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers: extract "Close" from each
        close_prices = pd.concat([data[ticker]["Close"] for ticker in assets], axis=1)
        close_prices.columns = assets
    else:
        # Single ticker: flat DataFrame
        close_prices = pd.DataFrame(data["Close"])
        close_prices.columns = assets

    return close_prices.dropna()

def optimize_portfolio(risk_tolerance, investment_amount, assets, time_range):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from stable_baselines3 import PPO
    from portfolio_env import CustomPortfolioEnv
    
    # Step 1: Fetch historical data
    data = fetch_market_data(assets, time_range)
    returns = data.pct_change().dropna()

    # Step 2: Set up RL environment
    env = CustomPortfolioEnv(data, risk_profile=risk_tolerance, initial_investment=investment_amount)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 3: Simulate RL agent performance
    obs, info = env.reset()
    rl_values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        rl_values.append(env.portfolio_value)

    # Step 4: Final portfolio weights (for pie chart)
    weights = np.clip(action, 0, 1)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
    allocation_chart = go.Figure(data=[go.Pie(labels=assets, values=weights, hole=0.4)])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Step 5: Passive equal-weight benchmark (start from same $)
    equal_weights = np.ones(len(assets)) / len(assets)
    passive_returns = returns.dot(equal_weights)
    passive_growth = (1 + passive_returns).cumprod()
    passive_growth = (passive_growth / passive_growth.iloc[0]) * investment_amount  # Force same start
    passive_dates = passive_growth.index

    # Step 6: Performance comparison plot
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

    # Step 7: CSV Export
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
