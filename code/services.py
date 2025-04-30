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
    # Step 1: Fetch full data
    full_data = fetch_market_data(assets, time_range)

    # Step 2: Filter data based on selected timeframe
    end_date = full_data.index[-1]
    if time_range == "7 Days":
        start_date = end_date - pd.Timedelta(days=7)
    elif time_range == "1 Month":
        start_date = end_date - pd.DateOffset(months=1)
    elif time_range == "6 Months":
        start_date = end_date - pd.DateOffset(months=6)
    else:  # "1 Year"
        start_date = end_date - pd.DateOffset(years=1)

    data = full_data[full_data.index >= start_date]

    # Step 3: Calculate returns
    returns = data.pct_change().dropna()

    # Step 4: Create environment and model
    env = CustomPortfolioEnv(data, risk_profile=risk_tolerance, initial_investment=investment_amount)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 5: Run the trained model
    obs, info = env.reset()
    portfolio_values = [env.portfolio_value]
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(env.portfolio_value)

    # Step 6: RL Allocation chart
    weights = np.clip(action, 0, 1)
    weights /= np.sum(weights) if np.sum(weights) > 0 else len(weights)

    allocation_chart = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4
    )])
    allocation_chart.update_layout(title="Optimized Portfolio Allocation")

    # Step 7: Passive equal-weight portfolio
    # Passive baseline performance (force same starting value)
    equal_weights = np.ones(len(assets)) / len(assets)
    passive_returns = (data.pct_change().dropna() + 1).cumprod()
    passive_growth = passive_returns.dot(equal_weights)
    # Prepend initial investment manually
    passive_values = pd.concat([
        pd.Series([1.0], index=[passive_growth.index[0]]),
        passive_growth
    ]).cumprod() * investment_amount

    # Step 8: Performance comparison chart
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

    # Step 9: Allocation CSV
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
