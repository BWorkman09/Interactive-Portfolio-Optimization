import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from stable_baselines3 import PPO
from .data import fetch_market_data


def fetch_market_data(assets, start="2019-01-01", end="2024-01-01"):
    data = yf.download(assets, start=start, end=end)["Adj Close"]
    return data.dropna()


def train_rl_model(data, risk_profile):
    model = PPO("MlpPolicy", data, verbose=0)
    model.learn(total_timesteps=10000)
    return model


def optimize_portfolio(risk, amount, assets):
    # Step 1: Load market data
    data = fetch_market_data(assets)

    # Step 2: Create environment
    env = CustomPortfolioEnv(data, risk)  # You will define this class

    # Step 3: Train PPO model
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    # Step 4: Run inference
    obs = env.reset()
    done = False
    portfolio_values = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info.get("portfolio_value", 0))

    # Step 5: Visualization — Allocation chart (dummy equal weights for now)
    allocations = [1 / len(assets)] * len(assets)
    fig_alloc = go.Figure(data=[go.Pie(labels=assets, values=allocations)])
    fig_alloc.update_layout(title="Optimized Portfolio Allocation")

    # Performance chart (dummy data)
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=list(range(len(portfolio_values))),
        y=portfolio_values,
        mode='lines+markers',
        name='RL Portfolio Value'
    ))
    fig_perf.update_layout(title="Portfolio Performance", xaxis_title="Step", yaxis_title="Value ($)")

    return {
        "allocation_chart": fig_alloc,
        "performance_chart": fig_perf
    }


def train_rl_model(data, risk_profile):
    env = CustomPortfolioEnv(data, risk_profile)  # you'll define this
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    return model, env

def run_rl_inference(model, env):
    obs = env.reset()
    done = False
    portfolio_history = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_history.append(info["portfolio_value"])  # or weights

    return portfolio_history
