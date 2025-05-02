Interactive Portfolio Optimization

An interactive web app that uses Reinforcement Learning (RL) to optimize stock portfolios based on user-defined risk tolerance and investment horizon.
Features

    Enter custom stock tickers (e.g., AAPL, MSFT)

    Choose risk tolerance: Conservative, Moderate, Aggressive

    Select investment amount ($1K–$100K)

    Set performance timeframe: 7 Days, 1 Month, 6 Months, 1 Year

    Compare RL-optimized vs passive equal-weight portfolios

    Download allocation breakdown as CSV

Tech Stack

    Python, Streamlit, Stable-Baselines3 (PPO)

    Plotly for charts

    yfinance for stock data

How It Works

    User enters stocks and preferences.

    Historical market data is fetched.

    PPO (Proximal Policy Optimization) model learns asset allocations.

    Portfolio value is tracked over the selected timeframe.

    Results are plotted against a passive benchmark.

Risk Tolerance Logic

    Adjusts RL reward sensitivity using multipliers:

        Conservative → lower risk, smoother curve

        Aggressive → higher risk, higher potential reward

Run the App

streamlit run code/run.py
