import streamlit as st
from services import optimize_portfolio

# --- Page Config ---
st.set_page_config(page_title="Interactive Portfolio Optimizer", layout="wide")

# --- Title & Description ---
st.title("📈 Interactive Portfolio Optimization")
st.markdown("Use reinforcement learning to optimize your asset allocation based on your risk preferences and goals.")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Settings")

# Ticker Input
ticker_input = st.sidebar.text_input(
    "Enter stock tickers (comma-separated):",
    value="AAPL,MSFT,GOOGL"
)
selected_assets = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

# Risk Tolerance
risk_tolerance = st.sidebar.selectbox(
    "Select your risk tolerance:",
    options=["Conservative", "Moderate", "Aggressive"]
)

# Investment Amount
investment_amount = st.sidebar.slider(
    "Investment Amount ($)",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

# Performance Timeframe
time_range = st.sidebar.selectbox(
    "Performance Timeframe",
    ["7 Days", "1 Month", "6 Months", "1 Year"]
)

# --- Submit Button ---
if st.sidebar.button("Optimize Portfolio"):
    with st.spinner("Running optimization..."):
        portfolio = optimize_portfolio(risk_tolerance, investment_amount, selected_assets, time_range)

    st.success("Optimization complete!")

    st.subheader("Portfolio Allocation")
    st.plotly_chart(portfolio["allocation_chart"])

    st.download_button(
        label="Download Portfolio Allocations as CSV",
        data=portfolio["allocations_csv"],
        file_name="portfolio_allocations.csv",
        mime="text/csv",
    )

    st.subheader("Performance vs. Benchmark")
    st.plotly_chart(portfolio["performance_chart"])

else:
    st.info("Adjust your preferences in the sidebar and click 'Optimize Portfolio'.")


#streamlit run code/run.py

