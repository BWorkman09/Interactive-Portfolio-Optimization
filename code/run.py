import streamlit as st
from code.services import optimize_portfolio  # placeholder for future RL logic

# --- Page Config ---
st.set_page_config(page_title="Interactive Portfolio Optimizer", layout="wide")

# --- Title & Description ---
st.title("📈 Interactive Portfolio Optimization")
st.markdown("Use reinforcement learning to optimize your asset allocation based on your risk preferences and goals.")

# --- Sidebar for User Inputs ---
st.sidebar.header("User Preferences")

risk_tolerance = st.sidebar.selectbox(
    "Select Risk Tolerance:",
    options=["Conservative", "Moderate", "Aggressive"]
)

investment_amount = st.sidebar.number_input(
    "Maximum Investment Amount ($):",
    min_value=1000, max_value=1_000_000, value=10000, step=500
)

available_assets = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "BTC-USD", "ETH-USD", "BND"]
selected_assets = st.sidebar.multiselect(
    "Select Assets to Include:",
    options=available_assets,
    default=["AAPL", "MSFT", "BND"]
)

# --- Submit Button ---
if st.sidebar.button("Optimize Portfolio"):
    with st.spinner("Running optimization..."):
        # Placeholder for real logic
        portfolio = optimize_portfolio(risk_tolerance, investment_amount, selected_assets)

        st.success("Optimization complete!")

        # --- Placeholder Chart ---
        st.subheader("Portfolio Allocation (Example)")
        st.plotly_chart(portfolio["allocation_chart"])  # expects a Plotly figure

        st.subheader("Performance vs. Benchmark")
        st.plotly_chart(portfolio["performance_chart"])  # expects a Plotly figure

else:
    st.info("Adjust your preferences in the sidebar and click 'Optimize Portfolio'.")


