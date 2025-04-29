import streamlit as st

st.title("Interactive Portfolio Optimization")

risk = st.selectbox("Select Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
amount = st.number_input("Max Investment Amount", min_value=1000)
assets = st.multiselect("Choose Assets", ["AAPL", "MSFT", "GOOGL", "BTC", "BND"])

if st.button("Optimize Portfolio"):
    # call service logic and render chart
    pass

