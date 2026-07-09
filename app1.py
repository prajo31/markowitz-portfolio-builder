"""
Markowitz Portfolio Builder

Run with:
streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

TRADING_DAYS = 252

st.set_page_config(
    page_title="Markowitz Portfolio Builder",
    layout="wide",
)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("Portfolio settings")

tickers_input = st.sidebar.text_input(
    "Tickers comma-separated",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    help="Examples: AAPL, MSFT, VTI, SPY, QQQ, BTC-USD",
)

years = st.sidebar.slider(
    "Years of price history",
    min_value=1,
    max_value=10,
    value=3,
)

rf_rate = (
    st.sidebar.number_input(
        "Risk-free rate annual %",
        value=4.0,
        step=0.25,
    )
    / 100.0
)

max_weight = (
    st.sidebar.slider(
        "Max weight per asset %",
        min_value=10,
        max_value=100,
        value=100,
        help="Cap any single holding. 100% means unconstrained long-only.",
    )
    / 100.0
)

investment = st.sidebar.number_input(
    "Amount to invest $",
    value=10000.0,
    min_value=0.0,
    step=500.0,
)

n_frontier = st.sidebar.slider(
    "Frontier resolution points",
    min_value=20,
    max_value=100,
    value=50,
)

forecast_years = st.sidebar.slider(
    "Monte Carlo forecast years",
    min_value=1,
    max_value=10,
    value=5,
)

forecast_sims = st.sidebar.slider(
    "Monte Carlo simulations",
    min_value=500,
    max_value=10000,
    value=3000,
    step=500,
)

run = st.sidebar.button(
    "Build portfolio",
    type="primary",
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("Markowitz Portfolio Builder")

st.caption(
    "Live Yahoo Finance prices • Markowitz optimization • Backtesting • "
    "Risk analytics • Sector exposure • Rebalancing simulation"
)

# ---------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple, years: int) -> pd.DataFrame:
    pass


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> pd.Series:
    pass


@st.cache_data(ttl=86400, show_spinner=False)
def get_sector_data(tickers: tuple) -> dict:
    sector_map = {}
    return sector_map


# ---------------------------------------------------------------------
# Portfolio Math
# ---------------------------------------------------------------------


def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    pass


def optimize(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    objective="sharpe",
    target_return=None,
):
    pass


def efficient_frontier(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    n_points,
):
    pass


def random_portfolios(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    n=3000,
    seed=42,
):
    pass


# ---------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------


def annualized_return(returns):
    pass


def annualized_volatility(returns):
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf):
    pass


def max_drawdown_from_returns(returns):
    pass


def drawdown_series_from_value(value_series):
    running_peak = value_series.cummax()
    return value_series / running_peak - 1


def sortino_ratio(returns, rf):
    pass


def value_at_risk(returns, confidence=0.95):
    pass


def conditional_value_at_risk(returns, confidence=0.95):
    pass


def metrics_table(portfolio_returns, benchmark_returns, rf):
    data = {
        "Metric": [
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Daily VaR 95%",
            "Daily CVaR 95%",
        ],
        "Portfolio": [
            annualized_return(portfolio_returns),
            annualized_volatility(portfolio_returns),
            sharpe_ratio(portfolio_returns, rf),
            sortino_ratio(portfolio_returns, rf),
            max_drawdown_from_returns(portfolio_returns),
            value_at_risk(portfolio_returns),
            conditional_value_at_risk(portfolio_returns),
        ],
        "SPY": [
            annualized_return(benchmark_returns),
            annualized_volatility(benchmark_returns),
            sharpe_ratio(benchmark_returns, rf),
            sortino_ratio(benchmark_returns, rf),
            max_drawdown_from_returns(benchmark_returns),
            value_at_risk(benchmark_returns),
            conditional_value_at_risk(benchmark_returns),
        ],
    }

    return pd.DataFrame(data)


# ---------------------------------------------------------------------
# Backtest / Simulation
# ---------------------------------------------------------------------


def portfolio_return_series(daily_returns, weights):
    return pd.Series(
        daily_returns.values @ weights,
        index=daily_returns.index,
        name="Portfolio",
    )


def growth_from_returns(returns, starting_value):
    return (1 + returns).cumprod() * starting_value


def simulate_rebalanced_portfolio(
    prices,
    target_weights,
    starting_value,
    frequency,
):
    pass


def simulate_buy_and_hold(
    prices,
    target_weights,
    starting_value,
):
    prices = prices.dropna(how="any")
    pass


def monte_carlo_forecast(
    returns,
    starting_value,
    years,
    simulations,
    seed=42,
):
    pass


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------


def make_frontier_chart(
    mc_vols,
    mc_rets,
    mc_sharpes,
    ef_vols,
    ef_rets,
    ms_vol,
    ms_ret,
    mv_vol,
    mv_ret,
):
    fig = go.Figure()
    return fig


def make_growth_chart(portfolio_growth, benchmark_growth):
    fig = go.Figure()
    return fig


def make_drawdown_chart(portfolio_growth, benchmark_growth):
    portfolio_dd = drawdown_series_from_value(portfolio_growth)
    benchmark_dd = drawdown_series_from_value(benchmark_growth)

    fig = go.Figure()
    return fig


def make_sector_chart(sector_weights):
    fig = go.Figure(
        go.Treemap(
            labels=sector_weights.index,
            parents=[""] * len(sector_weights),
            values=sector_weights.values,
            textinfo="label+percent root",
        )
    )
    return fig


def make_rebalance_chart(rebalance_results):
    fig = go.Figure()
    return fig


def make_monte_carlo_chart(mc_paths):
    percentiles = pd.DataFrame(
        {
            "5th": mc_paths.quantile(0.05, axis=1),
            "25th": mc_paths.quantile(0.25, axis=1),
            "Median": mc_paths.quantile(0.50, axis=1),
            "75th": mc_paths.quantile(0.75, axis=1),
            "95th": mc_paths.quantile(0.95, axis=1),
        }
    )

    fig = go.Figure()
    return fig


# ---------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------


def render_portfolio(
    name,
    weights,
    tickers,
    last_prices,
    investment,
    ret,
    vol,
    sharpe,
):
    st.subheader(name)


# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------


if run:
    pass
else:
    st.write(
        "Enter tickers in the sidebar and click **Build portfolio**. "
        "The app will optimize a portfolio, compare it with SPY, analyze risk, "
        "show sector exposure, simulate rebalancing, and run a Monte Carlo forecast."
    )
