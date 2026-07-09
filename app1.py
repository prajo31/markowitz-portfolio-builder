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
    raw = yf.download(
        list(tickers),
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = list(tickers)

    prices = prices.dropna(how="all").ffill().dropna()

    return prices


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> pd.Series:
    raw = yf.download(
        list(tickers),
        period="5d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = list(tickers)

    return prices.ffill().iloc[-1]


@st.cache_data(ttl=86400, show_spinner=False)
def get_sector_data(tickers: tuple) -> dict:
    sector_map = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector", "Unknown")
        except Exception:
            sector_map[ticker] = "Unknown"

    return sector_map


# ---------------------------------------------------------------------
# Portfolio Math
# ---------------------------------------------------------------------


def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def optimize(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    objective="sharpe",
    target_return=None,
):
    n = len(mean_returns)
    x0 = np.full(n, 1.0 / n)

    bounds = [(0.0, max_w) for _ in range(n)]

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
    ]

    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: np.dot(w, mean_returns) - target_return,
            }
        )

    def objective_function(w):
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix, rf)

        if objective == "sharpe":
            return -sharpe
        if objective == "vol":
            return vol
        if objective == "return":
            return -ret

        return vol

    result = minimize(
        objective_function,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    return result


def efficient_frontier(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    n_points,
):
    min_vol_result = optimize(
        mean_returns,
        cov_matrix,
        rf,
        max_w,
        objective="vol",
    )

    max_return_result = optimize(
        mean_returns,
        cov_matrix,
        rf,
        max_w,
        objective="return",
    )

    if not min_vol_result.success or not max_return_result.success:
        return np.array([]), np.array([])

    min_ret = portfolio_stats(min_vol_result.x, mean_returns, cov_matrix, rf)[0]
    max_ret = portfolio_stats(max_return_result.x, mean_returns, cov_matrix, rf)[0]

    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        result = optimize(
            mean_returns,
            cov_matrix,
            rf,
            max_w,
            objective="vol",
            target_return=target,
        )

        if result.success:
            ret, vol, _ = portfolio_stats(result.x, mean_returns, cov_matrix, rf)
            frontier_rets.append(ret)
            frontier_vols.append(vol)

    return np.array(frontier_vols), np.array(frontier_rets)


def random_portfolios(
    mean_returns,
    cov_matrix,
    rf,
    max_w,
    n=3000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    k = len(mean_returns)

    weights_list = []
    attempts = 0
    max_attempts = n * 50

    while len(weights_list) < n and attempts < max_attempts:
        w = rng.dirichlet(np.ones(k))

        if np.all(w <= max_w + 1e-9):
            weights_list.append(w)

        attempts += 1

    if not weights_list:
        return np.array([]), np.array([]), np.array([])

    weights = np.array(weights_list)
    rets = weights @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights))
    sharpes = (rets - rf) / vols

    return vols, rets, sharpes


# ---------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------


def annualized_return(returns):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    cumulative_return = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS

    if years <= 0:
        return np.nan

    return cumulative_return ** (1 / years) - 1


def annualized_volatility(returns):
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf):
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)

    if ann_vol == 0 or pd.isna(ann_vol):
        return np.nan

    return (ann_ret - rf) / ann_vol


def max_drawdown_from_returns(returns):
    wealth = (1 + returns.dropna()).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return drawdown.min()


def drawdown_series_from_value(value_series):
    running_peak = value_series.cummax()
    return value_series / running_peak - 1


def sortino_ratio(returns, rf):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    downside = returns[returns < 0]

    if downside.empty:
        return np.nan

    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)

    if downside_vol == 0:
        return np.nan

    return (annualized_return(returns) - rf) / downside_vol


def value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return np.percentile(returns, 100 * (1 - confidence))


def conditional_value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


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
    prices = prices.dropna(how="any")

    if prices.empty:
        return pd.Series(dtype=float)

    returns = prices.pct_change().dropna()

    if frequency == "Monthly":
        rebalance_period = "ME"
    elif frequency == "Quarterly":
        rebalance_period = "QE"
    else:
        rebalance_period = "YE"

    rebalance_dates = returns.resample(rebalance_period).last().index
    rebalance_dates = returns.index.intersection(rebalance_dates)

    value = starting_value
    current_weights = target_weights.copy()
    values = []

    for date, row in returns.iterrows():
        if date in rebalance_dates:
            current_weights = target_weights.copy()

        daily_return = float(np.dot(current_weights, row.values))
        value *= 1 + daily_return

        current_weights = current_weights * (1 + row.values)
        current_weights = current_weights / current_weights.sum()

        values.append(value)

    return pd.Series(values, index=returns.index, name=frequency)


def simulate_buy_and_hold(
    prices,
    target_weights,
    starting_value,
):
    prices = prices.dropna(how="any")

    if prices.empty:
        return pd.Series(dtype=float)

    initial_prices = prices.iloc[0]
    shares = starting_value * target_weights / initial_prices
    values = prices @ shares

    return pd.Series(values, index=prices.index, name="Buy and Hold")


def monte_carlo_forecast(
    returns,
    starting_value,
    years,
    simulations,
    seed=42,
):
    rng = np.random.default_rng(seed)

    daily_returns = returns.dropna().values

    if len(daily_returns) == 0:
        return pd.DataFrame()

    days = years * TRADING_DAYS
    simulated = rng.choice(daily_returns, size=(days, simulations), replace=True)

    paths = starting_value * np.cumprod(1 + simulated, axis=0)

    return pd.DataFrame(paths)


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

    if len(mc_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=mc_vols,
                y=mc_rets,
                mode="markers",
                marker=dict(
                    size=5,
                    color=mc_sharpes,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe"),
                ),
                name="Random portfolios",
            )
        )

    if len(ef_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=ef_vols,
                y=ef_rets,
                mode="lines",
                line=dict(width=4),
                name="Efficient frontier",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[ms_vol],
            y=[ms_ret],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Max Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers",
            marker=dict(size=15, color="blue", symbol="diamond"),
            name="Min Volatility",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
    )

    return fig


def make_growth_chart(portfolio_growth, benchmark_growth):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_growth.index,
            y=portfolio_growth,
            mode="lines",
            name="Portfolio",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_growth.index,
            y=benchmark_growth,
            mode="lines",
            name="SPY",
        )
    )

    fig.update_layout(
        title="Backtested Growth",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

    return fig


def make_drawdown_chart(portfolio_growth, benchmark_growth):
    portfolio_dd = drawdown_series_from_value(portfolio_growth)
    benchmark_dd = drawdown_series_from_value(benchmark_growth)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_dd.index,
            y=portfolio_dd,
            mode="lines",
            name="Portfolio Drawdown",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_dd.index,
            y=benchmark_dd,
            mode="lines",
            name="SPY Drawdown",
        )
    )

    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
    )

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

    fig.update_layout(
        title="Sector Exposure",
        template="plotly_white",
    )

    return fig


def make_rebalance_chart(rebalance_results):
    fig = go.Figure()

    for name, series in rebalance_results.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Rebalancing Simulation",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

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

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["95th"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["5th"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="5th to 95th percentile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["Median"],
            mode="lines",
            name="Median",
        )
    )

    fig.update_layout(
        title="Monte Carlo Forecast",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

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

    weights_series = pd.Series(weights, index=tickers, name="Weight")
    dollars = weights_series * investment

    shares = dollars / last_prices.reindex(tickers)
    shares = shares.replace([np.inf, -np.inf], np.nan).fillna(0)

    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weights_series.values,
            "Dollar Allocation": dollars.values,
            "Last Price": last_prices.reindex(tickers).values,
            "Approx Shares": shares.values,
        }
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Expected Annual Return", f"{ret:.2%}")
    col2.metric("Annual Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.dataframe(
        table.style.format(
            {
                "Weight": "{:.2%}",
                "Dollar Allocation": "${:,.2f}",
                "Last Price": "${:,.2f}",
                "Approx Shares": "{:,.4f}",
            }
        ),
        use_container_width=True,
    )

    csv = table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download {name} allocation CSV",
        data=csv,
        file_name=f"{name.lower().replace(' ', '_')}_allocation.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------

if run:
    tickers = tuple(
        sorted({ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()})
    )

    if len(tickers) < 2:
        st.error("Enter at least two tickers.")
        st.stop()

    if max_weight * len(tickers) < 1:
        st.error(
            f"Max weight of {max_weight:.0%} is too low for {len(tickers)} assets. "
            "Increase max weight or add more tickers."
        )
        st.stop()

    with st.spinner("Downloading price data..."):
        prices = fetch_prices(tickers, years)

    if prices.empty:
        st.error("No price data was downloaded. Check your ticker symbols.")
        st.stop()

    missing = [ticker for ticker in tickers if ticker not in prices.columns]

    if missing:
        st.warning(f"These tickers were not found and will be ignored: {', '.join(missing)}")

    prices = prices.dropna(axis=1, how="all").ffill().dropna()

    tickers = tuple(prices.columns)

    if len(tickers) < 2:
        st.error("Need at least two valid tickers after downloading data.")
        st.stop()

    daily_returns = prices.pct_change().dropna()

    mean_returns = daily_returns.mean().values * TRADING_DAYS
    cov_matrix = daily_returns.cov().values * TRADING_DAYS

    max_sharpe_result = optimize(
        mean_returns,
        cov_matrix,
        rf_rate,
        max_weight,
        objective="sharpe",
    )

    min_vol_result = optimize(
        mean_returns,
        cov_matrix,
        rf_rate,
        max_weight,
        objective="vol",
    )

    if not max_sharpe_result.success or not min_vol_result.success:
        st.error("Optimization failed. Try increasing max weight or changing tickers.")
        st.stop()

    max_sharpe_weights = max_sharpe_result.x
    min_vol_weights = min_vol_result.x

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(
        max_sharpe_weights,
        mean_returns,
        cov_matrix,
        rf_rate,
    )

    mv_ret, mv_vol, mv_sharpe = portfolio_stats(
        min_vol_weights,
        mean_returns,
        cov_matrix,
        rf_rate,
    )

    with st.spinner("Building charts..."):
        ef_vols, ef_rets = efficient_frontier(
            mean_returns,
            cov_matrix,
            rf_rate,
            max_weight,
            n_frontier,
        )

        mc_vols, mc_rets, mc_sharpes = random_portfolios(
            mean_returns,
            cov_matrix,
            rf_rate,
            max_weight,
            n=3000,
        )

        last_prices = fetch_last_prices(tickers)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Optimization",
            "Allocations",
            "Backtest",
            "Risk Metrics",
            "Forecast",
        ]
    )

    with tab1:
        st.plotly_chart(
            make_frontier_chart(
                mc_vols,
                mc_rets,
                mc_sharpes,
                ef_vols,
                ef_rets,
                ms_vol,
                ms_ret,
                mv_vol,
                mv_ret,
            ),
            use_container_width=True,
        )

    with tab2:
        render_portfolio(
            "Maximum Sharpe Portfolio",
            max_sharpe_weights,
            tickers,
            last_prices,
            investment,
            ms_ret,
            ms_vol,
            ms_sharpe,
        )

        st.divider()

        render_portfolio(
            "Minimum Volatility Portfolio",
            min_vol_weights,
            tickers,
            last_prices,
            investment,
            mv_ret,
            mv_vol,
            mv_sharpe,
        )

        sector_map = get_sector_data(tickers)

        weight_series = pd.Series(max_sharpe_weights, index=tickers)
        sectors = pd.Series(sector_map)
        sector_weights = weight_series.groupby(sectors).sum().sort_values(ascending=False)

        st.plotly_chart(
            make_sector_chart(sector_weights),
            use_container_width=True,
        )

    with tab3:
        portfolio_returns = portfolio_return_series(daily_returns, max_sharpe_weights)
        portfolio_growth = growth_from_returns(portfolio_returns, investment)

        spy_prices = fetch_prices(("SPY",), years)

        if spy_prices.empty:
            st.warning("Could not download SPY benchmark data.")
        else:
            spy_returns = spy_prices.iloc[:, 0].pct_change().dropna()
            spy_growth = growth_from_returns(spy_returns, investment)

            common_index = portfolio_growth.index.intersection(spy_growth.index)
            portfolio_growth = portfolio_growth.loc[common_index]
            spy_growth = spy_growth.loc[common_index]

            st.plotly_chart(
                make_growth_chart(portfolio_growth, spy_growth),
                use_container_width=True,
            )

            st.plotly_chart(
                make_drawdown_chart(portfolio_growth, spy_growth),
                use_container_width=True,
            )

        rebalance_results = {
            "Buy and Hold": simulate_buy_and_hold(
                prices,
                max_sharpe_weights,
                investment,
            ),
            "Monthly": simulate_rebalanced_portfolio(
                prices,
                max_sharpe_weights,
                investment,
                "Monthly",
            ),
            "Quarterly": simulate_rebalanced_portfolio(
                prices,
                max_sharpe_weights,
                investment,
                "Quarterly",
            ),
            "Annual": simulate_rebalanced_portfolio(
                prices,
                max_sharpe_weights,
                investment,
                "Annual",
            ),
        }

        st.plotly_chart(
            make_rebalance_chart(rebalance_results),
            use_container_width=True,
        )

    with tab4:
        portfolio_returns = portfolio_return_series(daily_returns, max_sharpe_weights)

        spy_prices = fetch_prices(("SPY",), years)

        if spy_prices.empty:
            st.warning("Could not download SPY benchmark data.")
        else:
            spy_returns = spy_prices.iloc[:, 0].pct_change().dropna()

            common_index = portfolio_returns.index.intersection(spy_returns.index)

            metrics = metrics_table(
                portfolio_returns.loc[common_index],
                spy_returns.loc[common_index],
                rf_rate,
            )

            st.dataframe(
                metrics.style.format(
                    {
                        "Portfolio": "{:.2%}",
                        "SPY": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

    with tab5:
        portfolio_returns = portfolio_return_series(daily_returns, max_sharpe_weights)

        mc_paths = monte_carlo_forecast(
            portfolio_returns,
            investment,
            forecast_years,
            forecast_sims,
        )

        if mc_paths.empty:
            st.warning("Monte Carlo simulation could not run.")
        else:
            st.plotly_chart(
                make_monte_carlo_chart(mc_paths),
                use_container_width=True,
            )

            ending_values = mc_paths.iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Median Ending Value", f"${ending_values.median():,.2f}")
            col2.metric("5th Percentile", f"${ending_values.quantile(0.05):,.2f}")
            col3.metric("95th Percentile", f"${ending_values.quantile(0.95):,.2f}")

else:
    st.write(
        "Enter tickers in the sidebar and click **Build portfolio**. "
        "The app will optimize a portfolio, compare it with SPY, analyze risk, "
        "show sector exposure, simulate rebalancing, and run a Monte Carlo forecast."
    )
