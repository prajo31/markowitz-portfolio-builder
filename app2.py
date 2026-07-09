"""
Markowitz Portfolio Builder

Run with:
streamlit run app.py
"""

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

optimization_mode = st.sidebar.selectbox(
    "Optimization goal",
    [
        "Maximum Sharpe",
        "Minimum Volatility",
        "Target Return",
    ],
)

target_return = (
    st.sidebar.number_input(
        "Target annual return %",
        min_value=1.0,
        max_value=50.0,
        value=14.0,
        step=0.5,
    )
    / 100.0
)

n_frontier = st.sidebar.slider(
    "Frontier resolution points",
    min_value=20,
    max_value=100,
    value=50,
)

st.sidebar.divider()

st.sidebar.subheader("Rebalancing Alerts")

rebalance_threshold = (
    st.sidebar.slider(
        "Rebalance Threshold (%)",
        min_value=1,
        max_value=20,
        value=5,
    )
    / 100.0
)

st.sidebar.subheader("Retirement planner")

retirement_years = st.sidebar.slider(
    "Planning years",
    min_value=1,
    max_value=40,
    value=20,
)

monthly_contribution = st.sidebar.number_input(
    "Monthly contribution $",
    value=1000.0,
    min_value=0.0,
    step=100.0,
)

target_wealth = st.sidebar.number_input(
    "Target wealth $",
    value=1000000.0,
    min_value=0.0,
    step=50000.0,
)

forecast_sims = st.sidebar.slider(
    "Monte Carlo simulations",
    min_value=500,
    max_value=10000,
    value=3000,
    step=500,
)

# ---------------------------------------------------------------------
# Rebalancing Alerts
# ---------------------------------------------------------------------


def generate_rebalancing_alerts(
    prices,
    target_weights,
    tickers,
    investment,
):
    latest_prices = prices.iloc[-1]

    initial_prices = prices.iloc[0]

    shares = (
        investment
        * target_weights
        / initial_prices.values
    )

    current_values = (
        shares
        * latest_prices.values
    )

    total_value = current_values.sum()

    current_weights = (
        current_values
        / total_value
    )

    df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Target Weight": target_weights,
            "Current Weight": current_weights,
        }
    )

    df["Deviation"] = (
        df["Current Weight"]
        - df["Target Weight"]
    )

    target_value = total_value * df["Target Weight"]

    df["Trade Amount"] = (
        target_value
        - current_values
    )

    return df
run = st.sidebar.button(
    "Build portfolio",
    type="primary",
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("Markowitz Portfolio Builder")

st.caption(
    "Live Yahoo Finance prices • Markowitz optimization • Target return planning • "
    "Retirement projection • Goal probability • AI-style portfolio commentary"
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
    years_count = len(returns) / TRADING_DAYS

    if years_count <= 0:
        return np.nan

    return cumulative_return ** (1 / years_count) - 1


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


def monte_carlo_forecast_with_contributions(
    returns,
    starting_value,
    monthly_contribution,
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

    values = np.zeros((days, simulations))
    values[0, :] = starting_value * (1 + simulated[0, :])

    monthly_day_interval = 21

    for day in range(1, days):
        values[day, :] = values[day - 1, :] * (1 + simulated[day, :])

        if day % monthly_day_interval == 0:
            values[day, :] += monthly_contribution

    return pd.DataFrame(values)


def goal_probability(mc_paths, target_wealth):
    if mc_paths.empty:
        return np.nan

    ending_values = mc_paths.iloc[-1]
    return float((ending_values >= target_wealth).mean())


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------


def make_frontier_chart(
    mc_vols,
    mc_rets,
    mc_sharpes,
    ef_vols,
    ef_rets,
    selected_vol,
    selected_ret,
    selected_name,
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
            marker=dict(size=13, color="red", symbol="star"),
            name="Max Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers",
            marker=dict(size=13, color="blue", symbol="diamond"),
            name="Min Volatility",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[selected_vol],
            y=[selected_ret],
            mode="markers",
            marker=dict(size=17, color="orange", symbol="circle"),
            name=selected_name,
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


def make_monte_carlo_chart(mc_paths, target_wealth):
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

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=[target_wealth] * len(percentiles),
            mode="lines",
            line=dict(dash="dash"),
            name="Target wealth",
        )
    )

    fig.update_layout(
        title="Retirement Monte Carlo Forecast",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------


def generate_portfolio_commentary(
    selected_name,
    tickers,
    weights,
    expected_return,
    volatility,
    sharpe,
    sector_weights,
    goal_prob,
    target_wealth,
    retirement_years,
    monthly_contribution,
):
    weights_series = pd.Series(weights, index=tickers)
    top_holdings = weights_series.sort_values(ascending=False).head(3)

    largest_weight = top_holdings.iloc[0]
    largest_ticker = top_holdings.index[0]

    dominant_sector = "Unknown"
    dominant_sector_weight = 0

    if not sector_weights.empty:
        dominant_sector = sector_weights.index[0]
        dominant_sector_weight = sector_weights.iloc[0]

    risk_label = "moderate"

    if volatility < 0.12:
        risk_label = "relatively conservative"
    elif volatility > 0.25:
        risk_label = "aggressive"

    goal_label = "uncertain"

    if goal_prob >= 0.75:
        goal_label = "strong"
    elif goal_prob >= 0.50:
        goal_label = "moderate"
    else:
        goal_label = "weak"

    commentary = f"""
### AI-Style Portfolio Commentary

The selected portfolio is the **{selected_name}**. It has an expected annual return of **{expected_return:.2%}**, annualized volatility of **{volatility:.2%}**, and a Sharpe ratio of **{sharpe:.2f}**.

From a risk perspective, this looks like a **{risk_label}** portfolio. The largest position is **{largest_ticker}** at **{largest_weight:.2%}**, which means this holding has the biggest influence on performance.

The top holdings are:

"""

    for ticker, weight in top_holdings.items():
        commentary += f"- **{ticker}**: {weight:.2%}\n"

    commentary += f"""

Sector concentration is led by **{dominant_sector}**, representing approximately **{dominant_sector_weight:.2%}** of the portfolio. If this percentage is high, the portfolio may be more sensitive to sector-specific shocks.

For the retirement goal, with a target of **${target_wealth:,.0f}** over **{retirement_years} years** and monthly contributions of **${monthly_contribution:,.0f}**, the Monte Carlo simulation estimates a **{goal_prob:.1%}** probability of reaching the goal. This is a **{goal_label}** probability of success under the historical-return assumptions used by the model.

Important note: this analysis is based on historical price behavior and simulated outcomes. It should be treated as a planning tool, not as a guarantee of future returns.
"""

    return commentary


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

    aligned_prices = last_prices.reindex(tickers)
    shares = dollars / aligned_prices
    shares = shares.replace([np.inf, -np.inf], np.nan).fillna(0)

    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weights_series.values,
            "Dollar Allocation": dollars.values,
            "Last Price": aligned_prices.values,
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
        sorted(
            {
                ticker.strip().upper()
                for ticker in tickers_input.split(",")
                if ticker.strip()
            }
        )
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

    target_result = optimize(
        mean_returns,
        cov_matrix,
        rf_rate,
        max_weight,
        objective="vol",
        target_return=target_return,
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

    if optimization_mode == "Maximum Sharpe":
        selected_name = "Maximum Sharpe Portfolio"
        selected_weights = max_sharpe_weights
        selected_ret = ms_ret
        selected_vol = ms_vol
        selected_sharpe = ms_sharpe

    elif optimization_mode == "Minimum Volatility":
        selected_name = "Minimum Volatility Portfolio"
        selected_weights = min_vol_weights
        selected_ret = mv_ret
        selected_vol = mv_vol
        selected_sharpe = mv_sharpe

    else:
        if not target_result.success:
            st.error(
                f"Could not find a feasible portfolio for target return {target_return:.2%}. "
                "Try lowering the target return or increasing max weight."
            )
            st.stop()

        selected_name = f"Target Return Portfolio ({target_return:.2%})"
        selected_weights = target_result.x
        selected_ret, selected_vol, selected_sharpe = portfolio_stats(
            selected_weights,
            mean_returns,
            cov_matrix,
            rf_rate,
        )

    with st.spinner("Building charts and simulations..."):
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

        selected_returns = portfolio_return_series(daily_returns, selected_weights)

        retirement_paths = monte_carlo_forecast_with_contributions(
            selected_returns,
            investment,
            monthly_contribution,
            retirement_years,
            forecast_sims,
        )

        prob_success = goal_probability(retirement_paths, target_wealth)

        sector_map = get_sector_data(tickers)
        selected_weight_series = pd.Series(selected_weights, index=tickers)
        sectors = pd.Series(sector_map)
        sector_weights = (
            selected_weight_series.groupby(sectors)
            .sum()
            .sort_values(ascending=False)
        )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Optimization",
        "Allocation",
        "Backtest",
        "Risk Metrics",
        "Retirement Planner",
        "Commentary",
        "Rebalancing Alerts",
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
                selected_vol,
                selected_ret,
                selected_name,
                ms_vol,
                ms_ret,
                mv_vol,
                mv_ret,
            ),
            use_container_width=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected Portfolio", selected_name)
        col2.metric("Expected Return", f"{selected_ret:.2%}")
        col3.metric("Volatility", f"{selected_vol:.2%}")
        col4.metric("Sharpe Ratio", f"{selected_sharpe:.2f}")

    with tab2:
        render_portfolio(
            selected_name,
            selected_weights,
            tickers,
            last_prices,
            investment,
            selected_ret,
            selected_vol,
            selected_sharpe,
        )

        st.plotly_chart(
            make_sector_chart(sector_weights),
            use_container_width=True,
        )

    with tab3:
        portfolio_growth = growth_from_returns(selected_returns, investment)

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

    with tab4:
        spy_prices = fetch_prices(("SPY",), years)

        if spy_prices.empty:
            st.warning("Could not download SPY benchmark data.")
        else:
            spy_returns = spy_prices.iloc[:, 0].pct_change().dropna()

            common_index = selected_returns.index.intersection(spy_returns.index)

            metrics = metrics_table(
                selected_returns.loc[common_index],
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
        if retirement_paths.empty:
            st.warning("Retirement simulation could not run.")
        else:
            ending_values = retirement_paths.iloc[-1]

            st.plotly_chart(
                make_monte_carlo_chart(retirement_paths, target_wealth),
                use_container_width=True,
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Target Wealth", f"${target_wealth:,.0f}")
            col2.metric("Goal Probability", f"{prob_success:.1%}")
            col3.metric("Median Ending Value", f"${ending_values.median():,.0f}")
            col4.metric("5th Percentile", f"${ending_values.quantile(0.05):,.0f}")

            st.write(
                f"With an initial investment of **${investment:,.0f}**, "
                f"monthly contributions of **${monthly_contribution:,.0f}**, "
                f"and a **{retirement_years}-year** horizon, the model estimates "
                f"a **{prob_success:.1%}** probability of reaching "
                f"**${target_wealth:,.0f}**."
            )

    with tab6:
        commentary = generate_portfolio_commentary(
            selected_name,
            tickers,
            selected_weights,
            selected_ret,
            selected_vol,
            selected_sharpe,
            sector_weights,
            prob_success,
            target_wealth,
            retirement_years,
            monthly_contribution,
        )
with tab7:

    st.subheader("Live Rebalancing Monitor")

    alerts = generate_rebalancing_alerts(
        prices,
        selected_weights,
        tickers,
        investment,
    )

    alerts["Alert"] = (
        alerts["Deviation"].abs()
        > rebalance_threshold
    )

    st.dataframe(
        alerts.style.format(
            {
                "Target Weight": "{:.2%}",
                "Current Weight": "{:.2%}",
                "Deviation": "{:+.2%}",
                "Trade Amount": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    triggered = alerts[
        alerts["Alert"]
    ]

    if len(triggered) == 0:

        st.success(
            "✅ Portfolio currently within rebalancing limits."
        )

    else:

        st.warning(
            f"⚠ {len(triggered)} positions exceed "
            f"the {rebalance_threshold:.0%} threshold."
        )

        st.subheader(
            "Suggested Trades"
        )

        for _, row in triggered.iterrows():

            if row["Deviation"] > 0:

                st.error(
                    f"SELL {row['Ticker']} "
                    f"| Drift {row['Deviation']:+.2%} "
                    f"| Amount ${abs(row['Trade Amount']):,.0f}"
                )

            else:

                st.success(
                    f"BUY {row['Ticker']} "
                    f"| Drift {row['Deviation']:+.2%} "
                    f"| Amount ${abs(row['Trade Amount']):,.0f}"
                )

    chart_df = alerts.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["Ticker"],
            y=chart_df["Current Weight"],
            name="Current",
        )
    )

    fig.add_trace(
        go.Bar(
            x=chart_df["Ticker"],
            y=chart_df["Target Weight"],
            name="Target",
        )
    )

    fig.update_layout(
        title="Current vs Target Weights",
        yaxis_tickformat=".0%",
        barmode="group",
        template="plotly_white",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
        st.markdown(commentary)

else:
    st.write(
        "Enter tickers in the sidebar and click **Build portfolio**. "
        "Use the optimization goal dropdown to choose Maximum Sharpe, "
        "Minimum Volatility, or a Target Return portfolio such as 14%."
    )
