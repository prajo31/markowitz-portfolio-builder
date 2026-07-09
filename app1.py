"""
Markowitz Portfolio Builder
---------------------------
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
    value=10_000.0,
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
    max_value=10_000,
    value=3_000,
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
        raise ValueError("Yahoo Finance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna(axis=1, how="all")

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
        raise ValueError("Yahoo Finance returned no recent price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    return prices.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1]


@st.cache_data(ttl=86400, show_spinner=False)
def get_sector_data(tickers: tuple) -> dict:
    sector_map = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            if sector is None or sector == "":
                sector = "Unknown"
            sector_map[ticker] = sector
        except Exception:
            sector_map[ticker] = "Unknown"

    return sector_map


# ---------------------------------------------------------------------
# Portfolio Math
# ---------------------------------------------------------------------

def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    ret = float(np.dot(weights, mean_returns))

    variance = float(
        np.dot(
            weights.T,
            np.dot(cov_matrix, weights),
        )
    )

    variance = max(variance, 0.0)
    vol = float(np.sqrt(variance))

    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0

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

    bounds = [(0.0, max_w)] * n

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

    if objective == "sharpe":

        def objective_fn(w):
            return -portfolio_stats(w, mean_returns, cov_matrix, rf)[2]

    elif objective == "vol":

        def objective_fn(w):
            return portfolio_stats(w, mean_returns, cov_matrix, rf)[1]

    else:
        raise ValueError("objective must be either 'sharpe' or 'vol'.")

    result = minimize(
        objective_fn,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
        },
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

    if not min_vol_result.success:
        return np.array([]), np.array([])

    ret_low = float(np.dot(min_vol_result.x, mean_returns))

    order = np.argsort(mean_returns)[::-1]
    remaining = 1.0
    high_weights = np.zeros(len(mean_returns))

    for idx in order:
        weight = min(max_w, remaining)
        high_weights[idx] = weight
        remaining -= weight

        if remaining <= 1e-12:
            break

    ret_high = float(np.dot(high_weights, mean_returns))

    if ret_high <= ret_low:
        ret_high = float(np.max(mean_returns))

    targets = np.linspace(
        ret_low,
        ret_high * 0.999,
        n_points,
    )

    vols = []
    rets = []

    for target in targets:
        result = optimize(
            mean_returns,
            cov_matrix,
            rf,
            max_w,
            objective="vol",
            target_return=target,
        )

        if result.success:
            r, v, _ = portfolio_stats(
                result.x,
                mean_returns,
                cov_matrix,
                rf,
            )
            rets.append(r)
            vols.append(v)

    return np.array(vols), np.array(rets)


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
    max_attempts = n * 100

    while len(weights_list) < n and attempts < max_attempts:
        w = rng.dirichlet(np.ones(k))

        if np.all(w <= max_w + 1e-12):
            weights_list.append(w)

        attempts += 1

    if len(weights_list) == 0:
        return np.array([]), np.array([]), np.array([])

    weights = np.array(weights_list)

    rets = weights @ mean_returns

    vols = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            weights,
            cov_matrix,
            weights,
        )
    )

    vols = np.where(vols <= 1e-10, np.nan, vols)
    sharpes = (rets - rf) / vols

    return vols, rets, sharpes


# ---------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------

def annualized_return(returns):
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    total_growth = float((1 + returns).prod())

    if total_growth <= 0:
        return np.nan

    years = len(returns) / TRADING_DAYS

    return total_growth ** (1 / years) - 1


def annualized_volatility(returns):
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf):
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)

    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan

    return (ann_ret - rf) / ann_vol


def max_drawdown_from_returns(returns):
    wealth = (1 + returns.dropna()).cumprod()

    if wealth.empty:
        return np.nan

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

    if downside.empty or downside.std() == 0:
        return np.nan

    ann_ret = annualized_return(returns)
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)

    return (ann_ret - rf) / downside_vol


def value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return np.percentile(returns, (1 - confidence) * 100)


def conditional_value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]

    if tail.empty:
        return np.nan

    return tail.mean()


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

    return pd.DataFrame(data).set_index("Metric")


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

    freq_map = {
        "Monthly": "ME",
        "Quarterly": "QE",
        "Annually": "YE",
    }

    resample_freq = freq_map[frequency]

    rebalance_dates = (
        prices
        .resample(resample_freq)
        .last()
        .index
    )

    rebalance_dates = set(
        prices.index[
            prices.index.searchsorted(rebalance_dates, side="left")
        ].intersection(prices.index)
    )

    first_prices = prices.iloc[0]
    holdings = starting_value * target_weights / first_prices

    values = []

    for date, price_row in prices.iterrows():
        current_value = float((holdings * price_row).sum())
        values.append(current_value)

        if date in rebalance_dates:
            holdings = current_value * target_weights / price_row

    return pd.Series(
        values,
        index=prices.index,
        name=frequency,
    )


def simulate_buy_and_hold(
    prices,
    target_weights,
    starting_value,
):
    prices = prices.dropna(how="any")

    if prices.empty:
        return pd.Series(dtype=float)

    first_prices = prices.iloc[0]
    holdings = starting_value * target_weights / first_prices

    values = prices.mul(holdings, axis=1).sum(axis=1)

    values.name = "Buy & Hold"

    return values


def monte_carlo_forecast(
    returns,
    starting_value,
    years,
    simulations,
    seed=42,
):
    rng = np.random.default_rng(seed)

    returns = returns.dropna()

    mu = returns.mean()
    sigma = returns.std()

    days = int(years * TRADING_DAYS)

    simulated_daily_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=(days, simulations),
    )

    paths = starting_value * np.cumprod(
        1 + simulated_daily_returns,
        axis=0,
    )

    index = np.arange(1, days + 1)

    return pd.DataFrame(
        paths,
        index=index,
    )


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
                name="Random portfolios",
                marker=dict(
                    size=4,
                    color=mc_sharpes,
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe"),
                    opacity=0.45,
                ),
                hovertemplate=(
                    "Volatility %{x:.1%}<br>"
                    "Return %{y:.1%}<extra></extra>"
                ),
            )
        )

    if len(ef_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=ef_vols,
                y=ef_rets,
                mode="lines",
                name="Efficient frontier",
                line=dict(
                    color="#1f77b4",
                    width=3,
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[ms_vol],
            y=[ms_ret],
            mode="markers+text",
            name="Max Sharpe",
            marker=dict(
                symbol="star",
                size=18,
                color="#d62728",
            ),
            text=["Max Sharpe"],
            textposition="top center",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers+text",
            name="Min volatility",
            marker=dict(
                symbol="diamond",
                size=14,
                color="#2ca02c",
            ),
            text=["Min vol"],
            textposition="bottom center",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(
            title="Volatility",
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Expected Return",
            tickformat=".0%",
        ),
        height=520,
        legend=dict(
            orientation="h",
            y=-0.18,
        ),
    )

    return fig


def make_growth_chart(portfolio_growth, benchmark_growth):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_growth.index,
            y=portfolio_growth.values,
            mode="lines",
            name="Optimized portfolio",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_growth.index,
            y=benchmark_growth.values,
            mode="lines",
            name="SPY benchmark",
        )
    )

    fig.update_layout(
        title="Historical Growth of Investment",
        yaxis=dict(title="Portfolio Value $"),
        xaxis=dict(title="Date"),
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )

    return fig


def make_drawdown_chart(portfolio_growth, benchmark_growth):
    portfolio_dd = drawdown_series_from_value(portfolio_growth)
    benchmark_dd = drawdown_series_from_value(benchmark_growth)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_dd.index,
            y=portfolio_dd.values,
            mode="lines",
            name="Portfolio drawdown",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_dd.index,
            y=benchmark_dd.values,
            mode="lines",
            name="SPY drawdown",
        )
    )

    fig.update_layout(
        title="Drawdown",
        yaxis=dict(
            title="Drawdown",
            tickformat=".0%",
        ),
        xaxis=dict(title="Date"),
        height=420,
        legend=dict(orientation="h", y=-0.18),
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
        height=450,
    )

    return fig


def make_rebalance_chart(rebalance_results):
    fig = go.Figure()

    for name, series in rebalance_results.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Rebalancing Simulator",
        yaxis=dict(title="Portfolio Value $"),
        xaxis=dict(title="Date"),
        height=500,
        legend=dict(orientation="h", y=-0.18),
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
            name="95th",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["5th"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.18)",
            line=dict(width=0),
            name="5th-95th percentile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["75th"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="75th",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["25th"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.30)",
            line=dict(width=0),
            name="25th-75th percentile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["Median"],
            mode="lines",
            name="Median",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title="Monte Carlo Forecast",
        xaxis=dict(title="Trading Days"),
        yaxis=dict(title="Projected Portfolio Value $"),
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )

    return fig, percentiles


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

    c1, c2, c3 = st.columns(3)

    c1.metric("Expected Return", f"{ret:.1%}")
    c2.metric("Volatility", f"{vol:.1%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    w = pd.Series(weights, index=tickers)
    w = w[w > 0.001].sort_values(ascending=False)

    alloc = pd.DataFrame(
        {
            "Weight": w,
            "Amount $": w * investment,
            "Last Price $": last_prices[w.index],
        }
    )

    alloc["Shares"] = (
        alloc["Amount $"] / alloc["Last Price $"]
    ).round(2)

    st.dataframe(
        alloc.style.format(
            {
                "Weight": "{:.1%}",
                "Amount $": "${:,.0f}",
                "Last Price $": "${:,.2f}",
                "Shares": "{:,.2f}",
            }
        ),
        use_container_width=True,
    )

    fig = go.Figure(
        go.Pie(
            labels=w.index,
            values=w.values,
            hole=0.45,
            textinfo="label+percent",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------

if run:

    tickers = tuple(
        sorted(
            {
                t.strip().upper()
                for t in tickers_input.split(",")
                if t.strip()
            }
        )
    )

    if len(tickers) < 2:
        st.error("Enter at least two valid tickers.")
        st.stop()

    if max_weight * len(tickers) < 1:
        st.error(
            f"Max weight {max_weight:.0%} × {len(tickers)} assets cannot sum to 100%. "
            "Raise the cap or add more tickers."
        )
        st.stop()

    with st.spinner("Fetching market data..."):
        try:
            prices = fetch_prices(tickers, years)
            last_prices = fetch_last_prices(tickers)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    missing = [
        t for t in tickers
        if t not in prices.columns
    ]

    if missing:
        st.warning(
            f"No usable data for: {', '.join(missing)}. These were skipped."
        )

    prices = prices.dropna(axis=1, how="all")
    tickers = tuple(prices.columns)

    if len(tickers) < 2:
        st.error("Fewer than two tickers had usable data.")
        st.stop()

    last_prices = last_prices.reindex(tickers)
    last_prices = last_prices.replace([np.inf, -np.inf], np.nan)

    bad_latest = list(last_prices[last_prices.isna()].index)

    if bad_latest:
        st.warning(
            f"Missing recent prices for: {', '.join(bad_latest)}. "
            "These tickers were removed."
        )

        keep = [
            t for t in tickers
            if t not in bad_latest
        ]

        prices = prices[keep]
        last_prices = last_prices[keep]
        tickers = tuple(keep)

    if len(tickers) < 2:
        st.error("Fewer than two tickers had valid prices.")
        st.stop()

    daily_returns = prices.pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan)
    daily_returns = daily_returns.dropna(how="any")

    if daily_returns.empty:
        st.error("Unable to calculate daily returns.")
        st.stop()

    mean_returns_series = daily_returns.mean() * TRADING_DAYS
    cov_matrix_df = daily_returns.cov() * TRADING_DAYS

    if mean_returns_series.isna().any() or cov_matrix_df.isna().any().any():
        st.error("Return estimates contain NaN values.")
        st.stop()

    mean_returns = mean_returns_series.values
    cov_matrix = cov_matrix_df.values

    cov_matrix = cov_matrix + np.eye(len(mean_returns)) * 1e-8

    with st.spinner("Optimizing..."):
        max_sharpe = optimize(
            mean_returns,
            cov_matrix,
            rf_rate,
            max_weight,
            objective="sharpe",
        )

        min_vol = optimize(
            mean_returns,
            cov_matrix,
            rf_rate,
            max_weight,
            objective="vol",
        )

        if not max_sharpe.success:
            st.error(f"Max Sharpe optimization failed: {max_sharpe.message}")
            st.stop()

        if not min_vol.success:
            st.error(f"Min Volatility optimization failed: {min_vol.message}")
            st.stop()

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

    max_sharpe_weights = max_sharpe.x
    min_vol_weights = min_vol.x

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

    # Use max-Sharpe portfolio for analytics tabs.
    selected_weights = max_sharpe_weights

    portfolio_returns = portfolio_return_series(
        daily_returns,
        selected_weights,
    )

    portfolio_growth = growth_from_returns(
        portfolio_returns,
        investment,
    )

    with st.spinner("Fetching SPY benchmark..."):
        try:
            spy_prices = fetch_prices(("SPY",), years)
            spy_returns = spy_prices.iloc[:, 0].pct_change().dropna()
            spy_returns = spy_returns.reindex(portfolio_returns.index).dropna()

            common_index = portfolio_returns.index.intersection(spy_returns.index)
            portfolio_returns_aligned = portfolio_returns.reindex(common_index)
            spy_returns_aligned = spy_returns.reindex(common_index)

            portfolio_growth_aligned = growth_from_returns(
                portfolio_returns_aligned,
                investment,
            )

            spy_growth = growth_from_returns(
                spy_returns_aligned,
                investment,
            )

        except Exception:
            spy_returns_aligned = pd.Series(dtype=float)
            portfolio_returns_aligned = portfolio_returns
            portfolio_growth_aligned = portfolio_growth
            spy_growth = pd.Series(dtype=float)

    with st.spinner("Fetching sector data..."):
        sector_map = get_sector_data(tickers)

    sector_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": selected_weights,
            "Sector": [
                sector_map.get(t, "Unknown")
                for t in tickers
            ],
        }
    )

    sector_weights = (
        sector_df
        .groupby("Sector")["Weight"]
        .sum()
        .sort_values(ascending=False)
    )

    rebalance_results = {
        "Buy & Hold": simulate_buy_and_hold(
            prices,
            selected_weights,
            investment,
        ),
        "Monthly": simulate_rebalanced_portfolio(
            prices,
            selected_weights,
            investment,
            "Monthly",
        ),
        "Quarterly": simulate_rebalanced_portfolio(
            prices,
            selected_weights,
            investment,
            "Quarterly",
        ),
        "Annually": simulate_rebalanced_portfolio(
            prices,
            selected_weights,
            investment,
            "Annually",
        ),
    }

    mc_paths = monte_carlo_forecast(
        portfolio_returns,
        investment,
        forecast_years,
        forecast_sims,
    )

    # -----------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Optimization",
            "Backtest vs SPY",
            "Risk Metrics",
            "Sector Exposure",
            "Rebalancing",
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

        left, right = st.columns(2)

        with left:
            render_portfolio(
                "⭐ Maximum Sharpe Portfolio",
                max_sharpe_weights,
                tickers,
                last_prices,
                investment,
                ms_ret,
                ms_vol,
                ms_sharpe,
            )

        with right:
            render_portfolio(
                "🛡️ Minimum Volatility Portfolio",
                min_vol_weights,
                tickers,
                last_prices,
                investment,
                mv_ret,
                mv_vol,
                mv_sharpe,
            )

        with st.expander("Model Inputs"):
            stats = pd.DataFrame(
                {
                    "Annualized Return": mean_returns,
                    "Annualized Volatility": np.sqrt(np.diag(cov_matrix)),
                },
                index=tickers,
            )

            stats["Sharpe"] = (
                stats["Annualized Return"] - rf_rate
            ) / stats["Annualized Volatility"]

            st.dataframe(
                stats.style.format(
                    {
                        "Annualized Return": "{:.1%}",
                        "Annualized Volatility": "{:.1%}",
                        "Sharpe": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            corr = daily_returns.corr()

            heat = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                )
            )

            heat.update_layout(
                title="Return Correlations",
                height=420,
            )

            st.plotly_chart(heat, use_container_width=True)

    with tab2:
        if spy_growth.empty:
            st.warning("SPY benchmark data could not be loaded.")
        else:
            st.plotly_chart(
                make_growth_chart(
                    portfolio_growth_aligned,
                    spy_growth,
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                make_drawdown_chart(
                    portfolio_growth_aligned,
                    spy_growth,
                ),
                use_container_width=True,
            )

            compare = metrics_table(
                portfolio_returns_aligned,
                spy_returns_aligned,
                rf_rate,
            )

            st.subheader("Portfolio vs SPY Metrics")

            st.dataframe(
                compare.style.format(
                    {
                        "Portfolio": "{:.2%}",
                        "SPY": "{:.2%}",
                    },
                    subset=pd.IndexSlice[
                        [
                            "Annualized Return",
                            "Annualized Volatility",
                            "Max Drawdown",
                            "Daily VaR 95%",
                            "Daily CVaR 95%",
                        ],
                        :
                    ],
                ).format(
                    {
                        "Portfolio": "{:.2f}",
                        "SPY": "{:.2f}",
                    },
                    subset=pd.IndexSlice[
                        [
                            "Sharpe Ratio",
                            "Sortino Ratio",
                        ],
                        :
                    ],
                ),
                use_container_width=True,
            )

    with tab3:
        st.subheader("Risk Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Max Drawdown",
            f"{max_drawdown_from_returns(portfolio_returns):.2%}",
        )

        c2.metric(
            "Sortino Ratio",
            f"{sortino_ratio(portfolio_returns, rf_rate):.2f}",
        )

        c3.metric(
            "Daily VaR 95%",
            f"{value_at_risk(portfolio_returns):.2%}",
        )

        c4.metric(
            "Daily CVaR 95%",
            f"{conditional_value_at_risk(portfolio_returns):.2%}",
        )

        risk_table = pd.DataFrame(
            {
                "Metric": [
                    "Annualized Return",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Max Drawdown",
                    "Daily VaR 95%",
                    "Daily CVaR 95%",
                ],
                "Value": [
                    annualized_return(portfolio_returns),
                    annualized_volatility(portfolio_returns),
                    sharpe_ratio(portfolio_returns, rf_rate),
                    sortino_ratio(portfolio_returns, rf_rate),
                    max_drawdown_from_returns(portfolio_returns),
                    value_at_risk(portfolio_returns),
                    conditional_value_at_risk(portfolio_returns),
                ],
            }
        ).set_index("Metric")

        st.dataframe(
            risk_table,
            use_container_width=True,
        )

    with tab4:
        st.subheader("Sector Exposure")

        st.plotly_chart(
            make_sector_chart(sector_weights),
            use_container_width=True,
        )

        st.dataframe(
            sector_df.sort_values("Weight", ascending=False).style.format(
                {
                    "Weight": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        top_sector = sector_weights.index[0]
        top_sector_weight = sector_weights.iloc[0]

        c1, c2 = st.columns(2)

        c1.metric(
            "Largest Sector",
            top_sector,
        )

        c2.metric(
            "Largest Sector Weight",
            f"{top_sector_weight:.1%}",
        )

    with tab5:
        st.subheader("Rebalancing Simulator")

        st.plotly_chart(
            make_rebalance_chart(rebalance_results),
            use_container_width=True,
        )

        rebalance_summary = []

        for name, series in rebalance_results.items():
            returns = series.pct_change().dropna()

            rebalance_summary.append(
                {
                    "Strategy": name,
                    "Ending Value": series.iloc[-1],
                    "Annualized Return": annualized_return(returns),
                    "Annualized Volatility": annualized_volatility(returns),
                    "Sharpe Ratio": sharpe_ratio(returns, rf_rate),
                    "Max Drawdown": max_drawdown_from_returns(returns),
                }
            )

        rebalance_summary = pd.DataFrame(
            rebalance_summary
        ).set_index("Strategy")

        st.dataframe(
            rebalance_summary.style.format(
                {
                    "Ending Value": "${:,.0f}",
                    "Annualized Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    with tab6:
        st.subheader("Monte Carlo Forecast")

        mc_fig, mc_percentiles = make_monte_carlo_chart(mc_paths)

        st.plotly_chart(
            mc_fig,
            use_container_width=True,
        )

        ending_values = mc_paths.iloc[-1]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "5th Percentile Ending Value",
            f"${ending_values.quantile(0.05):,.0f}",
        )

        c2.metric(
            "Median Ending Value",
            f"${ending_values.quantile(0.50):,.0f}",
        )

        c3.metric(
            "95th Percentile Ending Value",
            f"${ending_values.quantile(0.95):,.0f}",
        )

        st.dataframe(
            mc_percentiles.tail(1).T.rename(
                columns={
                    mc_percentiles.index[-1]: "Projected Value"
                }
            ).style.format(
                {
                    "Projected Value": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

    st.info(
        "This app uses historical prices, returns, covariance estimates, and simulations. "
        "Results are for analysis and education only, not investment advice."
    )

else:
    st.write(
        "Enter tickers in the sidebar and click **Build portfolio**. "
        "The app will optimize a portfolio, compare it with SPY, analyze risk, "
        "show sector exposure, simulate rebalancing, and run a Monte Carlo forecast."
    )
