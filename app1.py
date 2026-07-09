
"""
Markowitz Portfolio Builder
---------------------------
A clean Streamlit app for portfolio optimization and analytics.

Features
- Yahoo Finance price download
- Maximum Sharpe and minimum-volatility portfolios
- Efficient frontier and random portfolio cloud
- Portfolio backtest vs benchmark
- Risk metrics: return, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR
- Sector exposure treemap and table
- Rebalancing simulator: buy-and-hold, monthly, quarterly, annually
- Monte Carlo forecast

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

TRADING_DAYS = 252
DEFAULT_TICKERS = "AAPL, MSFT, GOOGL, AMZN, JPM"

st.set_page_config(page_title="Markowitz Portfolio Builder", layout="wide")


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("Portfolio settings")

tickers_input = st.sidebar.text_input(
    "Tickers comma-separated",
    value=DEFAULT_TICKERS,
    help="Examples: AAPL, MSFT, VTI, QQQ, BTC-USD. For Berkshire B use BRK-B.",
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark ticker",
    value="SPY",
)

years = st.sidebar.slider("Years of price history", 1, 10, 3)

rf_rate = st.sidebar.number_input(
    "Risk-free rate annual %",
    value=4.0,
    step=0.25,
) / 100.0

max_weight = st.sidebar.slider(
    "Max weight per asset %",
    min_value=10,
    max_value=100,
    value=100,
    help="Cap a single holding. 100% means unconstrained long-only.",
) / 100.0

investment = st.sidebar.number_input(
    "Amount to invest $",
    value=10_000.0,
    min_value=0.0,
    step=500.0,
)

n_frontier = st.sidebar.slider("Frontier resolution points", 20, 100, 50)
mc_cloud_size = st.sidebar.slider("Random portfolios on frontier chart", 500, 6000, 3000, step=500)
forecast_years = st.sidebar.slider("Monte Carlo forecast years", 1, 10, 5)
forecast_sims = st.sidebar.slider("Monte Carlo simulations", 500, 10000, 3000, step=500)
run = st.sidebar.button("Build portfolio", type="primary")

st.title("Markowitz Portfolio Builder")
st.caption(
    "Live Yahoo Finance prices | Markowitz optimization | Backtesting | "
    "Risk analytics | Sector exposure | Rebalancing simulation"
)


# =============================================================================
# Data helpers
# =============================================================================


def parse_tickers(text: str) -> tuple[str, ...]:
    tickers = [t.strip().upper() for t in text.split(",") if t.strip()]
    return tuple(sorted(set(tickers)))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], years: int) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
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
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not include Close prices.")
        prices = raw["Close"]
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not include Close prices.")
        prices = raw[["Close"]]
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
    prices = prices.sort_index()
    return prices


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple[str, ...]) -> pd.Series:
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


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_sector_data(tickers: tuple[str, ...]) -> dict[str, str]:
    """Fetch sector metadata. ETF/crypto tickers often return Unknown."""
    sectors: dict[str, str] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        sectors[ticker] = sector
    return sectors


# =============================================================================
# Optimization
# =============================================================================


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, rf: float):
    ret = float(np.dot(weights, mean_returns))
    variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    variance = max(variance, 0.0)
    vol = float(np.sqrt(variance))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def optimize_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float,
    max_w: float,
    objective: str = "sharpe",
    target_return: float | None = None,
):
    n = len(mean_returns)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return})

    if objective == "sharpe":
        def objective_fn(w):
            return -portfolio_stats(w, mean_returns, cov_matrix, rf)[2]
    elif objective == "vol":
        def objective_fn(w):
            return portfolio_stats(w, mean_returns, cov_matrix, rf)[1]
    else:
        raise ValueError("objective must be 'sharpe' or 'vol'.")

    return minimize(
        objective_fn,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )


def efficient_frontier(mean_returns, cov_matrix, rf, max_w, n_points):
    min_vol = optimize_portfolio(mean_returns, cov_matrix, rf, max_w, objective="vol")
    if not min_vol.success:
        return np.array([]), np.array([])

    ret_low = float(np.dot(min_vol.x, mean_returns))

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

    targets = np.linspace(ret_low, ret_high * 0.999, n_points)
    vols, rets = [], []
    for target in targets:
        res = optimize_portfolio(mean_returns, cov_matrix, rf, max_w, objective="vol", target_return=target)
        if res.success:
            r, v, _ = portfolio_stats(res.x, mean_returns, cov_matrix, rf)
            rets.append(r)
            vols.append(v)
    return np.array(vols), np.array(rets)


def random_portfolios(mean_returns, cov_matrix, rf, max_w, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(mean_returns)
    weights = []
    attempts = 0
    max_attempts = max(n * 100, 1000)

    while len(weights) < n and attempts < max_attempts:
        w = rng.dirichlet(np.ones(k))
        if np.all(w <= max_w + 1e-12):
            weights.append(w)
        attempts += 1

    if not weights:
        return np.array([]), np.array([]), np.array([])

    w = np.array(weights)
    rets = w @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov_matrix, w))
    vols = np.where(vols <= 1e-10, np.nan, vols)
    sharpes = (rets - rf) / vols
    return vols, rets, sharpes


# =============================================================================
# Metrics
# =============================================================================


def annualized_return(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    total_growth = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) <= 1:
        return np.nan
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float) -> float:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol <= 1e-10:
        return np.nan
    return float((ann_ret - rf) / ann_vol)


def sortino_ratio(returns: pd.Series, rf: float) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) <= 1:
        return np.nan
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    if np.isnan(downside_vol) or downside_vol <= 1e-10:
        return np.nan
    return float((annualized_return(returns) - rf) / downside_vol)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def drawdown_from_growth(growth: pd.Series) -> pd.Series:
    return growth / growth.cummax() - 1.0


def value_at_risk(returns: pd.Series, confidence=0.95) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    return float(np.percentile(returns, (1.0 - confidence) * 100.0))


def conditional_value_at_risk(returns: pd.Series, confidence=0.95) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def metrics_frame(returns_by_name: dict[str, pd.Series], rf: float) -> pd.DataFrame:
    rows = []
    for name, rets in returns_by_name.items():
        rows.append({
            "Strategy": name,
            "Annual return": annualized_return(rets),
            "Volatility": annualized_volatility(rets),
            "Sharpe": sharpe_ratio(rets, rf),
            "Sortino": sortino_ratio(rets, rf),
            "Max drawdown": max_drawdown_from_returns(rets),
            "VaR 95 daily": value_at_risk(rets),
            "CVaR 95 daily": conditional_value_at_risk(rets),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def growth_from_returns(returns: pd.Series, starting_value: float) -> pd.Series:
    returns = pd.Series(returns).dropna()
    return (1.0 + returns).cumprod() * starting_value


# =============================================================================
# Backtest and simulation
# =============================================================================


def fixed_weight_returns(daily_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return daily_returns.dot(weights).rename("Portfolio").dropna()


def buy_and_hold_returns(prices: pd.DataFrame, initial_weights: np.ndarray) -> pd.Series:
    clean = prices.dropna(how="any")
    if clean.empty:
        return pd.Series(dtype=float)
    relative_prices = clean / clean.iloc[0]
    values = relative_prices.dot(initial_weights)
    return values.pct_change().dropna().rename("Buy & Hold")


def period_end_trading_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """Return actual last trading date in each calendar period."""
    s = pd.Series(index=index, data=index)
    dates = s.groupby(index.to_period(freq)).max().tolist()
    return set(pd.Timestamp(d) for d in dates)


def simulate_rebalanced_portfolio(
    prices: pd.DataFrame,
    target_weights: np.ndarray,
    starting_value: float,
    frequency: str,
) -> tuple[pd.Series, pd.Series]:
    """Simulate periodic rebalancing to target_weights."""
    clean = prices.dropna(how="any")
    returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if clean.empty or returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if frequency == "Buy & Hold":
        rets = buy_and_hold_returns(clean, target_weights)
        return growth_from_returns(rets, starting_value).rename(frequency), rets.rename(frequency)

    freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
    rebalance_dates = period_end_trading_dates(returns.index, freq_map[frequency])

    weights = np.array(target_weights, dtype=float)
    asset_values = starting_value * weights
    values = []
    rets = []

    for date, row in returns.iterrows():
        previous_value = float(asset_values.sum())
        asset_values = asset_values * (1.0 + row.values.astype(float))
        current_value = float(asset_values.sum())
        values.append(current_value)
        rets.append(current_value / previous_value - 1.0)

        if date in rebalance_dates:
            asset_values = current_value * weights

    growth = pd.Series(values, index=returns.index, name=frequency)
    ret_series = pd.Series(rets, index=returns.index, name=frequency)
    return growth, ret_series


def monte_carlo_forecast(returns: pd.Series, starting_value: float, years: int, simulations: int, seed=42) -> pd.DataFrame:
    returns = pd.Series(returns).dropna()
    days = int(years * TRADING_DAYS)
    if returns.empty or days <= 0:
        return pd.DataFrame()

    mu = float(returns.mean())
    sigma = float(returns.std())
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    simulated = rng.normal(mu, sigma, size=(days, simulations))
    simulated = np.clip(simulated, -0.95, None)
    paths = starting_value * np.cumprod(1.0 + simulated, axis=0)
    return pd.DataFrame(paths, index=pd.RangeIndex(1, days + 1, name="Trading day"))


# =============================================================================
# Charts
# =============================================================================


def frontier_chart(mc_vols, mc_rets, mc_sharpes, ef_vols, ef_rets, ms, mv):
    fig = go.Figure()
    if len(mc_vols) > 0:
        fig.add_trace(go.Scatter(
            x=mc_vols,
            y=mc_rets,
            mode="markers",
            name="Random portfolios",
            marker=dict(size=4, color=mc_sharpes, colorscale="Viridis", colorbar=dict(title="Sharpe"), opacity=0.45),
            hovertemplate="Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        ))
    if len(ef_vols) > 0:
        fig.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode="lines", name="Efficient frontier", line=dict(width=3)))

    fig.add_trace(go.Scatter(
        x=[ms[1]], y=[ms[0]], mode="markers+text", name="Max Sharpe",
        marker=dict(symbol="star", size=18), text=["Max Sharpe"], textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=[mv[1]], y=[mv[0]], mode="markers+text", name="Min volatility",
        marker=dict(symbol="diamond", size=14), text=["Min Vol"], textposition="bottom center",
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(title="Volatility", tickformat=".0%"),
        yaxis=dict(title="Expected return", tickformat=".0%"),
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def growth_chart(series_by_name: dict[str, pd.Series], title: str):
    fig = go.Figure()
    for name, series in series_by_name.items():
        if series is not None and not series.empty:
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))
    fig.update_layout(title=title, yaxis=dict(title="Value $", tickprefix="$"), height=500, legend=dict(orientation="h", y=-0.18))
    return fig


def drawdown_chart(series_by_name: dict[str, pd.Series]):
    fig = go.Figure()
    for name, series in series_by_name.items():
        if series is not None and not series.empty:
            dd = drawdown_from_growth(series)
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name))
    fig.update_layout(title="Drawdown", yaxis=dict(title="Drawdown", tickformat=".0%"), height=420, legend=dict(orientation="h", y=-0.18))
    return fig


def sector_treemap(sector_weights: pd.Series):
    fig = go.Figure(go.Treemap(
        labels=sector_weights.index,
        parents=[""] * len(sector_weights),
        values=sector_weights.values,
        texttemplate="%{label}<br>%{value:.1%}",
    ))
    fig.update_layout(title="Sector Exposure", height=460)
    return fig


def monte_carlo_chart(paths: pd.DataFrame):
    percentiles = pd.DataFrame({
        "5th": paths.quantile(0.05, axis=1),
        "25th": paths.quantile(0.25, axis=1),
        "Median": paths.quantile(0.50, axis=1),
        "75th": paths.quantile(0.75, axis=1),
        "95th": paths.quantile(0.95, axis=1),
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["95th"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["5th"], fill="tonexty", name="5th-95th", line=dict(width=0), fillcolor="rgba(31,119,180,0.18)"))
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["75th"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["25th"], fill="tonexty", name="25th-75th", line=dict(width=0), fillcolor="rgba(31,119,180,0.30)"))
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["Median"], name="Median", line=dict(width=3)))
    fig.update_layout(title="Monte Carlo Forecast", xaxis_title="Trading days", yaxis=dict(title="Projected value $", tickprefix="$"), height=500, legend=dict(orientation="h", y=-0.18))
    return fig, percentiles


# =============================================================================
# Rendering helpers
# =============================================================================


def render_allocation(name, weights, tickers, last_prices, amount, stats_tuple, key):
    ret, vol, sharpe = stats_tuple
    st.subheader(name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{ret:.1%}")
    c2.metric("Volatility", f"{vol:.1%}")
    c3.metric("Sharpe", f"{sharpe:.2f}")

    weights_series = pd.Series(weights, index=tickers, name="Weight")
    display_weights = weights_series[weights_series > 0.001].sort_values(ascending=False)
    alloc = pd.DataFrame({
        "Weight": display_weights,
        "Amount $": display_weights * amount,
        "Last price $": last_prices.reindex(display_weights.index),
    })
    alloc["Shares"] = (alloc["Amount $"] / alloc["Last price $"]).round(2)

    st.dataframe(
        alloc.style.format({
            "Weight": "{:.1%}",
            "Amount $": "${:,.0f}",
            "Last price $": "${:,.2f}",
            "Shares": "{:,.2f}",
        }),
        use_container_width=True,
        key=f"alloc_{key}",
    )

    pie = go.Figure(go.Pie(labels=display_weights.index, values=display_weights.values, hole=0.45, textinfo="label+percent"))
    pie.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(pie, use_container_width=True, key=f"pie_{key}")
    return alloc


# =============================================================================
# Main app
# =============================================================================

if run:
    tickers = parse_tickers(tickers_input)
    benchmark_ticker = benchmark_ticker.strip().upper() or "SPY"

    if len(tickers) < 2:
        st.error("Enter at least two valid tickers.")
        st.stop()
    if max_weight * len(tickers) < 1.0:
        st.error(f"Max weight {max_weight:.0%} x {len(tickers)} assets cannot sum to 100%. Raise the cap or add more tickers.")
        st.stop()

    with st.spinner("Fetching market data..."):
        try:
            prices = fetch_prices(tickers, years)
            last_prices = fetch_last_prices(tickers)
        except Exception as exc:
            st.error(f"Data download failed: {exc}")
            st.stop()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No usable data for: {', '.join(missing)}. These were skipped.")

    prices = prices.dropna(axis=1, how="all")
    tickers = tuple(prices.columns)
    if len(tickers) < 2:
        st.error("Fewer than two tickers had usable data.")
        st.stop()

    last_prices = last_prices.reindex(tickers).replace([np.inf, -np.inf], np.nan)
    bad_latest = list(last_prices[last_prices.isna()].index)
    if bad_latest:
        st.warning(f"Missing recent prices for: {', '.join(bad_latest)}. These tickers were removed.")
        keep = [t for t in tickers if t not in bad_latest]
        prices = prices[keep]
        last_prices = last_prices[keep]
        tickers = tuple(keep)

    daily_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if daily_returns.empty or len(tickers) < 2:
        st.error("Unable to calculate valid returns for at least two assets.")
        st.stop()

    prices = prices.loc[daily_returns.index]
    mean_returns_series = daily_returns.mean() * TRADING_DAYS
    cov_matrix_df = daily_returns.cov() * TRADING_DAYS

    if mean_returns_series.isna().any() or cov_matrix_df.isna().any().any():
        st.error("Return estimates contain NaN values. Try different tickers or a longer history.")
        st.stop()

    mean_returns = mean_returns_series.values
    cov_matrix = cov_matrix_df.values + np.eye(len(mean_returns_series)) * 1e-8

    with st.spinner("Optimizing portfolios..."):
        max_sharpe = optimize_portfolio(mean_returns, cov_matrix, rf_rate, max_weight, objective="sharpe")
        min_vol = optimize_portfolio(mean_returns, cov_matrix, rf_rate, max_weight, objective="vol")

        if not max_sharpe.success:
            st.error(f"Maximum Sharpe optimization failed: {max_sharpe.message}")
            st.stop()
        if not min_vol.success:
            st.error(f"Minimum volatility optimization failed: {min_vol.message}")
            st.stop()

        ef_vols, ef_rets = efficient_frontier(mean_returns, cov_matrix, rf_rate, max_weight, n_frontier)
        mc_vols, mc_rets, mc_sharpes = random_portfolios(mean_returns, cov_matrix, rf_rate, max_weight, n=mc_cloud_size)

    max_sharpe_weights = max_sharpe.x
    min_vol_weights = min_vol.x
    max_sharpe_stats = portfolio_stats(max_sharpe_weights, mean_returns, cov_matrix, rf_rate)
    min_vol_stats = portfolio_stats(min_vol_weights, mean_returns, cov_matrix, rf_rate)

    selected_weights = max_sharpe_weights
    portfolio_rets = fixed_weight_returns(daily_returns, selected_weights)
    portfolio_growth = growth_from_returns(portfolio_rets, investment)

    # Benchmark
    benchmark_rets = pd.Series(dtype=float)
    benchmark_growth = pd.Series(dtype=float)
    portfolio_rets_for_benchmark = portfolio_rets
    try:
        benchmark_prices = fetch_prices((benchmark_ticker,), years)
        benchmark_rets = benchmark_prices.iloc[:, 0].pct_change().dropna()
        common_idx = portfolio_rets.index.intersection(benchmark_rets.index)
        portfolio_rets_for_benchmark = portfolio_rets.reindex(common_idx).dropna()
        benchmark_rets = benchmark_rets.reindex(portfolio_rets_for_benchmark.index).dropna()
        benchmark_growth = growth_from_returns(benchmark_rets, investment)
        portfolio_growth_bench = growth_from_returns(portfolio_rets_for_benchmark, investment)
    except Exception:
        portfolio_growth_bench = portfolio_growth
        st.warning(f"Benchmark {benchmark_ticker} could not be loaded.")

    # Sector exposure
    sector_map = fetch_sector_data(tickers)
    sector_by_ticker = pd.DataFrame({
        "Ticker": tickers,
        "Weight": selected_weights,
        "Sector": [sector_map.get(t, "Unknown") for t in tickers],
    }).sort_values("Weight", ascending=False)
    sector_weights = sector_by_ticker.groupby("Sector")["Weight"].sum().sort_values(ascending=False)

    # Rebalancing
    rebalance_growth = {}
    rebalance_returns = {}
    for strategy in ["Buy & Hold", "Monthly", "Quarterly", "Annually"]:
        g, r = simulate_rebalanced_portfolio(prices, selected_weights, investment, strategy)
        if not g.empty and not r.empty:
            rebalance_growth[strategy] = g
            rebalance_returns[strategy] = r

    # Forecast
    mc_paths = monte_carlo_forecast(portfolio_rets, investment, forecast_years, forecast_sims)

    tab_opt, tab_backtest, tab_risk, tab_sector, tab_rebalance, tab_forecast = st.tabs([
        "Optimization",
        "Backtest",
        "Risk Metrics",
        "Sector Exposure",
        "Rebalancing",
        "Forecast",
    ])

    with tab_opt:
        st.plotly_chart(
            frontier_chart(mc_vols, mc_rets, mc_sharpes, ef_vols, ef_rets, max_sharpe_stats, min_vol_stats),
            use_container_width=True,
        )
        left, right = st.columns(2)
        with left:
            max_alloc = render_allocation("Maximum Sharpe Portfolio", max_sharpe_weights, tickers, last_prices, investment, max_sharpe_stats, "max")
        with right:
            render_allocation("Minimum Volatility Portfolio", min_vol_weights, tickers, last_prices, investment, min_vol_stats, "min")

        with st.expander("Model inputs"):
            stats = pd.DataFrame({
                "Annualized return": mean_returns,
                "Annualized volatility": np.sqrt(np.diag(cov_matrix)),
                "Sector": [sector_map.get(t, "Unknown") for t in tickers],
            }, index=tickers)
            stats["Sharpe"] = (stats["Annualized return"] - rf_rate) / stats["Annualized volatility"]
            st.dataframe(stats.style.format({
                "Annualized return": "{:.1%}",
                "Annualized volatility": "{:.1%}",
                "Sharpe": "{:.2f}",
            }), use_container_width=True)

            corr = daily_returns.corr()
            heat = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
            ))
            heat.update_layout(title="Return Correlations", height=420)
            st.plotly_chart(heat, use_container_width=True)

    with tab_backtest:
        if benchmark_growth.empty:
            st.warning("Benchmark data is unavailable, showing optimized portfolio only.")
            st.plotly_chart(growth_chart({"Optimized portfolio": portfolio_growth}, f"Growth of ${investment:,.0f}"), use_container_width=True)
            st.plotly_chart(drawdown_chart({"Optimized portfolio": portfolio_growth}), use_container_width=True)
        else:
            st.plotly_chart(growth_chart({"Optimized portfolio": portfolio_growth_bench, benchmark_ticker: benchmark_growth}, f"Growth of ${investment:,.0f}"), use_container_width=True)
            st.plotly_chart(drawdown_chart({"Optimized portfolio": portfolio_growth_bench, benchmark_ticker: benchmark_growth}), use_container_width=True)
            compare_df = metrics_frame({"Optimized portfolio": portfolio_rets_for_benchmark, benchmark_ticker: benchmark_rets}, rf_rate)
            st.dataframe(compare_df.style.format({
                "Annual return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max drawdown": "{:.1%}",
                "VaR 95 daily": "{:.2%}",
                "CVaR 95 daily": "{:.2%}",
            }), use_container_width=True)

    with tab_risk:
        risk_df = metrics_frame({"Maximum Sharpe": portfolio_rets}, rf_rate)
        st.dataframe(risk_df.style.format({
            "Annual return": "{:.1%}",
            "Volatility": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Max drawdown": "{:.1%}",
            "VaR 95 daily": "{:.2%}",
            "CVaR 95 daily": "{:.2%}",
        }), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max drawdown", f"{max_drawdown_from_returns(portfolio_rets):.2%}")
        c2.metric("Sortino", f"{sortino_ratio(portfolio_rets, rf_rate):.2f}")
        c3.metric("Daily VaR 95%", f"{value_at_risk(portfolio_rets):.2%}")
        c4.metric("Daily CVaR 95%", f"{conditional_value_at_risk(portfolio_rets):.2%}")

    with tab_sector:
        if sector_weights.empty:
            st.warning("Sector data is unavailable.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Largest sector", str(sector_weights.index[0]))
            c2.metric("Largest sector weight", f"{sector_weights.iloc[0]:.1%}")
            effective_holdings = 1.0 / np.sum(np.array(selected_weights) ** 2)
            c3.metric("Effective holdings", f"{effective_holdings:.1f}")
            st.plotly_chart(sector_treemap(sector_weights), use_container_width=True)
            st.dataframe(sector_by_ticker.style.format({"Weight": "{:.1%}"}), use_container_width=True)

    with tab_rebalance:
        if not rebalance_growth:
            st.warning("Rebalancing simulation is unavailable.")
        else:
            st.plotly_chart(growth_chart(rebalance_growth, f"Rebalancing comparison for ${investment:,.0f}"), use_container_width=True)
            rebalance_metrics = metrics_frame(rebalance_returns, rf_rate)
            st.dataframe(rebalance_metrics.style.format({
                "Annual return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max drawdown": "{:.1%}",
                "VaR 95 daily": "{:.2%}",
                "CVaR 95 daily": "{:.2%}",
            }), use_container_width=True)

    with tab_forecast:
        if mc_paths.empty:
            st.warning("Monte Carlo forecast is unavailable.")
        else:
            fig, percentiles = monte_carlo_chart(mc_paths)
            st.plotly_chart(fig, use_container_width=True)
            terminal = mc_paths.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("5th percentile ending value", f"${terminal.quantile(0.05):,.0f}")
            c2.metric("Median ending value", f"${terminal.quantile(0.50):,.0f}")
            c3.metric("95th percentile ending value", f"${terminal.quantile(0.95):,.0f}")
            st.dataframe(percentiles.tail(1).T.rename(columns={percentiles.index[-1]: "Projected value"}).style.format({"Projected value": "${:,.0f}"}), use_container_width=True)

    st.info(
        "This app uses historical prices, return estimates, covariance estimates, and simulations. "
        "It is for analysis and education only, not investment advice."
    )

else:
    st.write(
        "Enter tickers in the sidebar and click **Build portfolio**. The app will optimize portfolios, "
        "compare against a benchmark, analyze risk, show sector exposure, simulate rebalancing, and run a Monte Carlo forecast."
    )
PY
python -m py_compile /mnt/data/app.py
wc -l /mnt/data/app.py

