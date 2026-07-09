"""
Markowitz Portfolio Builder
---------------------------
Fetches live market data via Yahoo Finance and builds optimal portfolios
using Modern Portfolio Theory / Markowitz mean-variance optimization.

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
    layout="wide"
)


# ----------------------------- Sidebar inputs -----------------------------

st.sidebar.title("Portfolio settings")

tickers_input = st.sidebar.text_input(
    "Tickers comma-separated",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    help="Any symbols Yahoo Finance recognizes, e.g. AAPL, MSFT, VTI, BTC-USD",
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
        help="Cap any single holding. 100% = unconstrained long-only.",
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

run = st.sidebar.button(
    "Build portfolio",
    type="primary",
)


# ----------------------------- Page header -----------------------------

st.title("Markowitz Portfolio Builder")

st.caption(
    "Live prices from Yahoo Finance • Mean-variance optimization "
    "max Sharpe and min volatility • Long-only"
)


# ----------------------------- Data fetching -----------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple, years: int) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers.
    """

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
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = raw["Close"]
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]]
            prices.columns = [tickers[0]]
        else:
            raise ValueError("Downloaded data does not contain Close prices.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna(axis=1, how="all")

    return prices


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> pd.Series:
    """
    Fetch most recent close price for each ticker.
    """

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

    last_prices = prices.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1]

    return last_prices


# ----------------------------- Markowitz engine -----------------------------

def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    """
    Return annualized return, volatility, and Sharpe ratio.
    """

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
    """
    Solve Markowitz optimization with SLSQP.
    Long-only portfolio, optional max weight cap.
    """

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
            _, _, sharpe = portfolio_stats(
                w,
                mean_returns,
                cov_matrix,
                rf,
            )
            return -sharpe

    elif objective == "vol":

        def objective_fn(w):
            _, vol, _ = portfolio_stats(
                w,
                mean_returns,
                cov_matrix,
                rf,
            )
            return vol

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
    """
    Trace the efficient frontier.
    """

    min_vol_result = optimize(
        mean_returns,
        cov_matrix,
        rf,
        max_w,
        objective="vol",
    )

    if not min_vol_result.success:
        return np.array([]), np.array([])

    ret_low = float(
        np.dot(
            min_vol_result.x,
            mean_returns,
        )
    )

    # Approximate highest attainable return under max weight constraint.
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
    """
    Generate random long-only portfolios for context.
    Honors max weight cap by rejection sampling.
    """

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


# ----------------------------- Main flow -----------------------------

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
        st.error("Enter at least two valid tickers to diversify.")
        st.stop()

    if max_weight * len(tickers) < 1:
        st.error(
            f"Max weight {max_weight:.0%} × {len(tickers)} assets cannot sum to 100%. "
            "Raise the cap or add more tickers."
        )
        st.stop()

    # ----------------------------- Fetch data -----------------------------

    with st.spinner("Fetching live market data..."):
        try:
            prices = fetch_prices(tickers, years)
            last_prices = fetch_last_prices(tickers)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    missing = [
        t
        for t in tickers
        if t not in prices.columns
    ]

    if missing:
        st.warning(
            f"No usable data for: {', '.join(missing)}. They were skipped."
        )
        prices = prices.drop(
            columns=missing,
            errors="ignore",
        )

    prices = prices.dropna(axis=1, how="all")
    tickers = tuple(prices.columns)

    if len(tickers) < 2:
        st.error("Fewer than two tickers had usable data.")
        st.stop()

    last_prices = last_prices.reindex(tickers)
    last_prices = last_prices.replace([np.inf, -np.inf], np.nan)

    if last_prices.isna().any():
        bad = list(last_prices[last_prices.isna()].index)
        st.warning(
            f"Missing latest prices for: {', '.join(bad)}. "
            "These tickers were removed."
        )

        keep = [
            t
            for t in tickers
            if t not in bad
        ]

        prices = prices[keep]
        last_prices = last_prices[keep]
        tickers = tuple(keep)

    if len(tickers) < 2:
        st.error("Fewer than two tickers had valid recent prices.")
        st.stop()

    # ----------------------------- Model inputs -----------------------------

    daily_returns = prices.pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan)
    daily_returns = daily_returns.dropna(how="any")

    if daily_returns.empty:
        st.error("Unable to calculate returns from downloaded prices.")
        st.stop()

    mean_returns_series = daily_returns.mean() * TRADING_DAYS
    cov_matrix_df = daily_returns.cov() * TRADING_DAYS

    if mean_returns_series.isna().any():
        st.error("Mean return estimates contain NaN values.")
        st.write(mean_returns_series)
        st.stop()

    if cov_matrix_df.isna().any().any():
        st.error("Covariance matrix contains NaN values.")
        st.write(cov_matrix_df)
        st.stop()

    mean_returns = mean_returns_series.values
    cov_matrix = cov_matrix_df.values

    # Small regularization to reduce numerical instability.
    cov_matrix = cov_matrix + np.eye(len(mean_returns)) * 1e-8

    if not np.isfinite(mean_returns).all():
        st.error("Mean returns contain invalid values.")
        st.stop()

    if not np.isfinite(cov_matrix).all():
        st.error("Covariance matrix contains invalid values.")
        st.stop()

    # ----------------------------- Optimization -----------------------------

    with st.spinner("Optimizing portfolio..."):

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
            st.error(
                f"Maximum Sharpe optimization failed: {max_sharpe.message}"
            )
            st.stop()

        if not min_vol.success:
            st.error(
                f"Minimum volatility optimization failed: {min_vol.message}"
            )
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
        )

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(
        max_sharpe.x,
        mean_returns,
        cov_matrix,
        rf_rate,
    )

    mv_ret, mv_vol, mv_sharpe = portfolio_stats(
        min_vol.x,
        mean_returns,
        cov_matrix,
        rf_rate,
    )

    # ----------------------------- Efficient frontier chart -----------------------------

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
                    "Return %{y:.1%}"
                    "<extra></extra>"
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
        title="Efficient frontier annualized",
        xaxis=dict(
            title="Volatility risk",
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Expected return",
            tickformat=".0%",
        ),
        height=520,
        legend=dict(
            orientation="h",
            y=-0.18,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------- Portfolio rendering -----------------------------

    def render_portfolio(
        name,
        weights,
        ret,
        vol,
        sharpe,
        key,
    ):
        st.subheader(name)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Expected return",
            f"{ret:.1%}",
        )

        c2.metric(
            "Volatility",
            f"{vol:.1%}",
        )

        c3.metric(
            "Sharpe ratio",
            f"{sharpe:.2f}",
        )

        w = pd.Series(
            weights,
            index=tickers,
        )

        w = w[w > 0.001].sort_values(
            ascending=False
        )

        alloc = pd.DataFrame(
            {
                "Weight": w,
                "Amount $": w * investment,
                "Last price $": last_prices[w.index],
            }
        )

        alloc["Shares"] = (
            alloc["Amount $"]
            / alloc["Last price $"]
        ).round(2)

        st.dataframe(
            alloc.style.format(
                {
                    "Weight": "{:.1%}",
                    "Amount $": "${:,.0f}",
                    "Last price $": "${:,.2f}",
                    "Shares": "{:,.2f}",
                }
            ),
            use_container_width=True,
            key=f"alloc_{key}",
        )

        pie = go.Figure(
            go.Pie(
                labels=w.index,
                values=w.values,
                hole=0.45,
                textinfo="label+percent",
            )
        )

        pie.update_layout(
            height=320,
            margin=dict(
                t=10,
                b=10,
                l=10,
                r=10,
            ),
            showlegend=False,
        )

        st.plotly_chart(
            pie,
            use_container_width=True,
            key=f"pie_{key}",
        )

    left, right = st.columns(2)

    with left:
        render_portfolio(
            "⭐ Maximum Sharpe portfolio",
            max_sharpe.x,
            ms_ret,
            ms_vol,
            ms_sharpe,
            key="sharpe",
        )

    with right:
        render_portfolio(
            "🛡️ Minimum volatility portfolio",
            min_vol.x,
            mv_ret,
            mv_vol,
            mv_sharpe,
            key="minvol",
        )

    # ----------------------------- Diagnostics -----------------------------

    with st.expander("Model inputs: correlations and per-asset stats"):

        stats = pd.DataFrame(
            {
                "Ann. return": mean_returns,
                "Ann. volatility": np.sqrt(
                    np.diag(cov_matrix)
                ),
            },
            index=tickers,
        )

        stats["Sharpe"] = (
            stats["Ann. return"] - rf_rate
        ) / stats["Ann. volatility"]

        st.dataframe(
            stats.style.format(
                {
                    "Ann. return": "{:.1%}",
                    "Ann. volatility": "{:.1%}",
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
            title="Return correlations",
            height=420,
        )

        st.plotly_chart(
            heat,
            use_container_width=True,
        )

    st.info(
        "Markowitz optimization uses historical means and covariances as estimates "
        "of the future. Treat results as a starting point, not investment advice."
    )

else:
    st.write(
        "Enter your tickers in the sidebar and click **Build portfolio**. "
        "The app pulls live price history, computes annualized returns and the "
        "covariance matrix, then solves the Markowitz mean-variance problem to "
        "find the maximum Sharpe and minimum volatility portfolios, plotted "
        "against the efficient frontier."
    )
