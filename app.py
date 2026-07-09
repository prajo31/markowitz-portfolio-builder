"""
Markowitz Portfolio Builder
---------------------------
Fetches live market data (via Yahoo Finance) and builds an optimal portfolio
using Modern Portfolio Theory (Markowitz mean-variance optimization).

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

TRADING_DAYS = 252

st.set_page_config(page_title="Markowitz Portfolio Builder", layout="wide")

# ----------------------------- Sidebar inputs -----------------------------

st.sidebar.title("Portfolio settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    help="Any symbols Yahoo Finance recognizes, e.g. AAPL, MSFT, VTI, BTC-USD",
)

years = st.sidebar.slider("Years of price history", 1, 10, 3)
rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual, %)", value=4.0, step=0.25
) / 100.0
max_weight = st.sidebar.slider(
    "Max weight per asset (%)", 10, 100, 100,
    help="Cap any single holding. 100% = unconstrained (long-only).",
) / 100.0
investment = st.sidebar.number_input(
    "Amount to invest ($)", value=10_000.0, min_value=0.0, step=500.0
)
n_frontier = st.sidebar.slider("Frontier resolution (points)", 20, 100, 50)
run = st.sidebar.button("Build portfolio", type="primary")

st.title("Markowitz Portfolio Builder")
st.caption(
    "Live prices from Yahoo Finance • Mean-variance optimization "
    "(max Sharpe & min volatility) • Long-only"
)

# ----------------------------- Data fetching ------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple, years: int) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(
        list(tickers),
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(data, pd.Series):  # single ticker
        data = data.to_frame(tickers[0])
    return data.dropna(how="all").ffill().dropna()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> pd.Series:
    """Most recent price for each ticker (for share allocation)."""
    data = yf.download(list(tickers), period="5d", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.ffill().iloc[-1]


# --------------------------- Markowitz engine -----------------------------

def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    ret = float(weights @ mean_returns)
    vol = float(np.sqrt(weights @ cov_matrix @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def optimize(mean_returns, cov_matrix, rf, max_w, objective="sharpe", target_return=None):
    """Solve the Markowitz problem with SLSQP (long-only, weight cap)."""
    n = len(mean_returns)
    x0 = np.full(n, 1 / n)
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mean_returns - target_return}
        )

    if objective == "sharpe":
        fun = lambda w: -portfolio_stats(w, mean_returns, cov_matrix, rf)[2]
    else:  # minimize volatility
        fun = lambda w: np.sqrt(w @ cov_matrix @ w)

    result = minimize(fun, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500})
    return result


def efficient_frontier(mean_returns, cov_matrix, rf, max_w, n_points):
    """Trace the efficient frontier between min-vol and max attainable return."""
    min_vol = optimize(mean_returns, cov_matrix, rf, max_w, "vol")
    ret_low = float(min_vol.x @ mean_returns)
    ret_high = float(mean_returns.max()) if max_w >= 1 else float(
        np.sort(mean_returns)[::-1][: int(np.ceil(1 / max_w))].mean()
    )
    targets = np.linspace(ret_low, ret_high * 0.999, n_points)
    vols, rets = [], []
    for t in targets:
        res = optimize(mean_returns, cov_matrix, rf, max_w, "vol", target_return=t)
        if res.success:
            r, v, _ = portfolio_stats(res.x, mean_returns, cov_matrix, rf)
            rets.append(r)
            vols.append(v)
    return np.array(vols), np.array(rets)


def random_portfolios(mean_returns, cov_matrix, rf, n=3000, seed=42):
    """Monte Carlo cloud for context on the risk/return plot."""
    rng = np.random.default_rng(seed)
    k = len(mean_returns)
    w = rng.dirichlet(np.ones(k), size=n)
    rets = w @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov_matrix, w))
    sharpes = (rets - rf) / vols
    return vols, rets, sharpes


# ------------------------------- Main flow --------------------------------

if run:
    tickers = tuple(
        sorted({t.strip().upper() for t in tickers_input.split(",") if t.strip()})
    )
    if len(tickers) < 2:
        st.error("Enter at least two tickers to diversify.")
        st.stop()
    if max_weight * len(tickers) < 1:
        st.error(
            f"Max weight {max_weight:.0%} × {len(tickers)} assets can't sum to 100%. "
            "Raise the cap or add tickers."
        )
        st.stop()

    with st.spinner("Fetching live market data…"):
        try:
            prices = fetch_prices(tickers, years)
            last_prices = fetch_last_prices(tickers)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No data for: {', '.join(missing)} — they were skipped.")
        prices = prices.drop(columns=missing, errors="ignore")
    tickers = tuple(prices.columns)
    if len(tickers) < 2:
        st.error("Fewer than two tickers had usable data.")
        st.stop()

    # Annualized inputs to the Markowitz model
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean().values * TRADING_DAYS
    cov_matrix = daily_returns.cov().values * TRADING_DAYS

    with st.spinner("Optimizing…"):
        max_sharpe = optimize(mean_returns, cov_matrix, rf_rate, max_weight, "sharpe")
        min_vol = optimize(mean_returns, cov_matrix, rf_rate, max_weight, "vol")
        ef_vols, ef_rets = efficient_frontier(
            mean_returns, cov_matrix, rf_rate, max_weight, n_frontier
        )
        mc_vols, mc_rets, mc_sharpes = random_portfolios(
            mean_returns, cov_matrix, rf_rate
        )

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(max_sharpe.x, mean_returns, cov_matrix, rf_rate)
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(min_vol.x, mean_returns, cov_matrix, rf_rate)

    # ------------------------- Efficient frontier -------------------------

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets, mode="markers", name="Random portfolios",
        marker=dict(size=4, color=mc_sharpes, colorscale="Viridis",
                    colorbar=dict(title="Sharpe"), opacity=0.45),
        hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ef_vols, y=ef_rets, mode="lines", name="Efficient frontier",
        line=dict(color="#1f77b4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret], mode="markers+text", name="Max Sharpe",
        marker=dict(symbol="star", size=18, color="#d62728"),
        text=["Max Sharpe"], textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret], mode="markers+text", name="Min volatility",
        marker=dict(symbol="diamond", size=14, color="#2ca02c"),
        text=["Min vol"], textposition="bottom center",
    ))
    fig.update_layout(
        title="Efficient frontier (annualized)",
        xaxis=dict(title="Volatility (risk)", tickformat=".0%"),
        yaxis=dict(title="Expected return", tickformat=".0%"),
        height=520, legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------- Two portfolios ---------------------------

    def render_portfolio(name, weights, ret, vol, sharpe, key):
        st.subheader(name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected return", f"{ret:.1%}")
        c2.metric("Volatility", f"{vol:.1%}")
        c3.metric("Sharpe ratio", f"{sharpe:.2f}")

        w = pd.Series(weights, index=tickers)
        w = w[w > 0.001].sort_values(ascending=False)
        alloc = pd.DataFrame({
            "Weight": w,
            "Amount ($)": w * investment,
            "Last price ($)": last_prices[w.index],
        })
        alloc["Shares"] = (alloc["Amount ($)"] / alloc["Last price ($)"]).round(2)
        st.dataframe(
            alloc.style.format({
                "Weight": "{:.1%}", "Amount ($)": "${:,.0f}",
                "Last price ($)": "${:,.2f}", "Shares": "{:,.2f}",
            }),
            use_container_width=True,
            key=f"alloc_{key}",
        )
        pie = go.Figure(go.Pie(labels=w.index, values=w.values, hole=0.45,
                               textinfo="label+percent"))
        pie.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(pie, use_container_width=True, key=f"pie_{key}")

    left, right = st.columns(2)
    with left:
        render_portfolio("⭐ Maximum Sharpe portfolio", max_sharpe.x, ms_ret, ms_vol, ms_sharpe, key="sharpe")
    with right:
        render_portfolio("🛡️ Minimum volatility portfolio", min_vol.x, mv_ret, mv_vol, mv_sharpe, key="minvol")

    # ----------------------------- Diagnostics ----------------------------

    with st.expander("Model inputs: correlations & per-asset stats"):
        stats = pd.DataFrame({
            "Ann. return": mean_returns,
            "Ann. volatility": np.sqrt(np.diag(cov_matrix)),
        }, index=tickers)
        stats["Sharpe"] = (stats["Ann. return"] - rf_rate) / stats["Ann. volatility"]
        st.dataframe(stats.style.format({
            "Ann. return": "{:.1%}", "Ann. volatility": "{:.1%}", "Sharpe": "{:.2f}",
        }), use_container_width=True)

        corr = daily_returns.corr()
        heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        heat.update_layout(title="Return correlations", height=420)
        st.plotly_chart(heat, use_container_width=True)

    st.info(
        "Markowitz optimization uses historical means and covariances as estimates "
        "of the future — treat results as a starting point, not investment advice."
    )
else:
    st.write(
        "Enter your tickers in the sidebar and click **Build portfolio**. "
        "The app pulls live price history, computes annualized returns and the "
        "covariance matrix, then solves the Markowitz mean-variance problem to "
        "find the **maximum Sharpe** and **minimum volatility** portfolios, "
        "plotted against the efficient frontier."
    )
