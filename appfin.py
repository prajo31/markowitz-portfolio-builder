"""
Markowitz Portfolio Builder

Run with:
    streamlit run app.py

Requirements:
    streamlit yfinance pandas numpy scipy plotly pandas-datareader curl_cffi
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from pandas_datareader import data as pdr

TRADING_DAYS = 252

st.set_page_config(page_title="Markowitz Portfolio Builder", layout="wide")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("Portfolio settings")

PRESETS = {
    "— custom —": "",
    "Big Tech": "AAPL, MSFT, GOOGL, AMZN, NVDA",
    "Diversified": "AAPL, JPM, JNJ, XOM, PG, VTI",
    "Index ETFs": "SPY, QQQ, IWM, VEA, AGG",
    "Sector Mix": "XLK, XLF, XLE, XLV, XLY",
}
preset = st.sidebar.selectbox("Example portfolios", list(PRESETS.keys()))

tickers_input = st.sidebar.text_input(
    "Tickers comma-separated",
    value=PRESETS[preset] or "AAPL, MSFT, GOOGL, AMZN, JPM",
    help="Examples: AAPL, MSFT, VTI, SPY, QQQ. (Crypto like BTC-USD only works when Yahoo is reachable.)",
)

years = st.sidebar.slider("Years of price history", 1, 10, 3)
rf_rate = st.sidebar.number_input("Risk-free rate annual %", value=4.0, step=0.25) / 100.0
max_weight = st.sidebar.slider(
    "Max weight per asset %", 10, 100, 100,
    help="Cap any single holding. 100% means unconstrained long-only.",
) / 100.0
investment = st.sidebar.number_input("Amount to invest $", value=10000.0, min_value=0.0, step=500.0)

optimization_mode = st.sidebar.selectbox(
    "Optimization goal", ["Maximum Sharpe", "Minimum Volatility", "Target Return"]
)
target_return = st.sidebar.number_input(
    "Target annual return %", min_value=1.0, max_value=50.0, value=14.0, step=0.5
) / 100.0
n_frontier = st.sidebar.slider("Frontier resolution points", 20, 100, 50)

st.sidebar.divider()
st.sidebar.subheader("Retirement planner")
retirement_years = st.sidebar.slider("Planning years", 1, 40, 20)
monthly_contribution = st.sidebar.number_input("Monthly contribution $", value=1000.0, min_value=0.0, step=100.0)
target_wealth = st.sidebar.number_input("Target wealth $", value=1000000.0, min_value=0.0, step=50000.0)
forecast_sims = st.sidebar.slider("Monte Carlo simulations", 500, 10000, 3000, step=500)

run = st.sidebar.button("Build portfolio", type="primary")

st.title("Markowitz Portfolio Builder")
st.caption(
    "Live prices (Yahoo + Stooq fallback) • Markowitz optimization • Interactive frontier • "
    "Retirement projection • Risk metrics • Portfolio commentary"
)

# ---------------------------------------------------------------------
# Data helpers  (resilient: Yahoo primary, Stooq fallback)
# ---------------------------------------------------------------------

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except Exception:
    _SESSION = None


def _normalize(df, tickers):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif "Close" in df.columns:
        df = df[["Close"]]
        df.columns = list(tickers)[:1]
    df = df.reindex(columns=[t for t in tickers if t in df.columns])
    return df.dropna(how="all").ffill().dropna(how="all")


def _from_yahoo(tickers, period):
    try:
        try:
            raw = yf.download(list(tickers), period=period, auto_adjust=True,
                              progress=False, threads=True, session=_SESSION)
        except TypeError:  # older yfinance without session kw
            raw = yf.download(list(tickers), period=period, auto_adjust=True,
                              progress=False, threads=True)
        return _normalize(raw, tickers)
    except Exception:
        return pd.DataFrame()


def _from_stooq(tickers, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    frames = {}
    for t in tickers:
        if "-" in t:  # Stooq has no BTC-USD style symbols
            continue
        try:
            s = pdr.DataReader(t, "stooq", start, end)["Close"].sort_index()
            if not s.empty:
                frames[t] = s
        except Exception:
            continue
    return pd.DataFrame(frames).ffill().dropna(how="all") if frames else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple, years: int) -> pd.DataFrame:
    prices = _from_yahoo(tickers, f"{years}y")
    if prices.empty or prices.shape[1] < len(tickers):
        stooq = _from_stooq(tickers, years)
        if not stooq.empty:
            if prices.empty:
                prices = stooq
            else:
                add = stooq[[c for c in stooq.columns if c not in prices.columns]]
                prices = prices.join(add, how="outer")
    return prices.ffill().dropna()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> pd.Series:
    prices = _from_yahoo(tickers, "5d")
    if prices.empty:
        prices = _from_stooq(tickers, 1).tail(5)
    return prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)


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
# Portfolio math
# ---------------------------------------------------------------------


def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def optimize(mean_returns, cov_matrix, rf, max_w, objective="sharpe",
             target_return=None, risk_aversion=None):
    n = len(mean_returns)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq",
                            "fun": lambda w: np.dot(w, mean_returns) - target_return})

    def objective_function(w):
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix, rf)
        if objective == "sharpe":
            return -sharpe
        if objective == "vol":
            return vol
        if objective == "return":
            return -ret
        if objective == "utility":  # Markowitz mean-variance utility
            return -(ret - 0.5 * risk_aversion * (w @ cov_matrix @ w))
        return vol

    return minimize(objective_function, x0, method="SLSQP", bounds=bounds,
                    constraints=constraints, options={"maxiter": 1000})


def efficient_frontier(mean_returns, cov_matrix, rf, max_w, n_points):
    min_vol_result = optimize(mean_returns, cov_matrix, rf, max_w, "vol")
    max_return_result = optimize(mean_returns, cov_matrix, rf, max_w, "return")
    if not min_vol_result.success or not max_return_result.success:
        return np.array([]), np.array([])
    min_ret = portfolio_stats(min_vol_result.x, mean_returns, cov_matrix, rf)[0]
    max_ret = portfolio_stats(max_return_result.x, mean_returns, cov_matrix, rf)[0]
    target_returns = np.linspace(min_ret, max_ret, n_points)
    fv, fr = [], []
    for target in target_returns:
        r = optimize(mean_returns, cov_matrix, rf, max_w, "vol", target_return=target)
        if r.success:
            ret, vol, _ = portfolio_stats(r.x, mean_returns, cov_matrix, rf)
            fr.append(ret)
            fv.append(vol)
    return np.array(fv), np.array(fr)


def random_portfolios(mean_returns, cov_matrix, rf, max_w, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(mean_returns)
    weights_list, attempts, max_attempts = [], 0, n * 50
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
# Risk metrics
# ---------------------------------------------------------------------


def annualized_return(returns):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    years_count = len(returns) / TRADING_DAYS
    if years_count <= 0:
        return np.nan
    return (1 + returns).prod() ** (1 / years_count) - 1


def annualized_volatility(returns):
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, rf):
    av = annualized_volatility(returns)
    if av == 0 or pd.isna(av):
        return np.nan
    return (annualized_return(returns) - rf) / av


def max_drawdown_from_returns(returns):
    wealth = (1 + returns.dropna()).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def drawdown_series_from_value(value_series):
    return value_series / value_series.cummax() - 1


def sortino_ratio(returns, rf):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    downside = returns[returns < 0]
    if downside.empty:
        return np.nan
    dv = downside.std() * np.sqrt(TRADING_DAYS)
    if dv == 0:
        return np.nan
    return (annualized_return(returns) - rf) / dv


def value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()
    return np.nan if returns.empty else np.percentile(returns, 100 * (1 - confidence))


def conditional_value_at_risk(returns, confidence=0.95):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def metrics_table(portfolio_returns, benchmark_returns, rf):
    rows = ["Annualized Return", "Annualized Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Daily VaR 95%", "Daily CVaR 95%"]
    def col(r):
        return [annualized_return(r), annualized_volatility(r), sharpe_ratio(r, rf),
                sortino_ratio(r, rf), max_drawdown_from_returns(r),
                value_at_risk(r), conditional_value_at_risk(r)]
    return pd.DataFrame({"Metric": rows, "Portfolio": col(portfolio_returns),
                         "SPY": col(benchmark_returns)})


# ---------------------------------------------------------------------
# Backtest / simulation
# ---------------------------------------------------------------------


def portfolio_return_series(daily_returns, weights):
    return pd.Series(daily_returns.values @ weights, index=daily_returns.index, name="Portfolio")


def growth_from_returns(returns, starting_value):
    return (1 + returns).cumprod() * starting_value


def monte_carlo_forecast_with_contributions(returns, starting_value, monthly_contribution,
                                            years, simulations, seed=42):
    rng = np.random.default_rng(seed)
    daily = returns.dropna().values
    if len(daily) == 0:
        return pd.DataFrame()
    days = years * TRADING_DAYS
    sim = rng.choice(daily, size=(days, simulations), replace=True)
    values = np.zeros((days, simulations))
    values[0, :] = starting_value * (1 + sim[0, :])
    for day in range(1, days):
        values[day, :] = values[day - 1, :] * (1 + sim[day, :])
        if day % 21 == 0:
            values[day, :] += monthly_contribution
    return pd.DataFrame(values)


def goal_probability(mc_paths, target_wealth):
    if mc_paths.empty:
        return np.nan
    return float((mc_paths.iloc[-1] >= target_wealth).mean())


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------


def make_frontier_chart(mc_vols, mc_rets, mc_sharpes, ef_vols, ef_rets,
                        selected_vol, selected_ret, selected_name,
                        ms_vol, ms_ret, mv_vol, mv_ret):
    fig = go.Figure()
    if len(mc_vols) > 0:
        fig.add_trace(go.Scatter(x=mc_vols, y=mc_rets, mode="markers",
            marker=dict(size=5, color=mc_sharpes, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Sharpe")),
            name="Random portfolios"))
    if len(ef_vols) > 0:
        fig.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode="lines",
            line=dict(width=4), name="Efficient frontier"))
    fig.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret], mode="markers",
        marker=dict(size=13, color="red", symbol="star"), name="Max Sharpe"))
    fig.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret], mode="markers",
        marker=dict(size=13, color="blue", symbol="diamond"), name="Min Volatility"))
    fig.add_trace(go.Scatter(x=[selected_vol], y=[selected_ret], mode="markers",
        marker=dict(size=17, color="orange", symbol="circle"), name=selected_name))
    fig.update_layout(title="Efficient Frontier", xaxis_title="Annualized Volatility",
                      yaxis_title="Annualized Return", template="plotly_white")
    return fig


def make_growth_chart(p, b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p, mode="lines", name="Portfolio"))
    fig.add_trace(go.Scatter(x=b.index, y=b, mode="lines", name="SPY"))
    fig.update_layout(title="Backtested Growth", xaxis_title="Date",
                      yaxis_title="Portfolio Value", template="plotly_white")
    return fig


def make_drawdown_chart(p, b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=drawdown_series_from_value(p),
                             mode="lines", name="Portfolio Drawdown"))
    fig.add_trace(go.Scatter(x=b.index, y=drawdown_series_from_value(b),
                             mode="lines", name="SPY Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date",
                      yaxis_title="Drawdown", template="plotly_white")
    return fig


def make_sector_chart(sector_weights):
    fig = go.Figure(go.Treemap(labels=sector_weights.index, parents=[""] * len(sector_weights),
                               values=sector_weights.values, textinfo="label+percent root"))
    fig.update_layout(title="Sector Exposure", template="plotly_white")
    return fig


def make_correlation_chart(corr):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                               colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                               text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Return Correlations", template="plotly_white", height=420)
    return fig


def make_monte_carlo_chart(mc_paths, target_wealth):
    pct = pd.DataFrame({
        "5th": mc_paths.quantile(0.05, axis=1), "Median": mc_paths.quantile(0.50, axis=1),
        "95th": mc_paths.quantile(0.95, axis=1)})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct.index, y=pct["95th"], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=pct.index, y=pct["5th"], mode="lines", fill="tonexty",
                             line=dict(width=0), name="5th to 95th percentile"))
    fig.add_trace(go.Scatter(x=pct.index, y=pct["Median"], mode="lines", name="Median"))
    fig.add_trace(go.Scatter(x=pct.index, y=[target_wealth] * len(pct), mode="lines",
                             line=dict(dash="dash"), name="Target wealth"))
    fig.update_layout(title="Retirement Monte Carlo Forecast", xaxis_title="Trading Days",
                      yaxis_title="Portfolio Value", template="plotly_white")
    return fig


# ---------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------


def generate_portfolio_commentary(selected_name, tickers, weights, expected_return,
                                  volatility, sharpe, sector_weights, goal_prob,
                                  target_wealth, retirement_years, monthly_contribution):
    ws = pd.Series(weights, index=tickers)
    top = ws.sort_values(ascending=False).head(3)
    largest_ticker, largest_weight = top.index[0], top.iloc[0]
    dom_sector, dom_w = ("Unknown", 0)
    if not sector_weights.empty:
        dom_sector, dom_w = sector_weights.index[0], sector_weights.iloc[0]
    risk_label = "relatively conservative" if volatility < 0.12 else "aggressive" if volatility > 0.25 else "moderate"
    goal_label = "strong" if goal_prob >= 0.75 else "moderate" if goal_prob >= 0.50 else "weak"
    text = f"""
### Portfolio Commentary

The selected portfolio is the **{selected_name}**, with an expected annual return of **{expected_return:.2%}**, annualized volatility of **{volatility:.2%}**, and a Sharpe ratio of **{sharpe:.2f}**.

This is a **{risk_label}** allocation. The largest position is **{largest_ticker}** at **{largest_weight:.2%}**, so it has the biggest influence on performance.

Top holdings:
"""
    for t, w in top.items():
        text += f"- **{t}**: {w:.2%}\n"
    text += f"""

Sector concentration is led by **{dom_sector}** at roughly **{dom_w:.2%}**. A high figure here means more exposure to sector-specific shocks.

For the retirement goal — target **${target_wealth:,.0f}** over **{retirement_years} years** with **${monthly_contribution:,.0f}**/month — the Monte Carlo simulation estimates a **{goal_prob:.1%}** probability of success, a **{goal_label}** outcome under historical-return assumptions.

*This is a planning tool based on historical data, not a guarantee of future returns or investment advice.*
"""
    return text


# ---------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------


def compute_rebalance(tickers, current_shares, last_prices, target_weights,
                      new_cash=0.0, band=0.0):
    """Compare current holdings to target weights and produce trades.

    band = tolerance in percentage points; drifts smaller than this are left as HOLD.
    Buys and sells self-finance when new_cash == 0.
    """
    tickers = list(tickers)
    shares = pd.Series(current_shares, index=tickers, dtype=float).fillna(0.0)
    px = pd.Series(last_prices, index=tickers, dtype=float).reindex(tickers)
    tgt_w = pd.Series(target_weights, index=tickers, dtype=float)

    cur_val = (shares * px).fillna(0.0)
    total_cur = float(cur_val.sum())
    total_alloc = total_cur + new_cash

    cur_w = cur_val / total_cur if total_cur > 0 else pd.Series(0.0, index=tickers)
    tgt_val = tgt_w * total_alloc
    drift = (cur_w - tgt_w) * 100.0
    trade_val = tgt_val - cur_val
    trade_sh = (trade_val / px).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    threshold = 0.005 * max(total_alloc, 1.0)
    action = pd.Series("HOLD", index=tickers)
    action[trade_val > threshold] = "BUY"
    action[trade_val < -threshold] = "SELL"
    if band > 0:
        action[drift.abs() < band] = "HOLD"

    table = pd.DataFrame({
        "Ticker": tickers,
        "Current Shares": shares.values,
        "Last Price": px.values,
        "Current Value": cur_val.values,
        "Current Weight": cur_w.values,
        "Target Weight": tgt_w.values,
        "Drift (pp)": drift.values,
        "Target Value": tgt_val.values,
        "Trade $": trade_val.values,
        "Trade Shares": trade_sh.values,
        "Action": action.values,
    })
    summary = dict(
        total_current=total_cur, total_alloc=total_alloc,
        total_buy=float(trade_val[trade_val > 0].sum()),
        total_sell=float(-trade_val[trade_val < 0].sum()),
        turnover=float(trade_val.abs().sum() / total_alloc) if total_alloc > 0 else 0.0,
    )
    return table, summary


def make_weights_bar(table):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["Ticker"], y=table["Current Weight"],
                         name="Current", marker_color="#8888cc"))
    fig.add_trace(go.Bar(x=table["Ticker"], y=table["Target Weight"],
                         name="Target", marker_color="#ff9933"))
    fig.update_layout(title="Current vs Target Weights", barmode="group",
                      yaxis_tickformat=".0%", template="plotly_white")
    return fig


def make_drift_bar(table):
    colors = ["#d62728" if a == "SELL" else "#2ca02c" if a == "BUY" else "#999999"
              for a in table["Action"]]
    fig = go.Figure(go.Bar(x=table["Drift (pp)"], y=table["Ticker"], orientation="h",
                           marker_color=colors,
                           text=[f"{d:+.1f}pp" for d in table["Drift (pp)"]],
                           textposition="outside"))
    fig.update_layout(title="Portfolio Drift (overweight ▶ / ◀ underweight)",
                      xaxis_title="Drift vs target (percentage points)",
                      template="plotly_white")
    return fig


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------


def render_portfolio(name, weights, tickers, last_prices, investment, ret, vol, sharpe, key):
    st.subheader(name)
    ws = pd.Series(weights, index=tickers, name="Weight")
    dollars = ws * investment
    aligned = last_prices.reindex(tickers)
    shares = (dollars / aligned).replace([np.inf, -np.inf], np.nan).fillna(0)
    table = pd.DataFrame({"Ticker": tickers, "Weight": ws.values,
                          "Dollar Allocation": dollars.values,
                          "Last Price": aligned.values, "Approx Shares": shares.values})
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Annual Return", f"{ret:.2%}")
    c2.metric("Annual Volatility", f"{vol:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.dataframe(table.style.format({"Weight": "{:.2%}", "Dollar Allocation": "${:,.2f}",
                 "Last Price": "${:,.2f}", "Approx Shares": "{:,.4f}"}),
                 use_container_width=True, key=f"tbl_{key}")
    st.download_button(f"Download {name} allocation CSV",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name.lower().replace(' ', '_')}_allocation.csv",
                       mime="text/csv", key=f"dl_{key}")


# ---------------------------------------------------------------------
# Build step  (runs only on button press; results cached in session_state)
# ---------------------------------------------------------------------

if run:
    tickers = tuple(sorted({t.strip().upper() for t in tickers_input.split(",") if t.strip()}))
    if len(tickers) < 2:
        st.error("Enter at least two tickers.")
        st.stop()
    if max_weight * len(tickers) < 1:
        st.error(f"Max weight {max_weight:.0%} is too low for {len(tickers)} assets. "
                 "Increase max weight or add more tickers.")
        st.stop()

    with st.spinner("Downloading price data..."):
        prices = fetch_prices(tickers, years)
    if prices.empty:
        st.error("No price data was downloaded (Yahoo and Stooq both returned nothing). "
                 "Check ticker symbols, or try again in a minute.")
        st.stop()

    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    tickers = tuple(prices.columns)
    if len(tickers) < 2:
        st.error("Need at least two valid tickers after downloading data.")
        st.stop()

    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean().values * TRADING_DAYS
    cov_matrix = daily_returns.cov().values * TRADING_DAYS

    ms = optimize(mean_returns, cov_matrix, rf_rate, max_weight, "sharpe")
    mv = optimize(mean_returns, cov_matrix, rf_rate, max_weight, "vol")
    tr = optimize(mean_returns, cov_matrix, rf_rate, max_weight, "vol", target_return=target_return)
    if not ms.success or not mv.success:
        st.error("Optimization failed. Try increasing max weight or changing tickers.")
        st.stop()

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms.x, mean_returns, cov_matrix, rf_rate)
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv.x, mean_returns, cov_matrix, rf_rate)

    if optimization_mode == "Maximum Sharpe":
        sel_name, sel_w = "Maximum Sharpe Portfolio", ms.x
    elif optimization_mode == "Minimum Volatility":
        sel_name, sel_w = "Minimum Volatility Portfolio", mv.x
    else:
        if not tr.success:
            st.error(f"No feasible portfolio for target return {target_return:.2%}. "
                     "Lower the target or raise max weight.")
            st.stop()
        sel_name, sel_w = f"Target Return Portfolio ({target_return:.2%})", tr.x
    sel_ret, sel_vol, sel_sharpe = portfolio_stats(sel_w, mean_returns, cov_matrix, rf_rate)

    with st.spinner("Building charts and simulations..."):
        ef_vols, ef_rets = efficient_frontier(mean_returns, cov_matrix, rf_rate, max_weight, n_frontier)
        mc_vols, mc_rets, mc_sharpes = random_portfolios(mean_returns, cov_matrix, rf_rate, max_weight)
        last_prices = fetch_last_prices(tickers)
        sel_returns = portfolio_return_series(daily_returns, sel_w)
        retirement_paths = monte_carlo_forecast_with_contributions(
            sel_returns, investment, monthly_contribution, retirement_years, forecast_sims)
        prob_success = goal_probability(retirement_paths, target_wealth)
        sector_map = get_sector_data(tickers)
        sectors = pd.Series(sector_map)
        sector_weights = pd.Series(sel_w, index=tickers).groupby(sectors).sum().sort_values(ascending=False)
        spy_prices = fetch_prices(("SPY",), years)

    # Persist everything the render step needs so widgets can rerun without rebuilding
    st.session_state["model"] = dict(
        tickers=tickers, mean_returns=mean_returns, cov_matrix=cov_matrix,
        rf_rate=rf_rate, max_weight=max_weight, investment=investment,
        daily_returns=daily_returns, prices=prices, last_prices=last_prices,
        corr=daily_returns.corr(),
        ef_vols=ef_vols, ef_rets=ef_rets, mc_vols=mc_vols, mc_rets=mc_rets, mc_sharpes=mc_sharpes,
        ms_ret=ms_ret, ms_vol=ms_vol, ms_sharpe=ms_sharpe, ms_w=ms.x,
        mv_ret=mv_ret, mv_vol=mv_vol, mv_sharpe=mv_sharpe, mv_w=mv.x,
        sel_name=sel_name, sel_w=sel_w, sel_ret=sel_ret, sel_vol=sel_vol, sel_sharpe=sel_sharpe,
        sel_returns=sel_returns, retirement_paths=retirement_paths, prob_success=prob_success,
        sector_weights=sector_weights, spy_prices=spy_prices,
        retirement_years=retirement_years, monthly_contribution=monthly_contribution,
        target_wealth=target_wealth, years=years,
    )

# ---------------------------------------------------------------------
# Render step  (reads from session_state -> sliders stay live)
# ---------------------------------------------------------------------

if "model" not in st.session_state:
    st.write("Enter tickers in the sidebar and click **Build portfolio**. "
             "Choose Maximum Sharpe, Minimum Volatility, or a Target Return goal. "
             "Once built, the **Explore** tab lets you drag a risk slider and watch "
             "the portfolio move along the efficient frontier live.")
    st.stop()

m = st.session_state["model"]

tabs = st.tabs(["Optimization", "Explore", "Allocation", "Rebalance", "Backtest",
                "Risk Metrics", "Retirement Planner", "Commentary"])

# ---- Optimization ----
with tabs[0]:
    st.plotly_chart(make_frontier_chart(
        m["mc_vols"], m["mc_rets"], m["mc_sharpes"], m["ef_vols"], m["ef_rets"],
        m["sel_vol"], m["sel_ret"], m["sel_name"],
        m["ms_vol"], m["ms_ret"], m["mv_vol"], m["mv_ret"]),
        use_container_width=True, key="chart_frontier")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Portfolio", m["sel_name"])
    c2.metric("Expected Return", f"{m['sel_ret']:.2%}")
    c3.metric("Volatility", f"{m['sel_vol']:.2%}")
    c4.metric("Sharpe Ratio", f"{m['sel_sharpe']:.2f}")
    st.plotly_chart(make_correlation_chart(m["corr"]), use_container_width=True, key="chart_corr")

# ---- Explore (live risk-aversion slider) ----
with tabs[1]:
    st.subheader("Explore the efficient frontier")
    st.write("Drag the slider to change **risk aversion (λ)**. Low λ chases return; "
             "high λ minimizes risk. The point moves along the frontier and the "
             "allocation updates instantly — no rebuild needed.")
    lam = st.slider("Risk aversion (λ)", 0.5, 50.0, 8.0, 0.5, key="risk_aversion")

    res = optimize(m["mean_returns"], m["cov_matrix"], m["rf_rate"],
                   m["max_weight"], "utility", risk_aversion=lam)
    if res.success:
        r, v, s = portfolio_stats(res.x, m["mean_returns"], m["cov_matrix"], m["rf_rate"])
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure()
            if len(m["ef_vols"]) > 0:
                fig.add_trace(go.Scatter(x=m["ef_vols"], y=m["ef_rets"], mode="lines",
                                         line=dict(width=4), name="Efficient frontier"))
            fig.add_trace(go.Scatter(x=[m["ms_vol"]], y=[m["ms_ret"]], mode="markers",
                marker=dict(size=12, color="red", symbol="star"), name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers",
                marker=dict(size=20, color="orange", symbol="circle"), name="Your portfolio"))
            fig.update_layout(title="Your position on the frontier",
                              xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True, key="chart_explore")
        with col2:
            a, b, c = st.columns(3)
            a.metric("Return", f"{r:.2%}")
            b.metric("Volatility", f"{v:.2%}")
            c.metric("Sharpe", f"{s:.2f}")
            ws = pd.Series(res.x, index=m["tickers"])
            ws = ws[ws > 0.001].sort_values(ascending=False)
            pie = go.Figure(go.Pie(labels=ws.index, values=ws.values, hole=0.45,
                                   textinfo="label+percent"))
            pie.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(pie, use_container_width=True, key="chart_explore_pie")
    else:
        st.warning("Could not solve at this risk-aversion level. Try a different value.")

# ---- Allocation ----
with tabs[2]:
    render_portfolio(m["sel_name"], m["sel_w"], m["tickers"], m["last_prices"],
                     m["investment"], m["sel_ret"], m["sel_vol"], m["sel_sharpe"], key="alloc")
    st.plotly_chart(make_sector_chart(m["sector_weights"]), use_container_width=True, key="chart_sector")

# ---- Rebalance ----
with tabs[3]:
    st.subheader("Rebalance to target")
    st.write("Enter the shares you currently hold. The app compares them to the "
             "**target weights** above, shows how far each position has drifted, and "
             "tells you exactly what to buy or sell to get back on target.")

    tks = list(m["tickers"])
    px = m["last_prices"].reindex(tks)

    # Seed the editor with an equal-weight starting portfolio so drift is visible.
    seed_val = m["investment"] / len(tks)
    seed_shares = (seed_val / px).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)
    seed_df = pd.DataFrame({"Ticker": tks, "Current Shares": seed_shares.values})

    colA, colB = st.columns([3, 2])
    with colA:
        edited = st.data_editor(
            seed_df, key="rebal_editor", use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(disabled=True),
                "Current Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
            },
        )
    with colB:
        new_cash = st.number_input("New cash to deploy $", value=0.0, min_value=0.0,
                                   step=500.0, key="rebal_cash")
        band = st.slider("Tolerance band (pp)", 0.0, 10.0, 0.0, 0.5, key="rebal_band",
                         help="Ignore drifts smaller than this — avoids tiny, costly trades.")

    table, summ = compute_rebalance(
        tks, edited["Current Shares"].values, px.values, m["sel_w"],
        new_cash=new_cash, band=band)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Value", f"${summ['total_current']:,.0f}")
    k2.metric("Total to Buy", f"${summ['total_buy']:,.0f}")
    k3.metric("Total to Sell", f"${summ['total_sell']:,.0f}")
    k4.metric("Turnover", f"{summ['turnover']:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_weights_bar(table), use_container_width=True, key="chart_rebal_weights")
    with c2:
        st.plotly_chart(make_drift_bar(table), use_container_width=True, key="chart_rebal_drift")

    buys = table[table["Action"] == "BUY"].sort_values("Trade $", ascending=False)
    sells = table[table["Action"] == "SELL"].sort_values("Trade $")

    fmt = {"Last Price": "${:,.2f}", "Current Value": "${:,.0f}",
           "Current Weight": "{:.1%}", "Target Weight": "{:.1%}", "Drift (pp)": "{:+.1f}",
           "Target Value": "${:,.0f}", "Trade $": "${:,.0f}", "Trade Shares": "{:+,.2f}",
           "Current Shares": "{:,.2f}"}

    b, s = st.columns(2)
    with b:
        st.markdown("#### 🟢 Buy")
        if buys.empty:
            st.write("Nothing to buy.")
        else:
            st.dataframe(buys[["Ticker", "Trade $", "Trade Shares", "Drift (pp)"]]
                         .style.format(fmt), use_container_width=True, hide_index=True,
                         key="tbl_buys")
    with s:
        st.markdown("#### 🔴 Sell")
        if sells.empty:
            st.write("Nothing to sell.")
        else:
            st.dataframe(sells[["Ticker", "Trade $", "Trade Shares", "Drift (pp)"]]
                         .style.format(fmt), use_container_width=True, hide_index=True,
                         key="tbl_sells")

    with st.expander("Full rebalancing detail"):
        st.dataframe(table.style.format(fmt), use_container_width=True, hide_index=True,
                     key="tbl_rebal_full")
        st.download_button("Download rebalance plan CSV",
                           table.to_csv(index=False).encode("utf-8"),
                           file_name="rebalance_plan.csv", mime="text/csv",
                           key="dl_rebal")

# ---- Backtest ----
with tabs[4]:
    pg = growth_from_returns(m["sel_returns"], m["investment"])
    spy = m["spy_prices"]
    if spy.empty:
        st.warning("Could not download SPY benchmark data.")
    else:
        spy_ret = spy.iloc[:, 0].pct_change().dropna()
        sg = growth_from_returns(spy_ret, m["investment"])
        idx = pg.index.intersection(sg.index)
        pg, sg = pg.loc[idx], sg.loc[idx]
        st.plotly_chart(make_growth_chart(pg, sg), use_container_width=True, key="chart_growth")
        st.plotly_chart(make_drawdown_chart(pg, sg), use_container_width=True, key="chart_dd")

# ---- Risk Metrics ----
with tabs[5]:
    spy = m["spy_prices"]
    if spy.empty:
        st.warning("Could not download SPY benchmark data.")
    else:
        spy_ret = spy.iloc[:, 0].pct_change().dropna()
        idx = m["sel_returns"].index.intersection(spy_ret.index)
        metrics = metrics_table(m["sel_returns"].loc[idx], spy_ret.loc[idx], m["rf_rate"])
        st.dataframe(metrics.style.format({"Portfolio": "{:.2%}", "SPY": "{:.2%}"}),
                     use_container_width=True, key="tbl_metrics")

# ---- Retirement Planner ----
with tabs[6]:
    rp = m["retirement_paths"]
    if rp.empty:
        st.warning("Retirement simulation could not run.")
    else:
        ev = rp.iloc[-1]
        st.plotly_chart(make_monte_carlo_chart(rp, m["target_wealth"]),
                        use_container_width=True, key="chart_mc")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Wealth", f"${m['target_wealth']:,.0f}")
        c2.metric("Goal Probability", f"{m['prob_success']:.1%}")
        c3.metric("Median Ending Value", f"${ev.median():,.0f}")
        c4.metric("5th Percentile", f"${ev.quantile(0.05):,.0f}")
        st.write(f"With an initial **${m['investment']:,.0f}**, monthly "
                 f"**${m['monthly_contribution']:,.0f}**, over **{m['retirement_years']} years**, "
                 f"the model estimates a **{m['prob_success']:.1%}** probability of reaching "
                 f"**${m['target_wealth']:,.0f}**.")

# ---- Commentary ----
with tabs[7]:
    st.markdown(generate_portfolio_commentary(
        m["sel_name"], m["tickers"], m["sel_w"], m["sel_ret"], m["sel_vol"], m["sel_sharpe"],
        m["sector_weights"], m["prob_success"], m["target_wealth"],
        m["retirement_years"], m["monthly_contribution"]))
