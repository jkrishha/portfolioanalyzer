"""
Portfolio Risk Analyzer — Flask Backend
========================================
pip install flask flask-cors yfinance numpy scipy pandas google-generativeai
python app.py  →  open index.html
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder=".")
CORS(app)

# ─────────────────────────────────────────────────────────────
#  ALIAS MAP
# ─────────────────────────────────────────────────────────────
ALIAS_MAP = {
    "icici bank":"ICICIBANK.NS","icici":"ICICIBANK.NS",
    "reliance":"RELIANCE.NS","reliance industries":"RELIANCE.NS",
    "infosys":"INFY.NS","infy":"INFY.NS",
    "tcs":"TCS.NS","tata consultancy":"TCS.NS",
    "hdfc bank":"HDFCBANK.NS","hdfc":"HDFCBANK.NS",
    "wipro":"WIPRO.NS",
    "sbi":"SBIN.NS","state bank":"SBIN.NS","state bank of india":"SBIN.NS",
    "axis bank":"AXISBANK.NS",
    "kotak":"KOTAKBANK.NS","kotak bank":"KOTAKBANK.NS",
    "bajaj finance":"BAJFINANCE.NS",
    "maruti":"MARUTI.NS","maruti suzuki":"MARUTI.NS",
    "tata motors":"TATAMOTORS.NS",
    "hul":"HINDUNILVR.NS","hindustan unilever":"HINDUNILVR.NS",
    "sun pharma":"SUNPHARMA.NS",
    "airtel":"BHARTIARTL.NS","bharti airtel":"BHARTIARTL.NS",
    "adani ports":"ADANIPORTS.NS","adani enterprises":"ADANIENT.NS",
    "asian paints":"ASIANPAINT.NS",
    "hcl":"HCLTECH.NS","hcl tech":"HCLTECH.NS",
    "titan":"TITAN.NS",
    "nestle":"NESTLEIND.NS","nestle india":"NESTLEIND.NS",
    "l&t":"LT.NS","larsen":"LT.NS",
    "ongc":"ONGC.NS","ntpc":"NTPC.NS","power grid":"POWERGRID.NS",
    "dr reddy":"DRREDDY.NS","cipla":"CIPLA.NS",
    "bajaj auto":"BAJAJ-AUTO.NS","hero motocorp":"HEROMOTOCO.NS",
    "tech mahindra":"TECHM.NS","indusind bank":"INDUSINDBK.NS",
    "tata steel":"TATASTEEL.NS","jsw steel":"JSWSTEEL.NS",
    "hindalco":"HINDALCO.NS","coal india":"COALINDIA.NS",
    "bpcl":"BPCL.NS","ioc":"IOC.NS","indian oil":"IOC.NS",
    "ltimindtree":"LTIM.NS","ultratech cement":"ULTRACEMCO.NS",
    "grasim":"GRASIM.NS","divis lab":"DIVISLAB.NS","eicher motors":"EICHERMOT.NS",
    "nifty":"^NSEI","nifty 50":"^NSEI","sensex":"^BSESN",
    "bank nifty":"^NSEBANK","nifty it":"^CNXIT",
    "gold":"GLD","gold etf":"GLD","silver":"SLV",
    "oil":"USO","crude oil":"USO",
    "real estate":"VNQ","reit":"VNQ",
    "bitcoin":"BTC-USD","btc":"BTC-USD",
    "ethereum":"ETH-USD","eth":"ETH-USD","crypto":"BTC-USD",
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","tesla":"TSLA","nvidia":"NVDA","meta":"META","netflix":"NFLX",
    "s&p 500":"SPY","sp500":"SPY","sp 500":"SPY",
    "nasdaq":"QQQ","dow jones":"DIA","us bonds":"TLT","bonds":"TLT",
    "emerging markets":"EEM",
}

# ─────────────────────────────────────────────────────────────
#  STRESS SCENARIOS
#  Each scenario defines percentage shocks per asset class.
#  Shocks are applied proportionally based on asset beta / sector.
# ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "2008 Global Financial Crisis": {
        "description": "Lehman collapse, global credit freeze (Sep 2008 – Mar 2009)",
        "equity_shock":   -55.0,
        "bond_shock":       8.0,
        "gold_shock":       5.0,
        "oil_shock":       -70.0,
        "crypto_shock":     0.0,   # didn't exist
        "reit_shock":      -68.0,
        "default_shock":   -45.0,
        "color": "#ef4444",
        "icon": "💥",
    },
    "COVID-19 Crash (Mar 2020)": {
        "description": "Global pandemic panic, fastest 30% drop in history",
        "equity_shock":   -34.0,
        "bond_shock":      10.0,
        "gold_shock":      -8.0,
        "oil_shock":       -65.0,
        "crypto_shock":   -50.0,
        "reit_shock":      -40.0,
        "default_shock":   -30.0,
        "color": "#f97316",
        "icon": "🦠",
    },
    "Dot-com Bust (2000–2002)": {
        "description": "Tech bubble collapse, NASDAQ fell ~78%",
        "equity_shock":   -49.0,
        "bond_shock":      15.0,
        "gold_shock":      12.0,
        "oil_shock":       -35.0,
        "crypto_shock":     0.0,
        "reit_shock":      -20.0,
        "default_shock":   -40.0,
        "color": "#a78bfa",
        "icon": "💻",
    },
    "India Demonetisation (Nov 2016)": {
        "description": "Sudden INR demonetisation — Nifty fell ~6% in weeks",
        "equity_shock":   -8.0,
        "bond_shock":      3.0,
        "gold_shock":      2.0,
        "oil_shock":       -5.0,
        "crypto_shock":   -10.0,
        "reit_shock":      -8.0,
        "default_shock":   -7.0,
        "color": "#f59e0b",
        "icon": "🏦",
    },
    "Russia–Ukraine War (Feb 2022)": {
        "description": "Geopolitical shock, energy prices spiked, markets sold off",
        "equity_shock":   -12.0,
        "bond_shock":      -5.0,
        "gold_shock":      10.0,
        "oil_shock":       65.0,
        "crypto_shock":   -20.0,
        "reit_shock":      -15.0,
        "default_shock":   -10.0,
        "color": "#34d399",
        "icon": "⚔️",
    },
    "Interest Rate Spike (+300bps)": {
        "description": "Central bank aggressive rate hike cycle (like Fed 2022)",
        "equity_shock":   -20.0,
        "bond_shock":      -25.0,
        "gold_shock":      -8.0,
        "oil_shock":        5.0,
        "crypto_shock":   -45.0,
        "reit_shock":      -30.0,
        "default_shock":   -18.0,
        "color": "#60a5fa",
        "icon": "📈",
    },
    "Severe Inflation (1970s Style)": {
        "description": "Runaway inflation — equities suffer, commodities soar",
        "equity_shock":   -30.0,
        "bond_shock":      -20.0,
        "gold_shock":      80.0,
        "oil_shock":       50.0,
        "crypto_shock":   -15.0,
        "reit_shock":       5.0,
        "default_shock":   -25.0,
        "color": "#e879f9",
        "icon": "🔥",
    },
    "Flash Crash / Black Monday": {
        "description": "Single-day extreme market event (like Oct 1987 −22%)",
        "equity_shock":   -22.0,
        "bond_shock":       5.0,
        "gold_shock":       1.0,
        "oil_shock":       -8.0,
        "crypto_shock":   -30.0,
        "reit_shock":      -20.0,
        "default_shock":   -18.0,
        "color": "#fb7185",
        "icon": "⚡",
    },
}

# Ticker → asset class mapping
def classify_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t in ("GLD","IAU","SGOL","GOLD"): return "gold"
    if t in ("SLV","SIVR"):              return "gold"   # precious metals
    if t in ("USO","UCO","BNO"):         return "oil"
    if t in ("BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD"): return "crypto"
    if t in ("TLT","IEF","SHY","AGG","BND","LQD"): return "bond"
    if t in ("VNQ","IYR","SCHH"):        return "reit"
    return "equity"   # default: equity / index


def resolve_ticker(name: str) -> str:
    key = name.strip().lower()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    upper = name.strip().upper()
    if name.strip() == upper and len(upper) <= 12 and " " not in upper:
        return upper
    return upper.replace(" ", "") + ".NS"


def fetch_prices(tickers: list, years: int = 3) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365.25))
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all")
    available = [t for t in tickers if t in prices.columns]
    return prices[available].dropna()


def compute_stats(prices: pd.DataFrame):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    ann_ret = (log_ret.mean()  * 252 * 100).to_dict()
    ann_vol = (log_ret.std()   * np.sqrt(252) * 100).to_dict()
    corr    = log_ret.corr().to_dict()
    cov_df  = log_ret.cov() * 252 * 10000
    return ann_ret, ann_vol, corr, cov_df, log_ret


def portfolio_perf(weights, tickers, ann_ret, cov_df):
    w  = np.array([weights[t] for t in tickers])
    r  = float(w @ np.array([ann_ret[t] for t in tickers]))
    v2 = float(w @ cov_df.loc[tickers, tickers].values @ w)
    return r, float(np.sqrt(max(v2, 0)))


def monte_carlo(ret_pct, vol_pct, years=10, n=600, initial=100000):
    mu, sig = ret_pct / 100, vol_pct / 100
    rng     = np.random.default_rng(42)
    z       = rng.standard_normal((n, years))
    paths   = np.zeros((n, years + 1))
    paths[:, 0] = initial
    for t in range(years):
        paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5*sig**2) + sig*z[:, t])
    return paths


def efficient_frontier_data(tickers, ann_ret, cov_df):
    if len(tickers) < 2:
        return [], [], []
    k   = len(tickers)
    rng = np.random.default_rng(7)
    vols, rets, sharpes = [], [], []
    for _ in range(3000):
        w_arr  = rng.dirichlet(np.ones(k))
        w_dict = {t: float(w_arr[i]) for i, t in enumerate(tickers)}
        r, v   = portfolio_perf(w_dict, tickers, ann_ret, cov_df)
        vols.append(round(v, 3))
        rets.append(round(r, 3))
        sharpes.append(round((r - 6.5) / v if v > 0 else 0, 3))
    return vols, rets, sharpes


# ─────────────────────────────────────────────────────────────
#  STRESS TESTING ENGINE
# ─────────────────────────────────────────────────────────────

def compute_stress(weights_dict, tickers, investment, log_ret_df, ann_ret, ann_vol):
    """
    For each scenario, compute:
    - Portfolio P&L under the shock
    - Asset-level contribution to loss
    - Recovery estimate based on historical volatility
    - Historical drawdown for the scenario period (if data available)
    """
    results = []

    for scenario_name, params in SCENARIOS.items():
        asset_impacts = {}
        portfolio_loss_pct = 0.0

        for ticker in tickers:
            w      = weights_dict.get(ticker, 0)
            cls    = classify_ticker(ticker)
            shock  = params.get(f"{cls}_shock", params["default_shock"]) / 100

            # Scale shock by asset's beta-like sensitivity (vol ratio vs market)
            # Higher-vol assets amplify the shock more in crashes
            asset_vol  = ann_vol.get(ticker, 20)
            market_vol = 18.0  # reference equity vol
            if cls == "equity":
                # amplify for high-vol stocks
                beta_scale = min(asset_vol / market_vol, 2.5)
                adj_shock  = shock * beta_scale
            else:
                adj_shock = shock

            adj_shock = max(adj_shock, -0.95)   # floor at -95%

            asset_loss_pct  = adj_shock * 100
            asset_loss_amt  = w * investment * adj_shock
            portfolio_loss_pct += w * adj_shock

            asset_impacts[ticker] = {
                "shock_pct":    round(adj_shock * 100, 2),
                "loss_amt":     round(asset_loss_amt, 2),
                "weight":       round(w * 100, 2),
                "contribution": round(w * adj_shock * 100, 2),
                "asset_class":  cls,
            }

        total_loss_amt = portfolio_loss_pct * investment

        # Recovery estimate: months to recover using expected annual return
        port_ret   = sum(weights_dict.get(t, 0) * ann_ret.get(t, 8) for t in tickers)
        port_vol   = ann_vol.get(list(tickers)[0], 20) if len(tickers) == 1 else 18
        monthly_ret = port_ret / 100 / 12

        if portfolio_loss_pct < 0 and monthly_ret > 0:
            # Months to recover: solve (1+r)^n = 1/(1+loss)
            recovery_months = round(
                np.log(1 / (1 + portfolio_loss_pct)) / np.log(1 + monthly_ret)
            )
        else:
            recovery_months = 0

        # Survival value
        survival_value = investment * (1 + portfolio_loss_pct)

        results.append({
            "name":             scenario_name,
            "description":      params["description"],
            "color":            params["color"],
            "icon":             params["icon"],
            "portfolio_loss_pct": round(portfolio_loss_pct * 100, 2),
            "portfolio_loss_amt": round(total_loss_amt, 2),
            "survival_value":   round(survival_value, 2),
            "recovery_months":  max(recovery_months, 0),
            "asset_impacts":    asset_impacts,
        })

    # Sort worst-first
    results.sort(key=lambda x: x["portfolio_loss_pct"])
    return results


def compute_sensitivity(weights_dict, tickers, investment, ann_ret, cov_df):
    """
    Sensitivity analysis: how does portfolio return/vol/sharpe change as each
    asset's weight shifts ±5%, ±10%, ±20%?
    Returns a matrix for a heatmap.
    """
    rf = 6.5
    base_ret, base_vol = portfolio_perf(weights_dict, tickers, ann_ret, cov_df)
    base_sharpe = (base_ret - rf) / base_vol if base_vol > 0 else 0

    shifts = [-20, -10, -5, 5, 10, 20]
    sensitivity = []

    for ticker in tickers:
        row = {"ticker": ticker, "shifts": []}
        for shift in shifts:
            # Shift this ticker's weight by shift%, redistribute proportionally
            new_w = dict(weights_dict)
            delta  = shift / 100
            if new_w[ticker] + delta < 0:
                row["shifts"].append(None)
                continue
            new_w[ticker] = new_w[ticker] + delta

            # Normalise
            total = sum(new_w.values())
            new_w = {t: v / total for t, v in new_w.items()}

            r, v = portfolio_perf(new_w, tickers, ann_ret, cov_df)
            sh   = (r - rf) / v if v > 0 else 0
            row["shifts"].append({
                "shift":         shift,
                "ret_delta":     round(r - base_ret, 3),
                "vol_delta":     round(v - base_vol, 3),
                "sharpe_delta":  round(sh - base_sharpe, 4),
                "sharpe":        round(sh, 4),
            })
        sensitivity.append(row)

    return sensitivity, shifts


def compute_drawdown(log_ret_df, weights_dict, tickers):
    """Compute historical portfolio drawdown from daily log returns."""
    valid = [t for t in tickers if t in log_ret_df.columns]
    if not valid:
        return [], [], 0

    w_arr      = np.array([weights_dict.get(t, 0) for t in valid])
    port_ret   = (log_ret_df[valid] * w_arr).sum(axis=1)
    cum_ret    = (1 + port_ret).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown   = ((cum_ret - rolling_max) / rolling_max * 100).round(3)

    max_dd = float(drawdown.min())

    # sample every 5 days for chart
    step   = max(1, len(drawdown) // 200)
    dates  = [d.strftime("%Y-%m-%d") for d in drawdown.index[::step]]
    values = drawdown.values[::step].tolist()

    return dates, values, round(max_dd, 2)


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data             = request.json
    portfolio_assets = data.get("portfolio", [])
    new_asset        = data.get("new_asset", {})
    investment       = float(data.get("investment", 100000))
    rf               = float(data.get("risk_free_rate", 6.5))
    lookback_years   = int(data.get("lookback_years", 3))

    p_items = [{"name": a["name"],
                "ticker": resolve_ticker(a["name"]),
                "weight": float(a["weight"])} for a in portfolio_assets]

    new_ticker = resolve_ticker(new_asset["name"])
    new_weight = float(new_asset.get("weight", 10))

    all_tickers = list(dict.fromkeys([x["ticker"] for x in p_items] + [new_ticker]))

    try:
        prices = fetch_prices(all_tickers, lookback_years)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {str(e)}"}), 500

    if prices.empty:
        return jsonify({"error": "No price data returned. Check asset names."}), 400

    valid_tickers = list(prices.columns)
    valid_p_items = [x for x in p_items if x["ticker"] in valid_tickers]
    new_valid     = new_ticker in valid_tickers

    if not valid_p_items:
        return jsonify({"error": "Could not fetch data for any portfolio asset."}), 400

    ann_ret, ann_vol, corr, cov_df, log_ret_df = compute_stats(prices)

    # ── Before ──
    total_w_before = sum(x["weight"] for x in valid_p_items) or 1
    w_before       = {x["ticker"]: x["weight"] / total_w_before for x in valid_p_items}
    tickers_before = list(w_before.keys())
    ret_b, vol_b   = portfolio_perf(w_before, tickers_before, ann_ret, cov_df)

    # ── After ──
    alloc_frac    = new_weight / 100
    tickers_after = tickers_before + (
        [new_ticker] if new_valid and new_ticker not in tickers_before else [])

    if new_valid:
        w_after = {t: w_before[t] * (1 - alloc_frac) for t in tickers_before}
        w_after[new_ticker] = alloc_frac
        ret_a, vol_a = portfolio_perf(w_after, tickers_after, ann_ret, cov_df)
    else:
        w_after, ret_a, vol_a = w_before, ret_b, vol_b

    def sharpe_r(r, v): return round((r - rf) / v, 4) if v > 0 else 0
    def sortino_r(r, v): return round((r - rf) / (v * 0.7), 4) if v > 0 else 0
    def calmar_r(r, v): return round(r / (v * 0.5), 4) if v > 0 else 0
    def var95(r, v): return round(r - 1.645 * v, 4)
    def cvar95(r, v):
        z = norm.ppf(0.05)
        return round(r + v * norm.pdf(z) / 0.05 * -1, 4)

    # ── Monte Carlo ──
    mc_paths = monte_carlo(ret_a, vol_a, years=10, n=600, initial=investment)
    mc_years = np.arange(11)
    mc_p10   = [round(float(np.percentile(mc_paths[:, y], 10)), 2) for y in mc_years]
    mc_p50   = [round(float(np.percentile(mc_paths[:, y], 50)), 2) for y in mc_years]
    mc_p90   = [round(float(np.percentile(mc_paths[:, y], 90)), 2) for y in mc_years]
    p_loss   = round(float(np.mean(mc_paths[:, -1] < investment) * 100), 2)

    # ── Efficient Frontier ──
    ef_vols, ef_rets, ef_sharpes = efficient_frontier_data(tickers_after, ann_ret, cov_df)
    max_sh_idx = int(np.argmax(ef_sharpes)) if ef_sharpes else 0

    # ── Price History ──
    prices_1yr  = prices[tickers_before].tail(252)
    norm_prices = (prices_1yr / prices_1yr.iloc[0] * 100).round(2)
    price_dates  = [d.strftime("%Y-%m-%d") for d in norm_prices.index]
    price_series = {t: norm_prices[t].tolist() for t in tickers_before}

    # ── Asset Stats ──
    asset_stats = []
    for item in valid_p_items:
        t = item["ticker"]
        asset_stats.append({
            "name":   item["name"], "ticker": t,
            "weight": round(item["weight"] / total_w_before * (1 - alloc_frac) * 100, 2),
            "ret":    round(ann_ret.get(t, 0), 2),
            "vol":    round(ann_vol.get(t, 0), 2),
            "class":  classify_ticker(t),
        })
    if new_valid:
        asset_stats.append({
            "name":   new_asset["name"], "ticker": new_ticker,
            "weight": new_weight,
            "ret":    round(ann_ret.get(new_ticker, 0), 2),
            "vol":    round(ann_vol.get(new_ticker, 0), 2),
            "class":  classify_ticker(new_ticker),
        })

    # ── STRESS TESTING ──
    stress_before = compute_stress(w_before, tickers_before, investment, log_ret_df, ann_ret, ann_vol)
    stress_after  = compute_stress(w_after,  tickers_after,  investment, log_ret_df, ann_ret, ann_vol)

    # ── SENSITIVITY ANALYSIS ──
    sensitivity, shift_labels = compute_sensitivity(w_after, tickers_after, investment, ann_ret, cov_df)

    # ── DRAWDOWN ──
    dd_dates_b, dd_vals_b, max_dd_b = compute_drawdown(log_ret_df, w_before, tickers_before)
    dd_dates_a, dd_vals_a, max_dd_a = compute_drawdown(log_ret_df, w_after,  tickers_after)

    return jsonify({
        "ticker_map":    {x["name"]: x["ticker"] for x in p_items},
        "new_ticker":    new_ticker,
        "new_valid":     new_valid,
        "asset_stats":   asset_stats,
        "lookback_years": lookback_years,
        "investment":    investment,
        "before": {
            "ret": round(ret_b, 3), "vol": round(vol_b, 3),
            "sharpe": sharpe_r(ret_b, vol_b), "sortino": sortino_r(ret_b, vol_b),
            "calmar": calmar_r(ret_b, vol_b), "var95": var95(ret_b, vol_b),
            "cvar95": cvar95(ret_b, vol_b), "max_dd": max_dd_b,
        },
        "after": {
            "ret": round(ret_a, 3), "vol": round(vol_a, 3),
            "sharpe": sharpe_r(ret_a, vol_a), "sortino": sortino_r(ret_a, vol_a),
            "calmar": calmar_r(ret_a, vol_a), "var95": var95(ret_a, vol_a),
            "cvar95": cvar95(ret_a, vol_a), "max_dd": max_dd_a,
        },
        "monte_carlo": {
            "labels": [f"Yr {y}" for y in mc_years],
            "p10": mc_p10, "p50": mc_p50, "p90": mc_p90,
            "prob_loss": p_loss,
            "final_p10": mc_p10[-1], "final_p50": mc_p50[-1], "final_p90": mc_p90[-1],
        },
        "frontier": {
            "vols": ef_vols, "rets": ef_rets, "sharpes": ef_sharpes,
            "max_sharpe_idx": max_sh_idx,
        },
        "price_history": {"dates": price_dates, "series": price_series},
        "drawdown": {
            "before": {"dates": dd_dates_b, "values": dd_vals_b, "max": max_dd_b},
            "after":  {"dates": dd_dates_a, "values": dd_vals_a, "max": max_dd_a},
        },
        "stress": {
            "before":      stress_before,
            "after":       stress_after,
            "scenario_names": [s["name"] for s in stress_before],
        },
        "sensitivity": {
            "data":   sensitivity,
            "shifts": shift_labels,
        },
    })






if __name__ == "__main__":
    print("\n  Portfolio Risk Analyzer API")
    print("  ─────────────────────────────")
    print("  http://localhost:5000")
    app.run(debug=True, port=5000)
