# Portfolio Risk Analyzer

A full-stack portfolio analytics platform that fetches live market data, computes institutional-grade risk metrics, runs stress tests against 8 historical crises.

---

## Features

- **Live market data** — fetches price history for 70+ Indian and global assets via yfinance
- **7 risk metrics** — Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%, Max Drawdown, correlation matrix
- **Before vs After analysis** — see exactly how adding a new asset changes every metric
- **Efficient Frontier** — 3,000 random portfolio simulations to find optimal allocations
- **Monte Carlo** — 600 GBM paths over 10 years with P10/P50/P90 outcome bands
- **Stress Testing** — 8 historical crisis scenarios with per-asset shock contributions
- **Sensitivity Analysis** — Sharpe heatmap for ±5%, ±10%, ±20% weight shifts per asset
- **Custom Shock** — define your own crisis scenario by asset class


---

## Project Structure

```
portfolio-risk-analyzer/
├── app.py          # Flask backend — data fetching, analytics, AI endpoint
└── index.html      # Single-page frontend — 5 tabs, Chart.js, streaming UI
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install flask flask-cors yfinance numpy scipy pandas google-generativeai
```

### 2. Run the backend

```bash
python app.py
```

### 3. Open the frontend

Open `index.html` in your browser

---

## How to Use

**Only two inputs are needed: asset names and weights.** Everything else is fetched live.

### Step 1 — Build your portfolio
Enter asset names and weights in the left panel. Use plain English names:

| What you type | Resolves to |
|---|---|
| `ICICI Bank` | `ICICIBANK.NS` |
| `Reliance` | `RELIANCE.NS` |
| `Gold` | `GLD` |
| `Nifty 50` | `^NSEI` |
| `Bitcoin` | `BTC-USD` |
| `Apple` | `AAPL` |

Over 70 aliases are supported. You can also type raw tickers directly (e.g. `TCS.NS`, `NVDA`, `BTC-USD`).

### Step 2 — Pick a new asset to evaluate
Type the name of any asset and its target allocation (%). The tool will fetch its data and show you the exact impact on every metric.

### Step 3 — Hit Analyze
The backend fetches 3 years of daily prices, computes all metrics, runs Monte Carlo and stress tests, and returns results in under 10 seconds.

### Step 4 — Explore the 4 tabs
| Tab | What it shows |
|---|---|
| **Analytics** | Efficient frontier, drawdown chart, full metrics table, asset stats, Monte Carlo |
| **Stress Testing** | 8 crisis scenarios — click any card for per-asset breakdown |
| **Sensitivity** | Sharpe ratio heatmap, weight shift impact chart |
| **Custom Shock** | Set your own shocks per asset class (equities, bonds, gold, oil, crypto, REITs) |


---

## Supported Assets

### Indian Stocks (NSE)
ICICI Bank, Reliance, Infosys, TCS, HDFC Bank, Wipro, SBI, Axis Bank, Kotak, Bajaj Finance, Maruti, Tata Motors, HUL, Sun Pharma, Airtel, Adani Ports, Asian Paints, HCL Tech, Titan, Nestle, L&T, ONGC, NTPC, Dr Reddy, Cipla, Tata Steel, JSW Steel, and more.

### Indian Indices
Nifty 50, Sensex, Bank Nifty, Nifty IT

### Global / Commodities
Gold (GLD), Silver (SLV), Crude Oil (USO), US REITs (VNQ), Bitcoin, Ethereum

### US Stocks & Indices
Apple, Microsoft, Google, Amazon, Tesla, Nvidia, Meta, Netflix, S&P 500 (SPY), NASDAQ (QQQ), US Bonds (TLT), Emerging Markets (EEM)

---

## Stress Test Scenarios

| Scenario | Description | Equity Shock |
|---|---|---|
| 2008 Global Financial Crisis | Lehman collapse, global credit freeze | −55% |
| COVID-19 Crash (Mar 2020) | Fastest 30% market drop in history | −34% |
| Dot-com Bust (2000–2002) | Tech bubble collapse, NASDAQ −78% | −49% |
| India Demonetisation (Nov 2016) | Sudden INR demonetisation | −8% |
| Russia–Ukraine War (Feb 2022) | Geopolitical shock, energy spike | −12% |
| Interest Rate Spike (+300bps) | Aggressive central bank hike cycle | −20% |
| Severe Inflation (1970s style) | Runaway inflation, commodities surge | −30% |
| Flash Crash / Black Monday | Single-day extreme market event | −22% |

Shocks are **volatility-adjusted** — high-beta equities amplify the shock proportionally. Each scenario also shows estimated recovery time based on the portfolio's expected return.




---


## Configuration

All settings are adjustable in the UI:

| Setting | Default | Description |
|---|---|---|
| Lookback period | 3 years | Historical data window for return/vol/correlation computation |
| Risk-free rate | 6.5% | Used in Sharpe, Sortino calculations (set to local rate — India ~6.5%) |
| Investment amount | ₹1,00,000 | Starting capital for Monte Carlo and stress test P&L |


