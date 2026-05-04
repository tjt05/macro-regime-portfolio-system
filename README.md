# PRAIDS: Macro-Regime Aware Personalized Investment Decision System

PRAIDS is a local, research-grade investment decision-support system that maps macro market regimes into personalized, practical portfolio recommendations.

It is **not a trading bot** and does not place live trades. The goal is to study whether macro-aware regime detection, rule-based interpretation, and user-specific constraints can produce investment decisions that are easier to understand and evaluate than a black-box allocation model.

## Problem

Most simple investment demos either:

- compare only equities versus bonds,
- produce tiny allocations across too many assets to be practical,
- ignore the user's financial situation,
- or show a backtest without explaining the market environment behind each decision.

PRAIDS addresses this by combining:

- unsupervised regime detection,
- macro asset-class interpretation,
- personalized allocation rules,
- practical implementation constraints,
- simulation against an SPY benchmark,
- decision logs and live journaling.

The system answers:

> Given the current macro regime and my personal risk profile, what portfolio allocation is reasonable, practical, and explainable?

## What The System Does

1. Downloads historical market data with `yfinance`.
2. Builds macro-aware technical and cross-asset features.
3. Trains a KMeans clustering model to detect market regimes.
4. Interprets clusters as economic regimes.
5. Generates an investment action: `ENTER`, `ADD`, `HOLD`, `REDUCE`, or `EXIT`.
6. Creates an ideal allocation using regime-specific model portfolios.
7. Personalizes the allocation using user profile rules.
8. Simplifies the allocation into a practical portfolio using implementation constraints.
9. Simulates performance against SPY buy-and-hold.
10. Evaluates performance using return, volatility, drawdown, Sharpe ratio, regime-wise analysis, and profile-based comparison.
11. Provides a Streamlit dashboard and FastAPI backend.

## Asset Universe

The model uses a diversified macro asset universe:

| Asset Group | Instruments |
|---|---|
| Growth equities | `SPY`, `QQQ`, `IWM` |
| Defensive equities | `XLP`, `XLV`, `XLU` |
| Treasuries / risk-off | `TLT`, `IEF`, `SHY` |
| Inflation hedges | `GLD`, `DBC` |
| Liquidity / risk sentiment | `BTC` via `BTC-USD` |
| Cash proxy | `CASH`, modeled as 0% return |

The broad universe is used for regime detection, but the final actionable portfolio can be constrained to a smaller number of holdings.

## Approach

### 1. Feature Engineering

For each asset, PRAIDS computes:

- price / 200-day moving average,
- 20-day return,
- 60-day return,
- 20-day volatility,
- rolling drawdown.

It also computes cross-asset macro signals:

- `SPY / TLT`: risk appetite,
- `SPY / GLD`: growth versus inflation hedge,
- `QQQ / IWM`: growth versus small-cap risk,
- `BTC / SPY`: liquidity and risk sentiment,
- commodity versus equity relative strength.

The current expanded feature matrix contains **68 features**.

### 2. Regime Detection

PRAIDS uses KMeans clustering with 5 clusters by default. The model is unsupervised: it does not train on predefined bull/bear labels.

Detected clusters are interpreted into macro regimes using centroid behavior across equities, bonds, commodities, gold, and BTC.

Current macro labels:

- `growth_expansion`
- `inflation_shock`
- `recession_risk_off`
- `liquidity_risk_on`
- `neutral_transition`

### 3. Macro Interpretation

The macro interpreter maps numeric clusters to economic labels by scoring centroid behavior:

- strong broad equity momentum -> growth expansion,
- strong gold/commodity momentum -> inflation shock,
- weak equities + stronger bonds -> recession/risk-off,
- strong QQQ/BTC behavior -> liquidity-driven risk-on,
- mixed signals -> neutral/transition.

This keeps the model interpretable instead of treating clusters as arbitrary IDs.

### 4. Personalization

The user profile affects the allocation:

```python
{
    "income": 1500,
    "expenses": 900,
    "savings": 5000,
    "risk_tolerance": "medium",
    "experience_level": "beginner",
    "investment_horizon": "long"
}
```

Examples:

- low savings buffer increases `SHY`, `IEF`, and cash-like exposure,
- beginner profile reduces high-volatility exposure such as `QQQ`, `IWM`, `BTC`, and `DBC`,
- high risk tolerance increases equity and BTC exposure,
- low risk tolerance increases defensive equities, bonds, gold, and cash.

### 5. Practical Implementation Layer

A key design issue is that a model may recommend many small 2-5% positions, which is often impractical for a retail investor.

PRAIDS solves this with an implementation layer:

- maximum number of assets to hold,
- minimum position size,
- user-defined avoid list,
- top-weight asset selection,
- automatic renormalization.

This means PRAIDS can use many assets for macro detection while still outputting a compact, actionable portfolio.

Example:

```text
Ideal model allocation: 13 assets
Implementation settings: max_assets = 4, min_position_weight = 8%, avoid BTC
Actionable allocation: SPY + QQQ only
```

### 6. Simulation And Journaling

PRAIDS separates historical simulation from real-life journaling.

Simulation mode:

- backtests the policy-driven portfolio from a selected start date,
- compares against SPY buy-and-hold,
- tracks target allocation, actual allocation, portfolio value, and benchmark value.

Live journal mode:

- records the current recommendation,
- distinguishes market-data date from intended execution date,
- lets the user record whether they followed, deferred, ignored, or customized the recommendation,
- stores structured trade plans by asset.

## Results

The following results are from a sample balanced profile with:

- start date: `2023-01-01`,
- starting capital: `$10,000`,
- monthly rebalance policy,
- 5 trading-day cooldown,
- maximum 4 holdings,
- minimum position size of 8%.

### Overall Performance

| Metric | PRAIDS Strategy | SPY Buy-and-Hold |
|---|---:|---:|
| Total return | 64.6% | 96.6% |
| Annualized volatility | 10.1% | 12.6% |
| Max drawdown | -9.8% | -18.8% |
| Sharpe ratio | 1.07 | 1.17 |
| Ending value | $16,458 | $19,657 |

These figures were generated from the local PRAIDS pipeline using available Yahoo Finance data as of the latest cached market date in this workspace. Results may change when data is refreshed or the model is retrained.

SPY outperformed the balanced PRAIDS strategy over this sample period. This is not surprising: SPY is a 100% equity benchmark and can dominate diversified strategies during strong equity markets.

The value of PRAIDS is not that it always beats SPY. Its purpose is to provide:

- explainable regime-aware decisions,
- controlled risk exposure,
- personalized allocations,
- practical portfolio constraints,
- diagnostics across regimes and user profiles.

### Profile-Based Evaluation

| Profile | Total Return | Volatility | Max Drawdown | Sharpe | Ending Value |
|---|---:|---:|---:|---:|---:|
| Conservative | 22.9% | 4.4% | -4.1% | 0.98 | $12,292 |
| Balanced | 64.6% | 10.1% | -9.8% | 1.07 | $16,458 |
| Aggressive | 100.1% | 13.1% | -18.1% | 1.16 | $20,010 |

This shows the intended tradeoff: conservative profiles reduce drawdown and volatility, while aggressive profiles increase return potential and risk.

### Regime-Wise Evaluation

PRAIDS also evaluates strategy performance by detected macro regime. This helps answer:

- which regimes contributed most to returns,
- where the allocation logic underperformed,
- whether defensive regimes are actually reducing risk,
- whether the model is adding value beyond a static benchmark.

The dashboard includes regime-wise return, volatility, Sharpe ratio, drawdown, and benchmark comparison tables.

## Dashboard

The Streamlit frontend includes:

- current macro regime,
- regime explanation,
- recommended action,
- ideal model allocation,
- actionable allocation,
- actual allocation,
- portfolio curve versus SPY,
- benchmark comparison,
- regime-wise performance,
- profile-based evaluation,
- simulated decision history,
- simulated portfolio ledger,
- live journal entries,
- CSV downloads.

## Architecture

```text
PRAIDS/
├── backend/
│   ├── api.py
│   ├── pipeline.py
│   ├── assets.py
│   ├── data/
│   ├── model/
│   ├── macro/
│   ├── strategy/
│   ├── simulation/
│   └── journal/
├── frontend/
│   └── streamlit_app.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── main.py
```

Backend:

- FastAPI service,
- model training and prediction,
- macro interpretation,
- simulation and evaluation,
- journaling endpoints.

Frontend:

- Streamlit dashboard,
- simulation controls,
- implementation constraints,
- live journal form,
- result visualizations.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the command-line pipeline:

```bash
python main.py
```

Run backend and frontend separately:

```bash
uvicorn backend.api:app --reload --port 8000
PRAIDS_API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

## Run With Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

If ports are occupied:

```bash
FRONTEND_PORT=8505 BACKEND_PORT=8005 docker compose up --build
```

Then open:

```text
http://localhost:8505
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `GET` | `/profile/default` | Default user profile |
| `GET` | `/assets` | Tradable asset universe |
| `GET` | `/journal` | Simulated decision history |
| `GET` | `/ledger` | Simulated portfolio ledger |
| `GET` | `/live-journal` | User-recorded live journal entries |
| `POST` | `/run` | Run PRAIDS pipeline |
| `POST` | `/live-journal` | Save a live journal entry |

## Generated Artifacts

Runtime outputs are written to `artifacts/`:

```text
artifacts/prices.csv
artifacts/regime_model.joblib
artifacts/decision_journal.jsonl
artifacts/portfolio_ledger.jsonl
artifacts/live_journal.jsonl
```

These are ignored by Git because they are generated from local runs.

## Limitations

PRAIDS is a research and decision-support project, not financial advice.

Important limitations:

- KMeans clusters are sensitive to feature choices and retraining windows.
- The allocation rules are interpretable but hand-designed.
- Results are backtests, not live trading results.
- Transaction costs, taxes, slippage, and dividend treatment are simplified.
- SPY can outperform diversified strategies during strong bull markets.
- BTC history begins later than traditional ETFs, shortening the usable feature window.

## Future Improvements

- Walk-forward validation,
- HMM regime model option,
- transaction costs and tax-aware simulation,
- macroeconomic data such as CPI, rates, unemployment, and yield curve features,
- allocation optimization under risk and simplicity constraints,
- persistent user portfolios and actual trade reconciliation,
- improved regime-wise diagnostics and stress testing.

## Disclaimer

This project is for education, research, and decision support only. It does not provide financial advice and does not execute trades.

## AI Attribution

Parts of this project were developed with assistance from AI tools, including ChatGPT/Codex, for code generation, refactoring, documentation, README polishing, and debugging support. The project design decisions, validation, interpretation of results, and final responsibility for the work remain with the author.
