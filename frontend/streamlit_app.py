from __future__ import annotations

"""Streamlit frontend for PRAIDS.

The frontend is intentionally presentation-focused. It gathers user profile,
portfolio, and implementation settings; calls the FastAPI backend; and renders
simulation results, downloads, and live-journal forms.
"""

import os
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(page_title="PRAIDS", layout="wide")
st.title("PRAIDS: Personalized Regime-Aware Investment Decision System")
st.caption("Educational decision support only. No live trading.")

API_URL = os.getenv("PRAIDS_API_URL", "http://localhost:8000").rstrip("/")


@st.cache_data(ttl=300)
def get_default_profile() -> dict:
    """Fetch sidebar defaults from the backend."""
    response = requests.get(f"{API_URL}/profile/default", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def get_assets() -> list[str]:
    """Fetch the asset list used by allocation controls and trade rows."""
    response = requests.get(f"{API_URL}/assets", timeout=10)
    response.raise_for_status()
    return response.json()


def run_praids(profile: dict, portfolio_settings: dict, force_train: bool) -> dict:
    """Run the backend pipeline with the current sidebar settings."""
    response = requests.post(
        f"{API_URL}/run",
        json={
            "user_profile": profile,
            "portfolio_settings": portfolio_settings,
            "force_train": force_train,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def save_live_journal_entry(entry: dict) -> dict:
    """Persist one user-entered real-life decision record."""
    response = requests.post(f"{API_URL}/live-journal", json=entry, timeout=20)
    response.raise_for_status()
    return response.json()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Prepare a DataFrame for Streamlit CSV download buttons."""
    return frame.to_csv(index=False).encode("utf-8")


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """Flatten nested JSON records such as allocation dictionaries for tables."""
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


try:
    default_profile = get_default_profile()
    available_assets = get_assets()
except requests.RequestException as exc:
    st.error(f"Could not reach PRAIDS backend at {API_URL}. Start the backend and refresh. Details: {exc}")
    st.stop()

with st.sidebar:
    # These inputs determine how PRAIDS personalizes the strategy and how
    # practical the final allocation should be for a real retail investor.
    st.header("User Profile")
    income = st.number_input("Monthly income", min_value=0.0, value=float(default_profile["income"]), step=100.0)
    expenses = st.number_input("Monthly expenses", min_value=0.0, value=float(default_profile["expenses"]), step=100.0)
    savings = st.number_input("Savings", min_value=0.0, value=float(default_profile["savings"]), step=500.0)
    risk_tolerance = st.selectbox("Risk tolerance", ["low", "medium", "high"], index=1)
    experience_level = st.selectbox("Experience level", ["beginner", "intermediate", "advanced"], index=0)
    investment_horizon = st.selectbox("Investment horizon", ["short", "medium", "long"], index=2)

    st.header("Portfolio Settings")
    start_date = st.date_input("Start date", value=date(2023, 1, 1))
    starting_capital = st.number_input("Starting capital", min_value=100.0, value=10_000.0, step=500.0)
    rebalance_policy = st.selectbox(
        "Rebalance policy",
        ["monthly", "weekly", "threshold", "regime_change", "daily"],
        index=0,
    )
    min_rebalance_threshold = st.slider("Minimum allocation change", 0.0, 0.25, 0.05, 0.01)
    cooldown_days = st.number_input("Cooldown trading days", min_value=0, max_value=252, value=5, step=1)

    st.header("Implementation Settings")
    max_assets = st.number_input("Maximum assets to hold", min_value=1, max_value=len(available_assets), value=4, step=1)
    min_position_weight = st.slider("Minimum position size", 0.0, 0.25, 0.08, 0.01)
    avoid_assets = st.multiselect("Avoid assets", options=available_assets, default=[])

    force_train = st.checkbox("Retrain model", value=False)

profile = {
    "income": income,
    "expenses": expenses,
    "savings": savings,
    "risk_tolerance": risk_tolerance,
    "experience_level": experience_level,
    "investment_horizon": investment_horizon,
}

portfolio_settings = {
    "start_date": start_date.isoformat(),
    "starting_capital": starting_capital,
    "rebalance_policy": rebalance_policy,
    "min_rebalance_threshold": min_rebalance_threshold,
    "cooldown_days": cooldown_days,
    "max_assets": max_assets,
    "min_position_weight": min_position_weight,
    "avoid_assets": avoid_assets,
}

with st.spinner("Running PRAIDS pipeline..."):
    # The backend does all model, allocation, simulation, and logging work.
    try:
        output = run_praids(profile, portfolio_settings, force_train=force_train)
    except requests.RequestException as exc:
        st.error(f"PRAIDS backend request failed: {exc}")
        st.stop()

tab_simulation, tab_live, tab_method = st.tabs(["Simulation", "Live Journal", "Method Notes"])

with tab_simulation:
    # Simulation results are historical/backtest-style outputs, useful for
    # understanding how the PRAIDS rules would have behaved over time.
    metric_cols = st.columns(4)
    metric_cols[0].metric("Current Regime", output["current_regime"])
    metric_cols[1].metric("Action", output["action"])
    metric_cols[2].metric("Stable >= 3 Days", "Yes" if output["stable"] else "No")
    metric_cols[3].metric("Ending Value", f"${output['metrics']['ending_value']:,.0f}")
    st.info(output["regime_explanation"])

    left, right = st.columns([1, 2])

    with left:
        ideal_allocation_df = pd.DataFrame(
            {"asset": list(output["ideal_allocation"].keys()), "weight": list(output["ideal_allocation"].values())}
        )
        target_allocation_df = pd.DataFrame(
            {"asset": list(output["allocation"].keys()), "weight": list(output["allocation"].values())}
        )
        actual_allocation_df = pd.DataFrame(
            {"asset": list(output["actual_allocation"].keys()), "weight": list(output["actual_allocation"].values())}
        )
        st.subheader("Ideal Model Allocation")
        st.plotly_chart(
            px.pie(ideal_allocation_df, values="weight", names="asset", hole=0.35),
            width="stretch",
            key="simulation_ideal_allocation_pie",
        )

        st.subheader("Actionable Allocation")
        st.plotly_chart(
            px.pie(target_allocation_df, values="weight", names="asset", hole=0.35),
            width="stretch",
            key="simulation_target_allocation_pie",
        )
        st.caption(
            "Actionable allocation applies max-asset, minimum-position, and avoid-list constraints to the ideal model allocation."
        )

        st.subheader("Actual Allocation")
        st.plotly_chart(
            px.pie(actual_allocation_df, values="weight", names="asset", hole=0.35),
            width="stretch",
            key="simulation_actual_allocation_pie",
        )

        st.subheader("Benchmark Comparison")
        st.write(
            {
                "Strategy total return": f"{output['metrics']['total_return']:.1%}",
                "SPY total return": f"{output['metrics']['benchmark_return']:.1%}",
                "Strategy max drawdown": f"{output['metrics']['max_drawdown']:.1%}",
                "SPY max drawdown": f"{output['metrics']['benchmark_max_drawdown']:.1%}",
                "Strategy volatility": f"{output['metrics']['annualized_volatility']:.1%}",
                "Strategy Sharpe": f"{output['metrics']['sharpe_ratio']:.2f}",
                "SPY Sharpe": f"{output['metrics']['benchmark_sharpe_ratio']:.2f}",
            }
        )

        if output["metrics"]["total_return"] < output["metrics"]["benchmark_return"]:
            st.warning(
                "PRAIDS underperformed SPY in this simulation. That can happen because the strategy owns bonds, gold, "
                "and cash for risk control, while SPY is a concentrated equity benchmark that can dominate in long bull markets."
            )

    with right:
        curve = pd.DataFrame(output["portfolio_curve"])
        curve["date"] = pd.to_datetime(curve["date"])
        curve = curve.set_index("date")[["portfolio_value", "benchmark_value"]].rename(
            columns={"portfolio_value": "PRAIDS Strategy", "benchmark_value": "SPY Buy-and-Hold"}
        )
        st.subheader("Portfolio Curve")
        st.line_chart(curve)

        st.subheader("Recent Regimes")
        recent = pd.DataFrame(output["recent_regimes"])
        st.dataframe(recent.tail(20), width="stretch")

    st.subheader("Cluster Interpretation")
    mapping_df = pd.DataFrame(
        [{"cluster": cluster, "macro_regime": regime} for cluster, regime in output["regime_mapping"].items()]
    ).sort_values("cluster")
    st.dataframe(mapping_df, width="stretch")

    eval_left, eval_right = st.columns(2)

    with eval_left:
        st.subheader("Regime-wise Performance")
        regime_perf = pd.DataFrame(output["regime_performance"])
        if not regime_perf.empty:
            st.download_button(
                "Download Regime Performance CSV",
                data=csv_bytes(regime_perf),
                file_name="praids_regime_performance.csv",
                mime="text/csv",
                key="download_regime_performance",
            )
            st.dataframe(regime_perf, width="stretch")
            st.bar_chart(regime_perf.set_index("regime")[["strategy_return", "benchmark_return"]])

    with eval_right:
        st.subheader("Profile-based Evaluation")
        profile_perf = pd.DataFrame(output["profile_performance"])
        if not profile_perf.empty:
            st.download_button(
                "Download Profile Evaluation CSV",
                data=csv_bytes(profile_perf),
                file_name="praids_profile_evaluation.csv",
                mime="text/csv",
                key="download_profile_evaluation",
            )
            st.dataframe(profile_perf, width="stretch")
            st.bar_chart(profile_perf.set_index("profile")[["total_return", "annualized_volatility", "max_drawdown"]])

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader("Simulated Decision History")
        decision_log = output["decision_log"]
        if decision_log:
            decision_df = flatten_records(decision_log)
            st.download_button(
                "Download Decision History CSV",
                data=csv_bytes(decision_df),
                file_name="praids_decision_history.csv",
                mime="text/csv",
                key="download_decision_history",
            )
            show_all_decisions = st.toggle("Show all decision rows", value=False, key="show_all_decisions")
            st.dataframe(decision_df if show_all_decisions else decision_df.tail(50), width="stretch")
        else:
            st.info("No simulated decision entries yet.")

    with bottom_right:
        st.subheader("Simulated Portfolio Ledger")
        portfolio_ledger = output["portfolio_ledger"]
        if portfolio_ledger:
            ledger_df = flatten_records(portfolio_ledger)
            st.download_button(
                "Download Portfolio Ledger CSV",
                data=csv_bytes(ledger_df),
                file_name="praids_portfolio_ledger.csv",
                mime="text/csv",
                key="download_portfolio_ledger",
            )
            show_all_ledger = st.toggle("Show all ledger rows", value=False, key="show_all_ledger")
            st.dataframe(ledger_df if show_all_ledger else ledger_df.tail(50), width="stretch")
        else:
            st.info("No simulated ledger entries yet.")

with tab_live:
    # Live journal entries are forward-looking notes about what the user actually
    # chose to do after receiving the latest recommendation.
    recommendation = output["current_recommendation"]
    st.subheader("Current Recommendation")
    rec_cols = st.columns(4)
    rec_cols[0].metric("Market Data Date", recommendation["market_data_date"])
    rec_cols[1].metric("Signal Date", recommendation["signal_date"])
    rec_cols[2].metric("Intended Execution", recommendation["intended_execution_date"])
    rec_cols[3].metric("Recommended Action", recommendation["action"])

    live_left, live_right = st.columns([1, 2])

    with live_left:
        st.write({"Regime": recommendation["regime"], "Stable signal": "Yes" if output["stable"] else "No"})
        st.info(recommendation["regime_explanation"])
        live_target_df = pd.DataFrame(
            {
                "asset": list(recommendation["target_allocation"].keys()),
                "weight": list(recommendation["target_allocation"].values()),
            }
        )
        st.plotly_chart(
            px.pie(live_target_df, values="weight", names="asset", hole=0.35),
            width="stretch",
            key="live_target_allocation_pie",
        )
        st.caption(f"Selected assets: {', '.join(output['implementation_details']['selected_assets'])}")

    with live_right:
        st.write(
            "Use this tab for real-life journaling. PRAIDS records the recommendation, but you decide whether "
            "to follow, partially follow, ignore, defer, or customize it."
        )
        with st.form("live_journal_form"):
            user_action = st.selectbox(
                "What did you actually decide?",
                ["follow", "partial_follow", "ignore", "defer", "custom"],
            )
            actual_execution_date = st.date_input(
                "Actual execution date",
                value=pd.to_datetime(recommendation["intended_execution_date"]).date(),
            )
            amount_type = st.radio("Input amount as", ["dollars", "shares", "target_weight"], horizontal=True)
            st.write("Trade Plan")
            trade_rows = []
            for asset in output["assets"]:
                asset_cols = st.columns([1, 2, 2])
                asset_cols[0].write(asset)
                trade_type = asset_cols[1].selectbox(
                    f"{asset} trade",
                    ["hold", "buy", "sell"],
                    key=f"{asset}_trade_type",
                    label_visibility="collapsed",
                )
                amount = asset_cols[2].number_input(
                    f"{asset} amount",
                    min_value=0.0,
                    value=0.0,
                    step=100.0 if amount_type == "dollars" else 0.01,
                    key=f"{asset}_trade_amount",
                    label_visibility="collapsed",
                )
                trade_rows.append(
                    {
                        "asset": asset,
                        "trade_type": trade_type,
                        "amount_type": amount_type,
                        "amount": amount,
                    }
                )
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save Live Journal Entry")

        if submitted:
            entry = {
                "signal_date": recommendation["signal_date"],
                "market_data_date": recommendation["market_data_date"],
                "intended_execution_date": recommendation["intended_execution_date"],
                "recommended_regime": recommendation["regime"],
                "recommended_action": recommendation["action"],
                "recommended_allocation": recommendation["target_allocation"],
                "user_action": user_action,
                "actual_execution_date": actual_execution_date.isoformat(),
                "actual_trades": trade_rows,
                "notes": notes,
            }
            try:
                save_live_journal_entry(entry)
                st.success("Live journal entry saved.")
            except requests.RequestException as exc:
                st.error(f"Could not save live journal entry: {exc}")

    st.subheader("Live Journal Entries")
    live_journal = output["live_journal"]
    if live_journal:
        live_df = flatten_records(live_journal)
        st.download_button(
            "Download Live Journal CSV",
            data=csv_bytes(live_df),
            file_name="praids_live_journal.csv",
            mime="text/csv",
            key="download_live_journal",
        )
        show_all_live = st.toggle("Show all live journal rows", value=True, key="show_all_live")
        st.dataframe(live_df if show_all_live else live_df.tail(50), width="stretch")
    else:
        st.info("No live journal entries yet.")

with tab_method:
    st.subheader("How To Interpret PRAIDS")
    st.write(
        "PRAIDS is a regime-aware decision-support and simulation tool, not a promise to beat SPY. "
        "SPY buy-and-hold can outperform a diversified strategy during strong equity bull markets because it is 100% equities. "
        "PRAIDS deliberately holds defensive and hedge assets to reduce exposure during risk-off or inflation regimes."
    )
    st.write(
        "Possible improvements include walk-forward testing, transaction costs, regime-specific validation, better macro features "
        "such as rates and inflation data, less defensive base allocations, and objective optimization of allocations instead of fixed rules."
    )
    st.subheader("Implementation Layer")
    st.write(
        "The model can inspect many assets without requiring you to hold them all. PRAIDS first creates an ideal diversified allocation, "
        "then converts it into an actionable allocation using maximum asset count, minimum position size, and avoided assets. "
        "This makes the output more realistic for a retail investor who wants a compact portfolio."
    )
    st.subheader("Retrain Model")
    st.write(
        "Retrain model discards the saved KMeans model and fits a new one using the currently available historical feature matrix. "
        "Use it when new data has accumulated or when feature/regime logic changes. Do not tick it every run unless you intentionally "
        "want cluster assignments and regime mappings to refresh."
    )
    st.subheader("Rebalance Policies")
    st.write(
        {
            "daily": "Can rebalance every trading day if regime or allocation gap warrants it.",
            "weekly": "Only permits rebalancing when the calendar week changes.",
            "monthly": "Only permits rebalancing when the month changes; a practical long-term default.",
            "threshold": "Rebalances only when target vs actual allocation drift exceeds the threshold.",
            "regime_change": "Rebalances only when the stable macro regime changes and the allocation gap is large enough.",
        }
    )
