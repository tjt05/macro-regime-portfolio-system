from __future__ import annotations

"""End-to-end orchestration for the PRAIDS backend.

`run_pipeline` is the backend's main application service. It fetches data,
builds features, trains/loads the regime model, predicts regimes, filters noisy
signals, generates decisions, simulates portfolio performance, evaluates metrics,
and persists logs for the API/frontend.
"""

from backend.data.fetch_data import fetch_adjusted_close
from backend.data.features import compute_features, compute_trend_signal_frame, latest_trend_signals
from backend.journal.evaluator import evaluate_performance, evaluate_regime_performance
from backend.journal.logger import save_decision_history, save_portfolio_ledger
from backend.macro.regime_interpreter import REGIME_EXPLANATIONS
from backend.model.predict import predict_regimes
from backend.model.train import train_regime_model
from backend.simulation.portfolio import simulate_portfolio
from backend.strategy.allocation import base_allocation_for_regime
from backend.strategy.decision_engine import generate_decision
from backend.strategy.implementation import simplify_allocation
from backend.strategy.personalization import DEFAULT_USER_PROFILE, personalize_allocation
from backend.strategy.stability_filter import apply_stability_filter, is_regime_stable


def run_pipeline(
    user_profile: dict | None = None,
    portfolio_settings: dict | None = None,
    force_train: bool = False,
) -> dict:
    """Run the full PRAIDS data, model, decision, simulation, and logging flow."""
    profile = {**DEFAULT_USER_PROFILE, **(user_profile or {})}

    prices = fetch_adjusted_close()
    features = compute_features(prices)
    artifact = train_regime_model(features, force=force_train)
    predictions = predict_regimes(features, artifact)
    stable_regimes = apply_stability_filter(predictions["regime"])
    trend_signal_frame = compute_trend_signal_frame(prices).loc[features.index]

    current_regime = stable_regimes.iloc[-1]
    stable = is_regime_stable(predictions["regime"])
    trend_signals = latest_trend_signals(prices.loc[: features.index[-1]])
    action = generate_decision(current_regime, trend_signals, profile, stable=stable)
    ideal_allocation = personalize_allocation(base_allocation_for_regime(current_regime), profile)
    allocation, implementation_details = simplify_allocation(ideal_allocation, portfolio_settings)

    results, allocations, decision_log = simulate_portfolio(
        prices=prices,
        regimes=stable_regimes,
        raw_regimes=predictions["regime"],
        trend_signals=trend_signal_frame,
        user_profile=profile,
        portfolio_settings=portfolio_settings,
    )
    starting_capital = (portfolio_settings or {}).get("starting_capital", 10_000.0)
    metrics = evaluate_performance(results, starting_capital=float(starting_capital))
    regime_performance = evaluate_regime_performance(results)
    profile_performance = _profile_performance(
        prices=prices,
        stable_regimes=stable_regimes,
        predictions=predictions,
        trend_signal_frame=trend_signal_frame,
        portfolio_settings=portfolio_settings,
        starting_capital=float(starting_capital),
    )

    decision_records = _decision_records(decision_log)
    ledger_records = _ledger_records(results)
    save_decision_history(decision_records)
    save_portfolio_ledger(ledger_records)
    journal_entry = decision_records[-1]

    return {
        "prices": prices,
        "features": features,
        "predictions": predictions,
        "stable_regimes": stable_regimes,
        "current_regime": current_regime,
        "stable": stable,
        "trend_signals": trend_signals,
        "action": action,
        "allocation": allocation,
        "ideal_allocation": ideal_allocation,
        "implementation_details": implementation_details,
        "results": results,
        "allocations": allocations,
        "decision_log": decision_log,
        "decision_records": decision_records,
        "ledger_records": ledger_records,
        "metrics": metrics,
        "regime_performance": regime_performance,
        "profile_performance": profile_performance,
        "journal_entry": journal_entry,
        "centroids": artifact["centroids"],
        "regime_mapping": artifact["regime_mapping"],
    }


def _profile_performance(
    prices,
    stable_regimes,
    predictions,
    trend_signal_frame,
    portfolio_settings,
    starting_capital: float,
) -> list[dict]:
    """Run the same simulation for representative user profiles."""
    profiles = {
        "Conservative": {
            **DEFAULT_USER_PROFILE,
            "risk_tolerance": "low",
            "experience_level": "beginner",
            "investment_horizon": "short",
        },
        "Balanced": {
            **DEFAULT_USER_PROFILE,
            "risk_tolerance": "medium",
            "experience_level": "beginner",
            "investment_horizon": "long",
        },
        "Aggressive": {
            **DEFAULT_USER_PROFILE,
            "risk_tolerance": "high",
            "experience_level": "intermediate",
            "investment_horizon": "long",
        },
    }
    rows = []
    for label, profile in profiles.items():
        profile_results, _, _ = simulate_portfolio(
            prices=prices,
            regimes=stable_regimes,
            raw_regimes=predictions["regime"],
            trend_signals=trend_signal_frame,
            user_profile=profile,
            portfolio_settings=portfolio_settings,
        )
        metrics = evaluate_performance(profile_results, starting_capital=starting_capital)
        rows.append(
            {
                "profile": label,
                "total_return": metrics["total_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "ending_value": metrics["ending_value"],
            }
        )
    return rows


def _decision_records(decision_log):
    """Convert the decision DataFrame into JSON-serializable API records."""
    records = []
    for market_date, row in decision_log.iterrows():
        records.append(
            {
                "market_date": str(market_date.date()),
                "regime": row["regime"],
                "macro_label": row["regime"],
                "regime_explanation": REGIME_EXPLANATIONS.get(row["regime"], ""),
                "raw_regime": row["raw_regime"],
                "action": row["action"],
                "rebalance_executed": bool(row["rebalance_executed"]),
                "ideal_allocation": row["ideal_allocation"],
                "target_allocation": row["target_allocation"],
                "actual_allocation": row["actual_allocation"],
                "implementation_details": row["implementation_details"],
                "portfolio_value": round(float(row["portfolio_value"]), 2),
            }
        )
    return records


def _ledger_records(results):
    """Convert the portfolio ledger DataFrame into JSON-serializable records."""
    records = []
    allocation_columns = [
        column
        for column in results.columns
        if column.startswith("ideal_") or column.startswith("target_") or column.startswith("actual_")
    ]
    for market_date, row in results.iterrows():
        records.append(
            {
                "market_date": str(market_date.date()),
                "regime": row["regime"],
                "macro_label": row["regime"],
                "action": row["action"],
                "rebalance_executed": bool(row["rebalance_executed"]),
                "portfolio_value": round(float(row["portfolio_value"]), 2),
                "benchmark_value": round(float(row["benchmark_value"]), 2),
                "daily_return": float(row["daily_return"]),
                "drawdown": float(row["drawdown"]),
                **{column: float(row[column]) for column in allocation_columns},
            }
        )
    return records
