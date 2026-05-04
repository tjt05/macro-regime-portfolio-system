from __future__ import annotations

"""FastAPI layer used by the Streamlit frontend.

The API keeps the frontend thin: Streamlit collects user inputs and displays
charts, while this backend runs the data/model/simulation pipeline and persists
decision logs. All request and response payloads are plain JSON for easy testing.
"""

from datetime import date
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.assets import ASSET_ORDER
from backend.journal.logger import (
    append_live_journal_entry,
    load_journal,
    load_live_journal,
    load_portfolio_ledger,
)
from backend.macro.regime_interpreter import REGIME_EXPLANATIONS
from backend.pipeline import run_pipeline
from backend.strategy.personalization import DEFAULT_USER_PROFILE


class UserProfile(BaseModel):
    """Financial profile fields used to personalize allocations."""

    income: float = Field(default=1500, ge=0)
    expenses: float = Field(default=900, ge=0)
    savings: float = Field(default=5000, ge=0)
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    experience_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    investment_horizon: Literal["short", "medium", "long"] = "long"


class PortfolioSettings(BaseModel):
    """Simulation and implementation constraints chosen in the sidebar."""

    start_date: str | None = None
    starting_capital: float = Field(default=10_000, gt=0)
    rebalance_policy: Literal["daily", "weekly", "monthly", "threshold", "regime_change"] = "monthly"
    min_rebalance_threshold: float = Field(default=0.05, ge=0, le=1)
    cooldown_days: int = Field(default=5, ge=0, le=252)
    max_assets: int = Field(default=4, ge=1, le=len(ASSET_ORDER))
    min_position_weight: float = Field(default=0.08, ge=0, le=1)
    avoid_assets: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    """Payload for one complete PRAIDS pipeline run."""

    user_profile: UserProfile = Field(default_factory=UserProfile)
    portfolio_settings: PortfolioSettings = Field(default_factory=PortfolioSettings)
    force_train: bool = False


class TradeIntent(BaseModel):
    """One structured real-life trade note for the live journal."""

    asset: str
    trade_type: Literal["buy", "sell", "hold"]
    amount_type: Literal["dollars", "shares", "target_weight"]
    amount: float = Field(default=0, ge=0)


class LiveJournalEntry(BaseModel):
    """User-entered record of what they actually decided to do."""

    signal_date: str
    market_data_date: str
    intended_execution_date: str
    recommended_regime: str
    recommended_action: str
    recommended_allocation: dict[str, float]
    user_action: Literal["follow", "partial_follow", "ignore", "defer", "custom"]
    actual_execution_date: str | None = None
    actual_trades: list[TradeIntent] = Field(default_factory=list)
    notes: str = ""


app = FastAPI(
    title="PRAIDS API",
    description="Backend API for the Personalized Regime-Aware Investment Decision System.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Container healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/profile/default")
def default_profile() -> dict:
    """Return the default profile used to prefill the frontend sidebar."""
    return DEFAULT_USER_PROFILE


@app.get("/assets")
def assets() -> list[str]:
    """Return the canonical tradable/display asset order."""
    return ASSET_ORDER


@app.get("/journal")
def journal() -> list[dict]:
    """Return the latest saved simulated decision history."""
    return load_journal()


@app.get("/ledger")
def ledger() -> list[dict]:
    """Return the latest saved simulated portfolio ledger."""
    return load_portfolio_ledger()


@app.get("/live-journal")
def live_journal() -> list[dict]:
    """Return user-entered live journal records."""
    return load_live_journal()


@app.post("/live-journal")
def create_live_journal_entry(entry: LiveJournalEntry) -> dict:
    """Append a structured real-life decision record."""
    return append_live_journal_entry(entry.model_dump())


@app.post("/run")
def run(request: RunRequest) -> dict:
    """Execute the complete PRAIDS pipeline and serialize it for Streamlit."""
    output = run_pipeline(
        user_profile=request.user_profile.model_dump(),
        portfolio_settings=request.portfolio_settings.model_dump(),
        force_train=request.force_train,
    )
    return serialize_pipeline_output(output)


def serialize_pipeline_output(output: dict) -> dict:
    """Convert pandas-heavy pipeline output into compact JSON-safe records."""
    results = output["results"].tail(750)
    predictions = output["predictions"].tail(120)
    stable_regimes = output["stable_regimes"].tail(120)

    recent_regimes = pd.concat([predictions, stable_regimes], axis=1)

    latest_market_date = output["journal_entry"]["market_date"]
    current_recommendation = {
        "signal_date": date.today().isoformat(),
        "market_data_date": latest_market_date,
        "intended_execution_date": _next_business_day(latest_market_date),
        "regime": output["current_regime"],
        "regime_explanation": REGIME_EXPLANATIONS.get(output["current_regime"], ""),
        "action": output["action"],
        "ideal_allocation": _clean_mapping(output["ideal_allocation"]),
        "target_allocation": _clean_mapping(output["allocation"]),
        "actual_allocation": _clean_mapping(_latest_actual_allocation(output["results"])),
        "implementation_details": output["implementation_details"],
    }

    return {
        "current_regime": output["current_regime"],
        "regime_explanation": REGIME_EXPLANATIONS.get(output["current_regime"], ""),
        "stable": bool(output["stable"]),
        "trend_signals": {key: bool(value) for key, value in output["trend_signals"].items()},
        "action": output["action"],
        "ideal_allocation": _clean_mapping(output["ideal_allocation"]),
        "allocation": _clean_mapping(output["allocation"]),
        "actual_allocation": _clean_mapping(_latest_actual_allocation(output["results"])),
        "implementation_details": output["implementation_details"],
        "metrics": _clean_mapping(output["metrics"]),
        "regime_performance": [_clean_mapping(row) for row in output["regime_performance"]],
        "profile_performance": [_clean_mapping(row) for row in output["profile_performance"]],
        "journal_entry": output["journal_entry"],
        "regime_mapping": {str(cluster): regime for cluster, regime in output["regime_mapping"].items()},
        "portfolio_curve": _frame_to_records(
            results[["portfolio_value", "benchmark_value", "drawdown", "benchmark_drawdown"]]
        ),
        "recent_regimes": _frame_to_records(recent_regimes),
        "decision_log": output["decision_records"],
        "portfolio_ledger": output["ledger_records"],
        "current_recommendation": current_recommendation,
        "live_journal": load_live_journal(),
        "assets": ASSET_ORDER,
        "journal": output["decision_records"],
    }


def _frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a date-indexed DataFrame into JSON records with string dates."""
    prepared = frame.reset_index()
    first_column = prepared.columns[0]
    prepared = prepared.rename(columns={first_column: "date"})
    prepared["date"] = prepared["date"].astype(str)
    return [
        {key: _clean_scalar(value) for key, value in row.items()}
        for row in prepared.to_dict(orient="records")
    ]


def _clean_mapping(mapping: dict) -> dict:
    """Convert mapping keys/values into JSON-safe Python scalars."""
    return {str(key): _clean_scalar(value) for key, value in mapping.items()}


def _clean_scalar(value):
    """Convert pandas/numpy scalar values into JSON-serializable values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _latest_actual_allocation(results: pd.DataFrame) -> dict[str, float]:
    """Extract the latest drifted allocation from simulation results."""
    latest = results.iloc[-1]
    return {
        column.replace("actual_", ""): float(latest[column])
        for column in results.columns
        if column.startswith("actual_")
    }


def _next_business_day(market_date: str) -> str:
    """Return the next business day after the latest available market date."""
    start = pd.Timestamp(market_date) + pd.offsets.BDay(1)
    return start.date().isoformat()
