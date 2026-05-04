from __future__ import annotations

"""CLI entry point for running PRAIDS without the Streamlit frontend."""

from backend.pipeline import run_pipeline


if __name__ == "__main__":
    output = run_pipeline()
    print("PRAIDS decision support run complete")
    print(f"Current regime: {output['current_regime']}")
    print(f"Action: {output['action']}")
    print(f"Stable signal: {output['stable']}")
    print(f"Allocation: {output['allocation']}")
    print(f"Metrics: {output['metrics']}")
