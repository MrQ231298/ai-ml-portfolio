"""
Streamlit demo: compare weekly tyre-demand forecasts on SYNTHETIC data.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "tyre_demand_forecast.joblib"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def main() -> None:
    st.set_page_config(page_title="Tyre demand forecast (demo)", page_icon="🛞", layout="wide")
    st.title("Weekly tyre demand forecast")
    st.caption("Portfolio demo — trained on SYNTHETIC weekly unit sales only.")

    st.info(
        "Every point in this series is invented example data, not a real "
        "depot extract and not official NCT statistics. The seasonal shape "
        "is *inspired by* winter-tyre changeover and a spring NCT-related "
        "uptick, but the numbers are made up."
    )

    bundle = load_bundle()
    if bundle is None:
        st.error("Saved model not found. Run `python src/generate_synthetic_series.py` and `python src/train.py`, then reload.")
        st.stop()

    full = bundle["full_series"].copy()
    full["week_start"] = pd.to_datetime(full["week_start"])
    hold = pd.read_csv(
        Path(__file__).resolve().parent / "reports" / "holdout_2025_forecasts.csv",
        parse_dates=["week_start"],
    )

    m = bundle["metrics"]
    st.subheader("2025 hold-out (forecast from the end of 2024)")
    context = full[full["week_start"] >= "2023-01-01"]
    plot_df = pd.DataFrame({"actual": context.set_index("week_start")["units_sold"]})
    hold_i = hold.set_index("week_start")
    plot_df = plot_df.join(hold_i[["seasonal_naive", "ets", "gradient_boosting"]], how="left")
    plot_df = plot_df.rename(
        columns={
            "seasonal_naive": "seasonal naive",
            "ets": "ETS (Holt–Winters)",
            "gradient_boosting": "gradient boosting",
        }
    )
    st.line_chart(plot_df)

    scores = m["holdout_2025"]
    table = pd.DataFrame(
        {
            "model": ["Seasonal naive", "ETS (Holt–Winters)", "Gradient boosting (lags)"],
            "holdout_MAE": [scores["seasonal_naive"]["mae"], scores["ets"]["mae"], scores["gradient_boosting"]["mae"]],
            "holdout_MAPE_pct": [scores["seasonal_naive"]["mape"], scores["ets"]["mape"], scores["gradient_boosting"]["mape"]],
            "backtest_MAE": [
                m["backtest"]["seasonal_naive"]["mae"],
                m["backtest"]["ets"]["mae"],
                m["backtest"]["gradient_boosting"]["mae"],
            ],
            "backtest_MAPE_pct": [
                m["backtest"]["seasonal_naive"]["mape"],
                m["backtest"]["ets"]["mape"],
                m["backtest"]["gradient_boosting"]["mape"],
            ],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    winner = min(m["backtest"], key=lambda n: m["backtest"][n]["mae"])
    st.caption(
        f"Back-test: expanding window, {m['backtest'][winner]['horizon_weeks']}-week "
        f"horizon, step {m['backtest'][winner]['step_weeks']} weeks. "
        f"Lowest mean back-test MAE: **{winner.replace('_', ' ')}** "
        f"({m['backtest'][winner]['mae']} units)."
    )
    st.divider()
    st.caption("Data: SYNTHETIC_weekly_tyre_sales.csv (SYNTHETIC_EXAMPLE_DATA).")


if __name__ == "__main__":
    main()
