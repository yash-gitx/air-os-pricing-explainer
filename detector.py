import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def detect_pricing_anomalies(df: pd.DataFrame, contamination: float = 0.25) -> pd.DataFrame:
    """
    Runs hybrid anomaly detection (Isolation Forest + Guardrail Rules)
    on the warehouse pricing snapshot.
    """
    data = df.copy()

    # Features for unsupervised anomaly detection
    feature_cols = [
        "dod_price_change_pct",
        "gross_margin_pct",
        "days_in_inventory",
        "local_supply_index",
        "test_drives_7d"
    ]
    X = data[feature_cols].fillna(0)

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X)

    # Calculate statistical Z-score on Day-over-Day price change
    mean_delta = data["dod_price_change_pct"].mean()
    std_delta = data["dod_price_change_pct"].std() or 1.0
    data["price_delta_zscore"] = ((data["dod_price_change_pct"] - mean_delta) / std_delta).round(2)

    data["anomaly_score"] = np.round(-scores, 3)
    
    # Hybrid Flag: Must be flagged by Isolation Forest OR breach hard margin/drop guardrails
    data["is_anomaly"] = (
        ((preds == -1) & (data["dod_price_change_pct"] < -1.5)) |
        (data["dod_price_change_pct"] <= -5.0) |
        (data["gross_margin_pct"] < 0.0)
    )

    def classify_severity(row):
        if not row["is_anomaly"]:
            return "Normal"
        if row["gross_margin_pct"] < 0 or row["dod_price_change_pct"] < -8.0:
            return "P0: Margin Breach"
        return "P1: Steep Markdown"

    data["severity"] = data.apply(classify_severity, axis=1)
    return data


def plot_car_trajectory(history_df: pd.DataFrame, car_meta: dict) -> plt.Figure:
    """Generates a two-panel Matplotlib figure showing Price vs Acquisition Cost & Demand Velocity."""
    dates = pd.to_datetime(history_df["snapshot_date"])
    prices = history_df["listed_price_lakhs"]
    acq_cost = history_df["acquisition_cost_lakhs"].iloc[0]
    views = history_df["page_views_7d"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor("#0E1117")
    
    for ax in (ax1, ax2):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363D")

    # Top Plot: Listed Price vs Acquisition Cost Floor
    ax1.plot(dates, prices, color="#58A6FF", linewidth=2.2, marker="o", markersize=4, label="Listed Price (₹ Lakhs)")
    ax1.axhline(acq_cost, color="#F85149", linestyle="--", linewidth=1.5, label=f"Acquisition Cost Floor (₹{acq_cost}L)")
    
    # Highlight the latest point
    ax1.scatter(dates.iloc[-1], prices.iloc[-1], color="#FF7B72", s=90, zorder=5, label="Latest Snapshot")
    ax1.set_title(
        f"{car_meta['car_id']} · {car_meta['make_model']} ({car_meta['city_hub']}) — 30-Day Price & Demand Telemetry",
        color="#F0F6FC", fontsize=10, fontweight="bold", pad=10
    )
    ax1.set_ylabel("Price (₹ Lakhs)", color="#C9D1D9", fontsize=9)
    ax1.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.25, color="#8B949E")

    # Bottom Plot: 7-Day Page Views
    ax2.bar(dates, views, color="#3FB950", alpha=0.7, width=0.6, label="7D Rolling Page Views")
    ax2.set_ylabel("Views (7D)", color="#C9D1D9", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.grid(True, linestyle=":", alpha=0.25, color="#8B949E")

    plt.tight_layout()
    return fig