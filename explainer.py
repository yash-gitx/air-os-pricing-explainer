import os
from typing import Dict, Any

def _build_deterministic_fallback(row: Dict[str, Any]) -> str:
    """
    Deterministic analytical fallback so the deployed app always works 
    even before the user enters a Gemini API key.
    """
    car_id = row["car_id"]
    model = f"{row['mfg_year']} {row['make_model']} {row['variant']}"
    drop = row["dod_price_change_pct"]
    margin = row["gross_margin_pct"]
    age = row["days_in_inventory"]
    views = row["page_views_7d"]
    tds = row["test_drives_7d"]
    supply = row["local_supply_index"]
    rule = row["pricing_rule_triggered"]

    if margin < 0:
        hypothesis = (
            f"**Unintended Valuation Override (Margin Breach):** `{rule}` repriced the vehicle to "
            f"**₹{row['listed_price_lakhs']}L** ({drop}% DoD), pushing gross margin to **{margin}%** "
            f"(below the ₹{row['acquisition_cost_lakhs']}L acquisition cost). However, top-of-funnel demand is strong "
            f"({views} views and {tds} test drives in the last 7 days), indicating this markdown is likely a pricing feed glitch rather than demand exhaustion."
        )
        action = (
            "1. **Immediate Rollback:** Revert listed price to the 7-day rolling average (₹"
            f"{row['rolling_7d_avg_price']}L) to protect unit economics.\n"
            "2. **Guardrail Patch:** Add a hard constraint on `External_Valuation_Sync_Override` so automated rules cannot price high-velocity inventory (<45 days old, >3 test drives/wk) below acquisition cost."
        )
    elif age >= 60 and tds == 0:
        hypothesis = (
            f"**Aging Inventory Liquidation Trigger:** `{rule}` executed a **{drop}%** markdown after the vehicle crossed "
            f"**{age} days in inventory** in {row['city_hub']} with **0 test drives** over the past 7 days (only {views} views). "
            f"Gross margin compressed to **{margin}%**, but holding cost erosion justifies the step-down."
        )
        action = (
            "1. **Keep Markdown Active:** Retain the ₹"
            f"{row['listed_price_lakhs']}L price point for 72 hours and monitor lead-to-test-drive conversion.\n"
            "2. **Merchandising Boost:** Trigger a 'Price Drop Alert' push notification to users in "
            f"{row['city_hub']} who wishlisted {row['segment']}s between ₹{row['listed_price_lakhs']-1:.1f}L–₹{row['listed_price_lakhs']+1:.1f}L."
        )
    elif supply > 1.3:
        hypothesis = (
            f"**Local Market Supply Surge:** `{rule}` cut the price by **{drop}%** in response to a regional supply spike "
            f"(`local_supply_index = {supply}x` baseline in {row['city_hub']}). While gross margin remains positive at **{margin}%**, "
            f"the vehicle still has healthy inspection quality ({row['inspection_score']}/100)."
        )
        action = (
            "1. **A/B Test Partial Recovery:** Test holding the price 2.5% higher for cars with Inspection Scores >85 instead of matching bottom-quartile market listings.\n"
            "2. **Hub Rebalancing:** Evaluate transferring stock if local SUV supply stays >1.4x for 14+ days."
        )
    else:
        hypothesis = (
            f"**Statistical Price Deviation:** Listed price shifted by **{drop}%** DoD (Z-score: `{row['price_delta_zscore']}`) "
            f"under rule `{rule}`, leaving current gross margin at **{margin}%**."
        )
        action = "1. Audit pricing log trigger conditions with the category pricing manager."

    return f"""### 🔍 Root-Cause Diagnosis: `{car_id}` ({model})
* **Primary Hypothesis:** {hypothesis}
* **Unit Economics Impact:** Current Listed Price: **₹{row['listed_price_lakhs']}L** | Acquisition Cost: **₹{row['acquisition_cost_lakhs']}L** | Net Margin: **{margin}%**
* **Recommended PM / Pricing Action:**
{action}
"""


def generate_anomaly_explanation(row: Dict[str, Any], api_key: str = "") -> str:
    """
    Calls Gemini API to synthesize SQL telemetry into an actionable PM root-cause memo.
    Falls back cleanly to deterministic logic if no API key is provided.
    """
    effective_key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "")
    if not effective_key:
        return _build_deterministic_fallback(row)

    try:
        from google import genai
        client = genai.Client(api_key=effective_key)

        prompt = f"""
You are AIR OS, Spinny's internal Applied AI operating system used by Product Managers and Pricing Analysts.
Analyze the following anomalous daily price movement detected in the inventory data warehouse and write a sharp, data-backed Root-Cause Memo.

### Warehouse Telemetry Snapshot:
- Car ID: {row['car_id']} ({row['mfg_year']} {row['make_model']} {row['variant']})
- Hub / Segment: {row['city_hub']} / {row['segment']}
- Acquisition Cost: ₹{row['acquisition_cost_lakhs']} Lakhs
- Previous Day Price: ₹{row['prev_day_price']} Lakhs -> Current Listed Price: ₹{row['listed_price_lakhs']} Lakhs
- Day-over-Day Change: {row['dod_price_change_pct']}% (Z-Score: {row['price_delta_zscore']})
- Current Gross Margin: {row['gross_margin_pct']}%
- Inventory Age: {row['days_in_inventory']} days
- Demand Velocity (Last 7D): {row['page_views_7d']} page views, {row['test_drives_7d']} test drives
- Local Competitor Supply Index: {row['local_supply_index']} (1.0 = normal baseline)
- Automated Pricing Rule Triggered: {row['pricing_rule_triggered']}
- Inspection Score: {row['inspection_score']}/100

### Output Format (Strict Markdown, concise, no fluff):
1. **Why the Number Moved (Root Cause):** Distinguish whether this drop is a valid liquidation markdown (aging + low demand), a market supply reaction, or a bad pricing rule / feed bug (e.g., cutting price despite high test drives or breaching acquisition cost).
2. **Margin & Funnel Impact:** Quantify the rupee/percentage margin risk.
3. **Recommended Action (Next 24 Hours):** Give 2 concrete actions for the Pricing PM (e.g., price rollback, guardrail rule update, or CRM push notification).
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        fallback = _build_deterministic_fallback(row)
        return f"*(Note: Live Gemini API call fell back to local AIR OS engine: `{str(e)}`)*\n\n" + fallback