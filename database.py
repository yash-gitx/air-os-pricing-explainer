import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DB_PATH = "spinny_warehouse.db"

def init_and_seed_db(force_reseed: bool = False):
    """Initializes the SQLite warehouse and seeds 30 days of realistic inventory telemetry."""
    if os.path.exists(DB_PATH) and not force_reseed:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS daily_pricing_logs;
        DROP TABLE IF EXISTS inventory_master;

        CREATE TABLE inventory_master (
            car_id TEXT PRIMARY KEY,
            make_model TEXT,
            variant TEXT,
            mfg_year INTEGER,
            city_hub TEXT,
            segment TEXT,
            acquisition_cost_lakhs REAL,
            inspection_score INTEGER
        );

        CREATE TABLE daily_pricing_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_id TEXT,
            snapshot_date DATE,
            listed_price_lakhs REAL,
            days_in_inventory INTEGER,
            page_views_7d INTEGER,
            test_drives_7d INTEGER,
            local_supply_index REAL,
            pricing_rule_triggered TEXT,
            FOREIGN KEY (car_id) REFERENCES inventory_master(car_id)
        );
    """)

    np.random.seed(42)

    cars = [
        ("SPN-GGN-101", "Hyundai Creta", "SX 1.5 Petrol", 2021, "Gurugram", "SUV", 10.80, 92),
        ("SPN-GGN-102", "Maruti Baleno", "Alpha CVT", 2022, "Gurugram", "Hatchback", 7.10, 89),
        ("SPN-GGN-103", "Tata Nexon", "XZ+ Diesel", 2021, "Gurugram", "SUV", 8.90, 85),
        ("SPN-GGN-104", "Mahindra XUV700", "AX7 Smart", 2022, "Gurugram", "SUV", 17.50, 94),
        ("SPN-BLR-201", "Kia Seltos", "GTX+ DCT", 2020, "Bengaluru", "SUV", 12.20, 88),
        ("SPN-BLR-202", "Hyundai i20", "Asta Opt", 2021, "Bengaluru", "Hatchback", 7.40, 91),
        ("SPN-BLR-203", "Honda City", "ZX CVT", 2021, "Bengaluru", "Sedan", 10.50, 90),
        ("SPN-BOM-301", "Maruti Swift", "ZXI AMT", 2022, "Mumbai", "Hatchback", 6.40, 87),
        ("SPN-BOM-302", "Volkswagen Taigun", "Topline AT", 2022, "Mumbai", "SUV", 13.10, 93),
        ("SPN-GGN-105", "Honda Amaze", "VX CVT", 2021, "Gurugram", "Sedan", 6.80, 86),
        ("SPN-BLR-204", "Tata Punch", "Creative AMT", 2023, "Bengaluru", "SUV", 7.60, 95),
        ("SPN-BOM-303", "Hyundai Venue", "SX Turbo", 2021, "Mumbai", "SUV", 8.70, 84),
    ]

    cursor.executemany(
        "INSERT INTO inventory_master VALUES (?, ?, ?, ?, ?, ?, ?, ?)", cars
    )

    base_date = datetime.today().date() - timedelta(days=29)
    logs = []

    for car in cars:
        car_id, model, _, _, city, segment, acq_cost, _ = car
        # Initial target margin ~12-15%
        current_price = round(acq_cost * np.random.uniform(1.12, 1.16), 2)
        start_age = np.random.randint(10, 45)

        for day_offset in range(30):
            snap_date = base_date + timedelta(days=day_offset)
            age = start_age + day_offset
            views = max(15, int(np.random.normal(140 - age * 0.8, 20)))
            test_drives = max(0, int(views / np.random.uniform(35, 60)))
            supply_idx = round(np.random.normal(1.0, 0.08), 2)
            rule = "Standard_Hold"

            # Normal gradual markdown for aging inventory
            if day_offset > 0 and age % 15 == 0:
                current_price = round(current_price * 0.992, 2)
                rule = "Routine_Age_Stepdown"

            # Inject realistic anomalies on the final day (day_offset == 29)
            if day_offset == 29:
                if car_id == "SPN-GGN-101":
                    # ANOMALY 1: Severe price drop due to aging >60d + zero test drives
                    current_price = round(current_price * 0.915, 2)
                    views = 32
                    test_drives = 0
                    age = 68
                    rule = "Auto_Liquidation_60D_Trigger"
                elif car_id == "SPN-BLR-201":
                    # ANOMALY 2: Aggressive markdown triggered by local competitor supply surge
                    current_price = round(current_price * 0.932, 2)
                    supply_idx = 1.48
                    rule = "Competitor_Index_Reprice"
                elif car_id == "SPN-BOM-302":
                    # ANOMALY 3: Accidental margin-negative feed glitch despite high demand
                    current_price = round(acq_cost * 0.96, 2)
                    views = 210
                    test_drives = 6
                    rule = "External_Valuation_Sync_Override"

            logs.append((
                car_id,
                snap_date.isoformat(),
                current_price,
                age,
                views,
                test_drives,
                supply_idx,
                rule
            ))

    cursor.executemany("""
        INSERT INTO daily_pricing_logs (
            car_id, snapshot_date, listed_price_lakhs, days_in_inventory,
            page_views_7d, test_drives_7d, local_supply_index, pricing_rule_triggered
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, logs)

    conn.commit()
    conn.close()


SNAPSHOT_SQL_QUERY = """
WITH RankedLogs AS (
    SELECT 
        l.car_id,
        m.make_model,
        m.variant,
        m.mfg_year,
        m.city_hub,
        m.segment,
        m.acquisition_cost_lakhs,
        m.inspection_score,
        l.snapshot_date,
        l.listed_price_lakhs,
        l.days_in_inventory,
        l.page_views_7d,
        l.test_drives_7d,
        l.local_supply_index,
        l.pricing_rule_triggered,
        LAG(l.listed_price_lakhs, 1) OVER (
            PARTITION BY l.car_id ORDER BY l.snapshot_date
        ) AS prev_day_price,
        AVG(l.listed_price_lakhs) OVER (
            PARTITION BY l.car_id ORDER BY l.snapshot_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_7d_avg_price,
        ROW_NUMBER() OVER (
            PARTITION BY l.car_id ORDER BY l.snapshot_date DESC
        ) AS rn
    FROM daily_pricing_logs l
    JOIN inventory_master m ON l.car_id = m.car_id
)
SELECT 
    car_id,
    make_model,
    variant,
    mfg_year,
    city_hub,
    segment,
    acquisition_cost_lakhs,
    inspection_score,
    snapshot_date,
    listed_price_lakhs,
    ROUND(prev_day_price, 2) AS prev_day_price,
    ROUND(rolling_7d_avg_price, 2) AS rolling_7d_avg_price,
    ROUND(((listed_price_lakhs - prev_day_price) / prev_day_price) * 100, 2) AS dod_price_change_pct,
    ROUND(((listed_price_lakhs - acquisition_cost_lakhs) / acquisition_cost_lakhs) * 100, 2) AS gross_margin_pct,
    days_in_inventory,
    page_views_7d,
    test_drives_7d,
    local_supply_index,
    pricing_rule_triggered
FROM RankedLogs
WHERE rn = 1
ORDER BY dod_price_change_pct ASC;
"""

def fetch_latest_warehouse_snapshot() -> pd.DataFrame:
    """Executes the window-function SQL query to retrieve the latest daily pricing metrics."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(SNAPSHOT_SQL_QUERY, conn)
    conn.close()
    return df

def fetch_car_history(car_id: str) -> pd.DataFrame:
    """Fetches the 30-day time series for a specific car."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT l.snapshot_date, l.listed_price_lakhs, l.page_views_7d, 
               l.test_drives_7d, l.local_supply_index, m.acquisition_cost_lakhs
        FROM daily_pricing_logs l
        JOIN inventory_master m ON l.car_id = m.car_id
        WHERE l.car_id = ?
        ORDER BY l.snapshot_date ASC
    """
    df = pd.read_sql_query(query, conn, params=(car_id,))
    conn.close()
    return df