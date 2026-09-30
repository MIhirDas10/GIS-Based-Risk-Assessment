"""
scripts/seed_synthetic_era5.py

Seeds weather.era5_district_weekly with plausible synthetic Bangladesh
weather (2019-2023) for ALL districts so the rest of the pipeline can be
verified end-to-end while the real CDS ingest runs overnight.

Climate model (per ISO week of year):
    temp:    sine wave, 21-32 °C, peak in July (~week 28)
    rainfall: monsoon June-Sep (weeks ~22-39), peak ~250-450mm/week
    humidity: 65-92%, peak in monsoon

Adds small per-district noise so districts aren't identical. NOT a replacement
for real ERA5 — purely a smoke-test fixture. Delete these rows and re-run
ingest_era5 when real data is available.
"""
import math
import os
import random

import psycopg2
from psycopg2.extras import execute_values

random.seed(42)

YEARS = list(range(2019, 2024))
WEEKS = list(range(1, 53))


def synth(district_id: int, year: int, week: int) -> tuple[float, float, float, float]:
    # Seasonal phase: peak around week 28 (July)
    phase = 2 * math.pi * (week - 28) / 52
    base_temp = 26.5 + 5.5 * math.cos(phase)  # ~21..32 °C
    # Small per-district offset (~±1.5°C)
    d_offset = (district_id * 37 % 30 - 15) / 10.0
    temp_mean = base_temp + d_offset + random.uniform(-0.8, 0.8)
    temp_max = temp_mean + random.uniform(4, 8)

    # Monsoon rainfall: bell-shaped over weeks 22..40, peak ~400mm
    if 22 <= week <= 40:
        monsoon = math.exp(-((week - 31) ** 2) / 28) * 380
        rainfall = max(0, monsoon + random.uniform(-60, 80))
    else:
        rainfall = max(0, random.uniform(0, 25))

    # Humidity: high in monsoon, dry in winter
    if 22 <= week <= 40:
        hum = 82 + random.uniform(-4, 8)
    elif week <= 8 or week >= 48:
        hum = 70 + random.uniform(-5, 5)
    else:
        hum = 75 + random.uniform(-5, 5)
    hum = max(40, min(96, hum))

    return round(temp_mean, 2), round(temp_max, 2), round(rainfall, 2), round(hum, 2)


def main():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT district_id FROM geo.districts ORDER BY district_id;")
            district_ids = [r[0] for r in cur.fetchall()]
            print(f"Districts: {len(district_ids)}")

            rows = []
            for did in district_ids:
                for year in YEARS:
                    for week in WEEKS:
                        t, tmax, rain, hum = synth(did, year, week)
                        rows.append((did, year, week, t, tmax, rain, hum))

            sql = """
                INSERT INTO weather.era5_district_weekly
                    (district_id, year, week, temp_mean_c, temp_max_c, rainfall_mm, humidity_pct)
                VALUES %s
                ON CONFLICT (district_id, year, week) DO UPDATE
                SET temp_mean_c  = EXCLUDED.temp_mean_c,
                    temp_max_c   = EXCLUDED.temp_max_c,
                    rainfall_mm  = EXCLUDED.rainfall_mm,
                    humidity_pct = EXCLUDED.humidity_pct,
                    ingested_at  = NOW();
            """
            execute_values(cur, sql, rows, page_size=2000)
            conn.commit()
            print(f"Upserted {len(rows)} rows into weather.era5_district_weekly")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
