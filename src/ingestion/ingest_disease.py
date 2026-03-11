import os
import csv
import logging
from datetime import datetime
import psycopg2 # importing postgresql library -> connect and query db
from psycopg2.extras import execute_values # faster for bulk inserts

logging.basicConfig(level=logging.INFO) # records msg
logger = logging.getLogger(__name__)

DIST_MAPPING = {
    "Chottogram": "Chittagong",
    "Sylet": "Sylhet",
    "Dhaka": "Dhaka",
    "Barishal": "Barisal",
    "Rajshahi": "Rajshahi",
    "Khulna": "Khulna",
    "Rangpur": "Rangpur",
    "Mymensingh": "Mymensingh",
}

DIST_DIVISION = {
   "Dhaka": "Dhaka",
    "Chittagong": "Chittagong",
    "Khulna": "Khulna",
    "Rangpur": "Rangpur",
    "Barisal": "Barisal",
    "Sylhet": "Sylhet",
    "Mymensingh": "Mymensingh",
    "Rajshahi": "Rajshahi", 
}

def get_conn():
    return psycopg2.connect(
        host = os.getenv("POSTGRES_HOST", "postgres"),
        port = os.getenv("POSTGRES_PORT", 5432),
        dbname = os.getenv("POSTGRES_DB", "dengue_db"),
        user = os.getenv("POSTGRES_USER", "dengue_admin"),
        password = os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024")
)

def ensure_dist_exists(conn):
    # insert dist if geo.dist not there
    cursor = conn.cursor()
    for dist_name, div_name in DIST_DIVISION.items():
        cursor.execute((dist_name, div_name))
    conn.commit()
    cursor.close()
    logger.info("Districts ensured in the database")

def get_dist_id_map(conn):
    # returns {dist_name: dist_id}
    cursor = conn.cursor()
    cursor.execute(
        "SELECT dist_id, dist_name FROM geo.districts"
    )
    rows = cursor.fetchall()
    cursor.close()
    return {name: did for did, name in rows}

def parse_date(date_str):
    # dd/mm/yyyy
    date = datetime.strptime(date_str(), "%d/%m/%Y")
    week = date.isocalendar()[1]
    return date.year, date.month, week

def load_csv(file_path):
    records = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dist = row["District"].strip()
            dist = DIST_MAPPING.get(raw_dist, raw_dist)
            date_str = row["Month"].strip()
            patients = int(row["Patients"].strip())
            year, month, week = parse_date(date_str)
            records.append((dist, year, month, week, patients))

    logger.info(f"Loaded {len(records)} records from csv file {file_path}")
    return records