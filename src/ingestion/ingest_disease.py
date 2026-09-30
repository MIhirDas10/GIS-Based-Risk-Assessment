import os
import csv
import logging
from datetime import datetime
import psycopg2  # importing postgresql library -> connect and query db
from psycopg2.extras import execute_values  # faster for bulk inserts

logging.basicConfig(level=logging.INFO)  # records msg
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
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )


def ensure_dist_exists(conn):
    # insert dist if geo.dist not there
    cursor = conn.cursor()
    for dist_name, div_name in DIST_DIVISION.items():
        cursor.execute(
            """
            INSERT INTO geo.districts (district_name, division_name)
            SELECT %s, %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM geo.districts
                WHERE district_name = %s
            )
            """,
            (dist_name, div_name, dist_name),
        )
    conn.commit()
    cursor.close()
    logger.info("Districts ensured in the database")

def get_dist_id_map(conn):
    # returns {dist_name: dist_id}
    cursor = conn.cursor()
    cursor.execute(
        "SELECT district_id, district_name FROM geo.districts"
    )
    rows = cursor.fetchall()
    cursor.close()
    return {name: did for did, name in rows}

def parse_date(date_str):
    # supports dd/mm/yy and dd/mm/yyyy
    cleaned = date_str.strip()
    date_formats = ("%d/%m/%y", "%d/%m/%Y")
    date = None

    for date_format in date_formats:
        try:
            date = datetime.strptime(cleaned, date_format)
            break
        except ValueError:
            continue

    if date is None:
        raise ValueError(
            f"Unsupported date format '{date_str}'. Expected dd/mm/yy or dd/mm/yyyy"
        )

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


def prepare_case_rows(records, dist_id_map):
    case_rows = []
    missing_districts = set()

    for dist_name, year, month, week, patients in records:
        dist_id = dist_id_map.get(dist_name)
        if not dist_id:
            missing_districts.add(dist_name)
            continue

        case_rows.append((dist_id, year, month, week, patients, "kaggle"))

    if missing_districts:
        logger.warning(
            "Skipped records for districts not found in geo.districts: %s",
            ", ".join(sorted(missing_districts)),
        )

    return case_rows


def insert_cases(conn, case_rows):
    if not case_rows:
        logger.info("No dengue case rows to insert")
        return 0

    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO disease.dengue_cases (
            district_id,
            year,
            month,
            week,
            dengue_cases,
            data_source
        )
        SELECT
            v.district_id,
            v.year,
            v.month,
            v.week,
            v.dengue_cases,
            v.data_source
        FROM (VALUES %s) AS v(
            district_id,
            year,
            month,
            week,
            dengue_cases,
            data_source
        )
        WHERE NOT EXISTS (
            SELECT 1
            FROM disease.dengue_cases d
            WHERE d.district_id = v.district_id
              AND d.year = v.year
              AND d.month = v.month
              AND d.week = v.week
              AND d.dengue_cases = v.dengue_cases
        )
    """

    execute_values(
        cursor,
        insert_sql,
        case_rows,
        template="(%s, %s, %s, %s, %s, %s)",
    )
    inserted = cursor.rowcount
    conn.commit()
    cursor.close()
    logger.info("Inserted %s new rows into disease.dengue_cases", inserted)
    return inserted


def run():
    csv_path = os.getenv("DENGUE_CSV_PATH", "/opt/airflow/data/raw/dengue_cases.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    conn = get_conn()
    try:
        ensure_dist_exists(conn)
        dist_id_map = get_dist_id_map(conn)
        records = load_csv(csv_path)
        case_rows = prepare_case_rows(records, dist_id_map)
        inserted = insert_cases(conn, case_rows)

        logger.info(
            "Ingestion completed. csv_rows=%s mapped_rows=%s inserted_rows=%s",
            len(records),
            len(case_rows),
            inserted,
        )
        return inserted
    finally:
        conn.close()


if __name__ == "__main__":
    run()
