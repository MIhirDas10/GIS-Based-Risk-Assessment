import os
import logging
import geopandas as gpd
import psycopg2
from psycopg2.extras import execute_values
from shapely.geometry import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map HUMDATA district names → our standardized names
DISTRICT_MAPPING = {
    "Chittagong": "Chittagong",
    "Comilla": "Comilla",
    "Cox's Bazar": "Cox's Bazar",
    "Feni": "Feni",
    "Khagrachhari": "Khagrachhari",
    "Lakshmipur": "Lakshmipur",
    "Noakhali": "Noakhali",
    "Rangamati": "Rangamati",
    "Brahmanbaria": "Brahmanbaria",
    "Chandpur": "Chandpur",
    "Dhaka": "Dhaka",
    "Faridpur": "Faridpur",
    "Gazipur": "Gazipur",
    "Gopalganj": "Gopalganj",
    "Kishoreganj": "Kishoreganj",
    "Madaripur": "Madaripur",
    "Manikganj": "Manikganj",
    "Munshiganj": "Munshiganj",
    "Narayanganj": "Narayanganj",
    "Narsingdi": "Narsingdi",
    "Rajbari": "Rajbari",
    "Shariatpur": "Shariatpur",
    "Tangail": "Tangail",
    "Bagerhat": "Bagerhat",
    "Chuadanga": "Chuadanga",
    "Jessore": "Jessore",
    "Jhenaidah": "Jhenaidah",
    "Khulna": "Khulna",
    "Kushtia": "Kushtia",
    "Magura": "Magura",
    "Meherpur": "Meherpur",
    "Narail": "Narail",
    "Satkhira": "Satkhira",
    "Barguna": "Barguna",
    "Barisal": "Barisal",
    "Bhola": "Bhola",
    "Jhalokati": "Jhalokati",
    "Patuakhali": "Patuakhali",
    "Pirojpur": "Pirojpur",
    "Bandarban": "Bandarban",
    "Habiganj": "Habiganj",
    "Moulvibazar": "Moulvibazar",
    "Sunamganj": "Sunamganj",
    "Sylhet": "Sylhet",
    "Bogra": "Bogra",
    "Joypurhat": "Joypurhat",
    "Naogaon": "Naogaon",
    "Natore": "Natore",
    "Nawabganj": "Nawabganj",
    "Pabna": "Pabna",
    "Rajshahi": "Rajshahi",
    "Sirajganj": "Sirajganj",
    "Dinajpur": "Dinajpur",
    "Gaibandha": "Gaibandha",
    "Kurigram": "Kurigram",
    "Lalmonirhat": "Lalmonirhat",
    "Nilphamari": "Nilphamari",
    "Panchagarh": "Panchagarh",
    "Rangpur": "Rangpur",
    "Thakurgaon": "Thakurgaon",
    "Jamalpur": "Jamalpur",
    "Mymensingh": "Mymensingh",
    "Netrakona": "Netrakona",
    "Sherpur": "Sherpur",
}

# Division mapping by district
DISTRICT_DIVISION_MAP = {
    "Chittagong": "Chittagong", "Comilla": "Chittagong",
    "Cox's Bazar": "Chittagong", "Feni": "Chittagong",
    "Khagrachhari": "Chittagong", "Lakshmipur": "Chittagong",
    "Noakhali": "Chittagong", "Rangamati": "Chittagong",
    "Brahmanbaria": "Chittagong", "Chandpur": "Chittagong",
    "Bandarban": "Chittagong",
    "Dhaka": "Dhaka", "Faridpur": "Dhaka", "Gazipur": "Dhaka",
    "Gopalganj": "Dhaka", "Kishoreganj": "Dhaka", "Madaripur": "Dhaka",
    "Manikganj": "Dhaka", "Munshiganj": "Dhaka", "Narayanganj": "Dhaka",
    "Narsingdi": "Dhaka", "Rajbari": "Dhaka", "Shariatpur": "Dhaka",
    "Tangail": "Dhaka",
    "Bagerhat": "Khulna", "Chuadanga": "Khulna", "Jessore": "Khulna",
    "Jhenaidah": "Khulna", "Khulna": "Khulna", "Kushtia": "Khulna",
    "Magura": "Khulna", "Meherpur": "Khulna", "Narail": "Khulna",
    "Satkhira": "Khulna",
    "Barguna": "Barisal", "Barisal": "Barisal", "Bhola": "Barisal",
    "Jhalokati": "Barisal", "Patuakhali": "Barisal", "Pirojpur": "Barisal",
    "Habiganj": "Sylhet", "Moulvibazar": "Sylhet",
    "Sunamganj": "Sylhet", "Sylhet": "Sylhet",
    "Bogra": "Rajshahi", "Joypurhat": "Rajshahi", "Naogaon": "Rajshahi",
    "Natore": "Rajshahi", "Nawabganj": "Rajshahi", "Pabna": "Rajshahi",
    "Rajshahi": "Rajshahi", "Sirajganj": "Rajshahi",
    "Dinajpur": "Rangpur", "Gaibandha": "Rangpur", "Kurigram": "Rangpur",
    "Lalmonirhat": "Rangpur", "Nilphamari": "Rangpur", "Panchagarh": "Rangpur",
    "Rangpur": "Rangpur", "Thakurgaon": "Rangpur",
    "Jamalpur": "Mymensingh", "Mymensingh": "Mymensingh",
    "Netrakona": "Mymensingh", "Sherpur": "Mymensingh",
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )

def to_multipolygon(geom):
    # ensuring that geometry is always a MultiPolygon
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom # else if it's Multipolygon