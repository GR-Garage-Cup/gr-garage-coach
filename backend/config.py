from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

TRD_TRACKS = [
    "barber",
    "cota",
    "indianapolis",
    "road_america",
    "sebring",
    "sonoma",
    "vir"
]

TELEMETRY_COLUMNS = [
    "timestamp",
    "speed",
    "gear",
    "nmot",
    "ath",
    "aps",
    "brake_front",
    "brake_rear",
    "accx_can",
    "accy_can",
    "steering_angle",
    "longitude",
    "latitude",
    "laptrigger_lapdist_dls"
]

CHAMPION_PERCENTILE = 0.90
ARC_LENGTH_RESOLUTION = 1.0
GPS_PROJECTION_EPSG = 32616
