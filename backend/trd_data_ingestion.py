"""
TRD Hackathon 2025 Data Ingestion Module

Processes real Toyota Racing Development (TRD) telemetry data from GR Cup championship.

Data Format (from trddev.com/hackathon-2025/):
- Speed (km/h)
- Gear
- nmot (Engine RPM)
- ath (throttle blade position, 0-100%)
- aps (accelerator pedal sensor, 0-100%)
- pbrake_f (front brake pressure, bar)
- pbrake_r (rear brake pressure, bar)
- accx_can (longitudinal acceleration, G)
- accy_can (lateral acceleration, G)
- Steering_Angle (degrees)
- VBOX_Long_Minutes, VBOX_Lat_Min (GPS coordinates in decimal minutes)
- Laptrigger_lapdist_dls (lap distance from start/finish, meters)
- timestamp (datetime)

Available Tracks:
1. Barber Motorsports Park
2. Circuit of the Americas (COTA)
3. Indianapolis
4. Road America
5. Sebring
6. Sonoma
7. Virginia International Raceway (VIR)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
from pyproj import Transformer
from scipy import signal
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Track metadata"""
    name: str
    length_meters: float
    corners: int
    sectors: int
    latitude: float  # Track center
    longitude: float  # Track center
    utm_zone: int


# Track database with real coordinates
TRACK_DATABASE = {
    "road_america": TrackInfo(
        name="Road America",
        length_meters=6515.0,
        corners=14,
        sectors=3,
        latitude=43.7980,
        longitude=-87.9934,
        utm_zone=16
    ),
    "cota": TrackInfo(
        name="Circuit of the Americas",
        length_meters=5513.0,
        corners=20,
        sectors=3,
        latitude=30.1328,
        longitude=-97.6411,
        utm_zone=14
    ),
    "barber": TrackInfo(
        name="Barber Motorsports Park",
        length_meters=3700.0,
        corners=17,
        sectors=3,
        latitude=33.5435,
        longitude=-86.3705,
        utm_zone=16
    ),
    "indianapolis": TrackInfo(
        name="Indianapolis Motor Speedway (Road Course)",
        length_meters=4192.0,
        corners=14,
        sectors=3,
        latitude=39.7950,
        longitude=-86.2352,
        utm_zone=16
    ),
    "sebring": TrackInfo(
        name="Sebring International Raceway",
        length_meters=6019.0,
        corners=17,
        sectors=3,
        latitude=27.4645,
        longitude=-81.3483,
        utm_zone=17
    ),
    "sonoma": TrackInfo(
        name="Sonoma Raceway",
        length_meters=4052.0,
        corners=12,
        sectors=3,
        latitude=38.1616,
        longitude=-122.4570,
        utm_zone=10
    ),
    "vir": TrackInfo(
        name="Virginia International Raceway",
        length_meters=5557.0,
        corners=20,
        sectors=3,
        latitude=36.5859,
        longitude=-79.2023,
        utm_zone=17
    )
}


class TRDDataParser:
    """
    Parser for real TRD GR Cup telemetry CSV files.

    Handles the exact format provided by trddev.com/hackathon-2025/
    """

    # Column name mappings from TRD format to internal format
    TRD_COLUMN_MAP = {
        'Speed': 'speed',  # km/h
        'Gear': 'gear',
        'nmot': 'nmot',  # Engine RPM
        'ath': 'ath',  # Throttle blade %
        'aps': 'aps',  # Pedal position %
        'pbrake_f': 'brake_front',  # Bar
        'pbrake_r': 'brake_rear',  # Bar
        'accx_can': 'accx_can',  # Longitudinal G
        'accy_can': 'accy_can',  # Lateral G
        'Steering_Angle': 'steering_angle',  # Degrees
        'VBOX_Long_Minutes': 'longitude',
        'VBOX_Lat_Min': 'latitude',
        'Laptrigger_lapdist_dls': 'laptrigger_lapdist_dls'  # Meters from start/finish
    }

    def __init__(self, track_name: str = "road_america"):
        self.track_info = TRACK_DATABASE.get(track_name.lower().replace(" ", "_"))
        if not self.track_info:
            raise ValueError(f"Unknown track: {track_name}. Available: {list(TRACK_DATABASE.keys())}")

        # GPS to UTM transformer for precise XY coordinates
        self.gps_to_utm = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (GPS)
            f"EPSG:326{self.track_info.utm_zone:02d}",  # UTM North
            always_xy=True
        )

        logger.info(f"Initialized TRD parser for {self.track_info.name}")

    def parse_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Parse TRD CSV file and return normalized DataFrame.

        Args:
            file_path: Path to TRD CSV file

        Returns:
            DataFrame with normalized columns and computed features
        """
        logger.info(f"Parsing TRD CSV: {file_path}")

        # Read CSV with TRD format
        df = pd.read_csv(file_path)

        # Rename columns to internal format
        df_renamed = df.rename(columns=self.TRD_COLUMN_MAP)

        # Parse timestamp if present
        if 'timestamp' in df.columns:
            df_renamed['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Convert GPS coordinates from decimal minutes to decimal degrees
        if 'longitude' in df_renamed.columns and 'latitude' in df_renamed.columns:
            df_renamed = self._convert_gps_coordinates(df_renamed)

        # Compute derived features
        df_enriched = self._enrich_telemetry(df_renamed)

        # Segment into laps
        df_enriched = self._segment_laps(df_enriched)

        logger.info(f"Parsed {len(df_enriched)} telemetry points across {df_enriched['lap_number'].max()} laps")

        return df_enriched

    def _convert_gps_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert VBOX GPS format (decimal minutes) to standard decimal degrees.

        VBOX format: degrees stored as integer part, minutes as decimal part
        Example: 43.7980 degrees = 43 + (0.7980 * 60) minutes = 43 + 47.88 minutes

        TRD stores as decimal minutes directly, so we need to convert back.
        """
        # Convert decimal minutes to decimal degrees
        # Format: whole degrees + (decimal_minutes / 60)
        if 'longitude' in df.columns:
            lon_degrees = np.floor(df['longitude'])
            lon_minutes = (df['longitude'] - lon_degrees) * 100
            df['longitude'] = lon_degrees + (lon_minutes / 60.0)

        if 'latitude' in df.columns:
            lat_degrees = np.floor(df['latitude'])
            lat_minutes = (df['latitude'] - lat_degrees) * 100
            df['latitude'] = lat_degrees + (lat_minutes / 60.0)

        # Project to UTM XY coordinates (meters)
        if 'longitude' in df.columns and 'latitude' in df.columns:
            x, y = self.gps_to_utm.transform(
                df['longitude'].values,
                df['latitude'].values
            )
            df['x'] = x
            df['y'] = y

        return df

    def _enrich_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from raw telemetry.

        Physics-based calculations for vehicle dynamics analysis.
        """
        # Total brake pressure (combined front + rear)
        if 'brake_front' in df.columns and 'brake_rear' in df.columns:
            df['brake_total'] = df['brake_front'] + df['brake_rear']

        # Speed in m/s for physics calculations
        if 'speed' in df.columns:
            df['speed_mps'] = df['speed'] / 3.6

        # Compute arc length (cumulative distance along track)
        if 'x' in df.columns and 'y' in df.columns:
            dx = np.diff(df['x'].values, prepend=df['x'].iloc[0])
            dy = np.diff(df['y'].values, prepend=df['y'].iloc[0])
            distances = np.sqrt(dx**2 + dy**2)

            # Remove outliers (GPS jumps)
            distances[distances > 50] = 0  # Max 50m between 10Hz samples

            df['arc_length'] = np.cumsum(distances)
        elif 'laptrigger_lapdist_dls' in df.columns:
            # Use TRD's built-in lap distance if available
            df['arc_length'] = df['laptrigger_lapdist_dls']

        # Compute time deltas
        if 'timestamp' in df.columns:
            df['dt'] = df['timestamp'].diff().dt.total_seconds()
            df['dt'] = df['dt'].fillna(0.1)  # Assume 10Hz if missing
        else:
            df['dt'] = 0.1  # 10Hz sampling rate

        # Curvature estimation (for corner detection)
        if 'x' in df.columns and 'y' in df.columns:
            df['curvature'] = self._estimate_curvature(df['x'].values, df['y'].values)

        return df

    def _estimate_curvature(self, x: np.ndarray, y: np.ndarray, window_size: int = 11) -> np.ndarray:
        """
        Estimate track curvature using finite differences.

        Curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

        Args:
            x: X coordinates
            y: Y coordinates
            window_size: Smoothing window size

        Returns:
            Curvature array (1/meters)
        """
        # Smooth coordinates to reduce GPS noise
        x_smooth = signal.savgol_filter(x, window_size, 3)
        y_smooth = signal.savgol_filter(y, window_size, 3)

        # First derivatives
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)

        curvature = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-10
        )

        return curvature

    def _segment_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment telemetry into individual laps.

        Uses lap distance (laptrigger_lapdist_dls) or arc length resets.
        """
        if 'laptrigger_lapdist_dls' in df.columns:
            # Detect lap transitions (distance resets to near 0)
            lap_resets = (df['laptrigger_lapdist_dls'].diff() < -1000).astype(int)
            df['lap_number'] = lap_resets.cumsum() + 1
        elif 'arc_length' in df.columns:
            # Detect when arc_length decreases (new lap)
            lap_resets = (df['arc_length'].diff() < -100).astype(int)
            df['lap_number'] = lap_resets.cumsum() + 1
        else:
            # No lap segmentation possible
            df['lap_number'] = 1

        return df

    def extract_lap(self, df: pd.DataFrame, lap_number: int) -> pd.DataFrame:
        """Extract a single lap from the telemetry."""
        return df[df['lap_number'] == lap_number].copy()

    def get_fastest_lap(self, df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
        """
        Find the fastest lap in the session.

        Returns:
            (lap_number, lap_dataframe)
        """
        lap_times = []
        for lap_num in df['lap_number'].unique():
            lap_data = self.extract_lap(df, lap_num)
            if 'timestamp' in lap_data.columns and len(lap_data) > 10:
                lap_time = (lap_data['timestamp'].max() - lap_data['timestamp'].min()).total_seconds()
                lap_times.append((lap_num, lap_time, lap_data))

        if not lap_times:
            return 1, df

        # Return fastest lap
        fastest = min(lap_times, key=lambda x: x[1])
        logger.info(f"Fastest lap: {fastest[0]} ({fastest[1]:.2f}s)")

        return fastest[0], fastest[2]


def main():
    """Test TRD data parser"""
    parser = TRDDataParser("road_america")

    # Test with sample file if available
    test_file = Path("/Users/ldm/Desktop/GARAGE COACH/sample_spa_lap.csv")

    if test_file.exists():
        df = parser.parse_csv(test_file)
        print(f"\nParsed {len(df)} points")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst row:\n{df.iloc[0]}")
        print(f"\nLaps: {df['lap_number'].max()}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    main()
