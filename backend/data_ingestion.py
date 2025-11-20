import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pyproj
from shapely.geometry import Point, LineString
import logging

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TELEMETRY_COLUMNS,
    GPS_PROJECTION_EPSG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryIngestion:

    def __init__(self):
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{GPS_PROJECTION_EPSG}",
            always_xy=True
        )

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading telemetry from {filepath}")
        df = pd.read_csv(filepath)

        if 'timestamp' in df.columns or 'time' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')

        return df

    def project_gps_to_xy(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Projecting GPS coordinates to XY")

        valid_gps = df[['longitude', 'latitude']].notna().all(axis=1)

        x_coords = np.full(len(df), np.nan)
        y_coords = np.full(len(df), np.nan)

        if valid_gps.any():
            lons = df.loc[valid_gps, 'longitude'].values
            lats = df.loc[valid_gps, 'latitude'].values

            x, y = self.transformer.transform(lons, lats)
            x_coords[valid_gps] = x
            y_coords[valid_gps] = y

        df['x'] = x_coords
        df['y'] = y_coords

        return df

    def segment_laps(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        logger.info("Segmenting telemetry into individual laps")

        if 'laptrigger_lapdist_dls' not in df.columns:
            logger.warning("No lap distance column found, returning single lap")
            return {1: df}

        df = df.sort_values('timestamp').reset_index(drop=True)

        distance = df['laptrigger_lapdist_dls'].values
        lap_starts = np.where(np.diff(distance) < -100)[0] + 1
        lap_starts = np.concatenate([[0], lap_starts, [len(df)]])

        laps = {}
        for i in range(len(lap_starts) - 1):
            start_idx = lap_starts[i]
            end_idx = lap_starts[i + 1]

            lap_df = df.iloc[start_idx:end_idx].copy()

            if len(lap_df) > 10:
                laps[i + 1] = lap_df.reset_index(drop=True)

        logger.info(f"Found {len(laps)} valid laps")
        return laps

    def compute_arc_length(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing arc length along trajectory")

        x = df['x'].values
        y = df['y'].values

        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)

        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        df['arc_length'] = arc_length

        return df

    def normalize_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing telemetry data")

        df = df.copy()

        if 'speed' in df.columns:
            df['speed'] = df['speed'].clip(lower=0)

        if 'brake_front' in df.columns and 'brake_rear' in df.columns:
            df['brake_total'] = df['brake_front'] + df['brake_rear']

        if 'accx_can' in df.columns and 'accy_can' in df.columns:
            df['acc_magnitude'] = np.sqrt(
                df['accx_can']**2 + df['accy_can']**2
            )

        return df

    def process_telemetry_file(
        self,
        filepath: Path,
        track_name: str
    ) -> Dict[str, any]:
        logger.info(f"Processing telemetry file for {track_name}")

        df = self.load_csv(filepath)
        df = self.project_gps_to_xy(df)
        df = self.normalize_telemetry(df)

        laps = self.segment_laps(df)

        processed_laps = {}
        for lap_num, lap_df in laps.items():
            lap_df = self.compute_arc_length(lap_df)
            processed_laps[lap_num] = lap_df

        output_dir = PROCESSED_DATA_DIR / track_name
        output_dir.mkdir(parents=True, exist_ok=True)

        driver_id = filepath.stem
        output_file = output_dir / f"{driver_id}.parquet"

        all_laps_df = pd.concat([
            lap_df.assign(lap_number=lap_num)
            for lap_num, lap_df in processed_laps.items()
        ])
        all_laps_df.to_parquet(output_file, index=False)

        logger.info(f"Saved processed data to {output_file}")

        return {
            'track': track_name,
            'driver': driver_id,
            'num_laps': len(processed_laps),
            'filepath': output_file
        }

    def process_track_directory(self, track_name: str) -> List[Dict]:
        logger.info(f"Processing all files for track: {track_name}")

        track_dir = RAW_DATA_DIR / track_name

        if not track_dir.exists():
            logger.error(f"Track directory not found: {track_dir}")
            return []

        csv_files = list(track_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        results = []
        for csv_file in csv_files:
            try:
                result = self.process_telemetry_file(csv_file, track_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue

        logger.info(f"Successfully processed {len(results)}/{len(csv_files)} files")
        return results


def main():
    ingestion = TelemetryIngestion()

    from config import TRD_TRACKS

    for track in TRD_TRACKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing track: {track}")
        logger.info(f"{'='*50}\n")

        results = ingestion.process_track_directory(track)

        logger.info(f"\nCompleted {track}: {len(results)} drivers processed\n")


if __name__ == "__main__":
    main()
