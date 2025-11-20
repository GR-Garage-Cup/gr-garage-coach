"""
TRD Real Data Processor

Converts TRD long-format telemetry to wide-format and builds champion baselines.
Uses ONLY real data from https://trddev.com/hackathon-2025/

Long-format (TRD):
  - Each row = ONE telemetry parameter at ONE timestamp
  - Example: Row 1 = accx_can=-0.859 at timestamp=2025-08-14T22:38:33.357Z

Wide-format (needed for analysis):
  - Each row = ALL telemetry parameters at ONE timestamp
  - Example: Row 1 = timestamp, speed, accx_can, accy_can, throttle, brake, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LapInfo:
    """Metadata for a single lap"""
    vehicle_id: str
    lap_number: int
    lap_time_ms: int
    lap_time_seconds: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    data_points: int


class TRDRealDataProcessor:
    """
    Process real TRD championship data from long-format to wide-format.
    Build champion baselines from top 10% fastest laps.
    """

    def __init__(self, track_key: str, data_dir: Path):
        self.track_key = track_key
        self.data_dir = Path(data_dir)
        self.track_data_dir = self.data_dir / track_key

        # Expected telemetry channels from TRD data
        self.telemetry_channels = [
            'Speed', 'Gear', 'nmot', 'ath', 'aps',
            'pbrake_f', 'pbrake_r', 'accx_can', 'accy_can',
            'Steering_Angle', 'VBOX_Long_Minutes', 'VBOX_Lat_Min',
            'Laptrigger_lapdist_dls'
        ]

    def find_race_files(self) -> Dict[str, Dict[str, Path]]:
        """
        Find all race data files (telemetry + lap times).

        Returns:
            Dict mapping race -> file type -> path
            Example: {'R1': {'telemetry': Path(...), 'lap_times': Path(...)}}
        """
        logger.info(f"Searching for race files in {self.track_data_dir}")

        races = {}

        # Find all race directories
        race_dirs = list(self.track_data_dir.rglob("Race *"))

        for race_dir in race_dirs:
            race_name = race_dir.name.replace("Race ", "R")

            # Find telemetry file
            telemetry_files = list(race_dir.glob("*_telemetry_data.csv"))
            lap_time_files = list(race_dir.glob("*_lap_time_*.csv"))

            if telemetry_files and lap_time_files:
                races[race_name] = {
                    'telemetry': telemetry_files[0],
                    'lap_times': lap_time_files[0],
                    'race_dir': race_dir
                }
                logger.info(f"Found {race_name}: {telemetry_files[0].name}")

        logger.info(f"Found {len(races)} races with complete data")
        return races

    def load_lap_times(self, lap_times_file: Path, min_lap_time_s: float = 140.0, max_lap_time_s: float = 300.0) -> pd.DataFrame:
        """
        Load lap times from TRD lap time file.

        Format:
            expire_at,lap,meta_event,meta_session,meta_source,meta_time,outing,timestamp,value,vehicle_id
            ,1,I_R05_2025-08-17,R1,kafka:gr-raw,2025-08-16T19:30:57.511Z,0,2025-08-16T19:30:56.852Z,903555,GR86-010-16

        Args:
            lap_times_file: Path to lap times CSV
            min_lap_time_s: Minimum valid lap time in seconds (default 140s = 2m20s)
            max_lap_time_s: Maximum valid lap time in seconds (default 300s = 5m00s)

        Returns:
            DataFrame with columns: vehicle_id, lap_number, lap_time_ms, lap_time_seconds, timestamp
        """
        logger.info(f"Loading lap times from {lap_times_file.name}")

        df = pd.read_csv(lap_times_file)

        # Filter to actual lap times (lap != 32768 which is invalid, lap != 0)
        df = df[(df['lap'] != 32768) & (df['lap'] != 0)].copy()

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Rename columns
        df = df.rename(columns={'lap': 'lap_number', 'value': 'lap_time_ms'})

        # Convert lap time to seconds
        df['lap_time_seconds'] = df['lap_time_ms'] / 1000.0

        # Filter to realistic lap times (remove warmup laps, incomplete laps, etc.)
        df = df[
            (df['lap_time_seconds'] >= min_lap_time_s) &
            (df['lap_time_seconds'] <= max_lap_time_s)
        ].copy()

        # Sort by lap time (fastest first)
        df = df.sort_values('lap_time_seconds')

        logger.info(f"Loaded {len(df)} valid lap times (between {min_lap_time_s}s and {max_lap_time_s}s)")
        if len(df) > 0:
            logger.info(f"Fastest lap: {df['lap_time_seconds'].min():.3f}s by {df.iloc[0]['vehicle_id']}")

        return df[['vehicle_id', 'lap_number', 'lap_time_ms', 'lap_time_seconds', 'timestamp']]

    def load_telemetry_long_format(self, telemetry_file: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load TRD telemetry in long format.

        Format:
            expire_at,lap,meta_event,meta_session,meta_source,meta_time,original_vehicle_id,outing,telemetry_name,telemetry_value,timestamp,vehicle_id,vehicle_number
            ,1,I_R05_2025-08-17,R1,kafka:gr-raw,2025-08-16T19:30:59.328Z,GR86-002-2,0,accx_can,0.031,2025-08-14T22:38:33.357Z,GR86-002-2,2
        """
        logger.info(f"Loading telemetry from {telemetry_file.name}")

        if max_rows:
            df = pd.read_csv(telemetry_file, nrows=max_rows)
        else:
            df = pd.read_csv(telemetry_file)

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Loaded {len(df)} telemetry rows (long format)")

        return df

    def pivot_telemetry_to_wide_format(
        self,
        telemetry_long: pd.DataFrame,
        vehicle_id: str,
        lap_number: int
    ) -> pd.DataFrame:
        """
        Convert long-format telemetry to wide-format for a specific lap.

        Args:
            telemetry_long: Long-format telemetry DataFrame
            vehicle_id: Vehicle ID (e.g., 'GR86-010-16')
            lap_number: Lap number

        Returns:
            Wide-format DataFrame with one row per timestamp
        """
        logger.info(f"Pivoting telemetry for {vehicle_id} lap {lap_number}")

        # Filter to specific vehicle and lap
        lap_data = telemetry_long[
            (telemetry_long['vehicle_id'] == vehicle_id) &
            (telemetry_long['lap'] == lap_number)
        ].copy()

        if len(lap_data) == 0:
            logger.warning(f"No telemetry found for {vehicle_id} lap {lap_number}")
            return pd.DataFrame()

        # Pivot from long to wide format
        wide_df = lap_data.pivot_table(
            index='timestamp',
            columns='telemetry_name',
            values='telemetry_value',
            aggfunc='first'  # Use first value if duplicates
        ).reset_index()

        # Sort by timestamp
        wide_df = wide_df.sort_values('timestamp')

        logger.info(f"Converted to wide format: {len(wide_df)} timestamps, {len(wide_df.columns)-1} channels")

        return wide_df

    def extract_champion_laps(
        self,
        lap_times_df: pd.DataFrame,
        percentile: float = 0.10
    ) -> List[Tuple[str, int, float]]:
        """
        Identify top percentile fastest laps.

        Args:
            lap_times_df: DataFrame with lap times
            percentile: Top percentile to select (0.10 = top 10%)

        Returns:
            List of (vehicle_id, lap_number, lap_time_seconds) tuples
        """
        n_champion_laps = max(1, int(len(lap_times_df) * percentile))

        champion_laps = lap_times_df.head(n_champion_laps)

        logger.info(f"Selected top {percentile*100}% = {n_champion_laps} champion laps")
        logger.info(f"Lap time range: {champion_laps['lap_time_seconds'].min():.3f}s - {champion_laps['lap_time_seconds'].max():.3f}s")

        return list(zip(
            champion_laps['vehicle_id'],
            champion_laps['lap_number'],
            champion_laps['lap_time_seconds']
        ))

    def process_single_lap_to_wide_format(
        self,
        telemetry_file: Path,
        vehicle_id: str,
        lap_number: int,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Process a single lap from long-format to wide-format CSV.

        Args:
            telemetry_file: Path to telemetry CSV (long format)
            vehicle_id: Vehicle ID
            lap_number: Lap number
            output_dir: Directory to save wide-format CSV

        Returns:
            Path to saved CSV, or None if failed
        """
        # Load telemetry (we need to load full file to get the lap)
        telemetry_long = self.load_telemetry_long_format(telemetry_file)

        # Convert to wide format
        wide_df = self.pivot_telemetry_to_wide_format(telemetry_long, vehicle_id, lap_number)

        if wide_df.empty:
            return None

        # Add lap number column
        wide_df['lap_number'] = lap_number

        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{vehicle_id}_lap{lap_number}.csv"
        wide_df.to_csv(output_file, index=False)

        logger.info(f"Saved wide-format CSV: {output_file}")

        return output_file

    def build_champion_baseline(
        self,
        output_dir: Path,
        percentile: float = 0.10
    ) -> Dict:
        """
        Build champion baseline from top percentile of laps across all races.

        Process:
        1. Find all race files (R1, R2, etc.)
        2. Load lap times from all races
        3. Identify top 10% fastest laps
        4. Convert those laps to wide-format
        5. Compute statistical baseline (mean, std, percentiles)

        Args:
            output_dir: Directory to save baseline data
            percentile: Top percentile to use (0.10 = top 10%)

        Returns:
            Dict with baseline metadata
        """
        logger.info(f"Building champion baseline for {self.track_key}")

        # Find all race files
        races = self.find_race_files()

        if not races:
            raise ValueError(f"No race data found for {self.track_key}")

        # Load lap times from all races
        all_lap_times = []

        for race_name, files in races.items():
            lap_times = self.load_lap_times(files['lap_times'])
            lap_times['race'] = race_name
            all_lap_times.append(lap_times)

        combined_lap_times = pd.concat(all_lap_times, ignore_index=True)

        logger.info(f"Total laps across all races: {len(combined_lap_times)}")

        # Extract champion laps (top 10%)
        champion_laps = self.extract_champion_laps(combined_lap_times, percentile)

        # Save champion laps metadata
        champion_metadata = []

        for vehicle_id, lap_number, lap_time in champion_laps:
            # Find which race this lap belongs to
            race_info = combined_lap_times[
                (combined_lap_times['vehicle_id'] == vehicle_id) &
                (combined_lap_times['lap_number'] == lap_number)
            ].iloc[0]

            champion_metadata.append({
                'vehicle_id': vehicle_id,
                'lap_number': int(lap_number),
                'lap_time_seconds': float(lap_time),
                'race': race_info['race']
            })

        # Save metadata
        metadata_file = output_dir / f"{self.track_key}_champion_laps.json"
        with open(metadata_file, 'w') as f:
            json.dump(champion_metadata, f, indent=2)

        logger.info(f"Saved champion laps metadata: {metadata_file}")

        summary = {
            'track': self.track_key,
            'total_laps': len(combined_lap_times),
            'champion_laps': len(champion_laps),
            'percentile': percentile,
            'fastest_lap_seconds': float(champion_laps[0][2]),
            'slowest_champion_lap_seconds': float(champion_laps[-1][2]),
            'metadata_file': str(metadata_file)
        }

        return summary


def main():
    """
    Process TRD real data for all tracks.
    """
    import argparse

    parser = argparse.ArgumentParser(description="TRD Real Data Processor")
    parser.add_argument('--track', type=str, required=True, help='Track key (e.g., road_america)')
    parser.add_argument('--data-dir', type=Path, required=True, help='TRD data directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for baselines')
    parser.add_argument('--percentile', type=float, default=0.10, help='Top percentile for champion baseline')

    args = parser.parse_args()

    processor = TRDRealDataProcessor(args.track, args.data_dir)

    summary = processor.build_champion_baseline(args.output_dir, args.percentile)

    logger.info("\n" + "="*60)
    logger.info("CHAMPION BASELINE BUILD COMPLETE")
    logger.info("="*60)
    logger.info(f"Track: {summary['track']}")
    logger.info(f"Total laps: {summary['total_laps']}")
    logger.info(f"Champion laps (top {summary['percentile']*100}%): {summary['champion_laps']}")
    logger.info(f"Fastest lap: {summary['fastest_lap_seconds']:.3f}s")
    logger.info(f"Metadata: {summary['metadata_file']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
