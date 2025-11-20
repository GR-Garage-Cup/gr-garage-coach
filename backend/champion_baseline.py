import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import interpolate
import logging

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CHAMPION_PERCENTILE,
    ARC_LENGTH_RESOLUTION
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChampionBaseline:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.track_dir = PROCESSED_DATA_DIR / track_name
        self.models_dir = MODELS_DIR / track_name
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_all_laps(self) -> pd.DataFrame:
        logger.info(f"Loading all laps for {self.track_name}")

        parquet_files = list(self.track_dir.glob("*.parquet"))

        if not parquet_files:
            logger.error(f"No parquet files found in {self.track_dir}")
            return pd.DataFrame()

        all_data = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            driver_id = pq_file.stem
            df['driver_id'] = driver_id
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} telemetry points from {len(parquet_files)} drivers")

        return combined_df

    def compute_lap_times(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing lap times")

        lap_times = []

        for (driver, lap_num), group in df.groupby(['driver_id', 'lap_number']):
            if 'timestamp' in group.columns and len(group) > 1:
                group = group.sort_values('timestamp')
                lap_time = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds()

                if lap_time > 0:
                    lap_times.append({
                        'driver_id': driver,
                        'lap_number': lap_num,
                        'lap_time': lap_time,
                        'data': group
                    })

        lap_times_df = pd.DataFrame([
            {
                'driver_id': lt['driver_id'],
                'lap_number': lt['lap_number'],
                'lap_time': lt['lap_time']
            }
            for lt in lap_times
        ])

        for lt in lap_times:
            key = (lt['driver_id'], lt['lap_number'])
            df.loc[
                (df['driver_id'] == lt['driver_id']) &
                (df['lap_number'] == lt['lap_number']),
                'lap_time'
            ] = lt['lap_time']

        logger.info(f"Computed {len(lap_times)} lap times")
        return df

    def identify_champion_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Identifying champion laps (top {100*(1-CHAMPION_PERCENTILE):.0f}%)")

        if 'lap_time' not in df.columns:
            logger.error("lap_time column not found")
            return pd.DataFrame()

        lap_summary = df.groupby(['driver_id', 'lap_number']).agg({
            'lap_time': 'first'
        }).reset_index()

        lap_summary = lap_summary[lap_summary['lap_time'] > 0]

        threshold_time = lap_summary['lap_time'].quantile(CHAMPION_PERCENTILE)
        logger.info(f"Champion lap time threshold: {threshold_time:.2f}s")

        champion_laps = lap_summary[lap_summary['lap_time'] <= threshold_time]

        champion_data = df.merge(
            champion_laps[['driver_id', 'lap_number']],
            on=['driver_id', 'lap_number'],
            how='inner'
        )

        logger.info(f"Identified {len(champion_laps)} champion laps from {champion_data['driver_id'].nunique()} drivers")

        return champion_data

    def resample_by_arc_length(
        self,
        df: pd.DataFrame,
        resolution: float = ARC_LENGTH_RESOLUTION
    ) -> pd.DataFrame:
        logger.info(f"Resampling by arc length (resolution: {resolution}m)")

        if 'arc_length' not in df.columns:
            logger.error("arc_length column not found")
            return df

        max_arc = df['arc_length'].max()
        min_arc = df['arc_length'].min()

        new_arc_lengths = np.arange(min_arc, max_arc, resolution)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'arc_length']

        resampled_data = {'arc_length': new_arc_lengths}

        for col in numeric_cols:
            valid_mask = df[col].notna() & df['arc_length'].notna()
            if valid_mask.sum() > 1:
                try:
                    f = interpolate.interp1d(
                        df.loc[valid_mask, 'arc_length'],
                        df.loc[valid_mask, col],
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    resampled_data[col] = f(new_arc_lengths)
                except:
                    resampled_data[col] = np.full(len(new_arc_lengths), np.nan)

        resampled_df = pd.DataFrame(resampled_data)

        logger.info(f"Resampled to {len(resampled_df)} points")
        return resampled_df

    def compute_baseline_statistics(
        self,
        champion_laps: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Computing baseline statistics from champion laps")

        resampled_laps = []

        for (driver, lap_num), group in champion_laps.groupby(['driver_id', 'lap_number']):
            resampled = self.resample_by_arc_length(group)
            resampled['driver_id'] = driver
            resampled['lap_number'] = lap_num
            resampled_laps.append(resampled)

        all_resampled = pd.concat(resampled_laps, ignore_index=True)

        baseline = all_resampled.groupby('arc_length').agg({
            'speed': ['mean', 'std', 'count'],
            'brake_total': ['mean', 'std'] if 'brake_total' in all_resampled.columns else ['mean'],
            'ath': ['mean', 'std'] if 'ath' in all_resampled.columns else ['mean'],
            'steering_angle': ['mean', 'std'] if 'steering_angle' in all_resampled.columns else ['mean'],
            'accx_can': ['mean', 'std'] if 'accx_can' in all_resampled.columns else ['mean'],
            'accy_can': ['mean', 'std'] if 'accy_can' in all_resampled.columns else ['mean'],
        }).reset_index()

        baseline.columns = ['_'.join(col).strip('_') for col in baseline.columns.values]

        z_score = 1.96
        if 'speed_std' in baseline.columns and 'speed_count' in baseline.columns:
            baseline['speed_ci_lower'] = (
                baseline['speed_mean'] -
                z_score * baseline['speed_std'] / np.sqrt(baseline['speed_count'])
            )
            baseline['speed_ci_upper'] = (
                baseline['speed_mean'] +
                z_score * baseline['speed_std'] / np.sqrt(baseline['speed_count'])
            )

        logger.info(f"Computed baseline with {len(baseline)} arc-length points")
        return baseline

    def build_and_save_baseline(self) -> Path:
        logger.info(f"Building champion baseline for {self.track_name}")

        df = self.load_all_laps()

        if df.empty:
            logger.error("No data loaded")
            return None

        df = self.compute_lap_times(df)

        champion_laps = self.identify_champion_laps(df)

        if champion_laps.empty:
            logger.error("No champion laps identified")
            return None

        baseline = self.compute_baseline_statistics(champion_laps)

        output_path = self.models_dir / "champion_baseline.parquet"
        baseline.to_parquet(output_path, index=False)

        logger.info(f"Saved champion baseline to {output_path}")

        return output_path


def main():
    from config import TRD_TRACKS

    for track in TRD_TRACKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Building baseline for: {track}")
        logger.info(f"{'='*50}\n")

        baseline_builder = ChampionBaseline(track)
        output_path = baseline_builder.build_and_save_baseline()

        if output_path:
            logger.info(f"Successfully created baseline: {output_path}\n")
        else:
            logger.error(f"Failed to create baseline for {track}\n")


if __name__ == "__main__":
    main()
