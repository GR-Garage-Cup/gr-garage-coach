import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import logging

from config import PROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerSegmentation:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.models_dir = MODELS_DIR / track_name
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def compute_curvature(self, df: pd.DataFrame, sigma: float = 2.0) -> np.ndarray:
        logger.info("Computing trajectory curvature")

        x = df['x'].values
        y = df['y'].values

        x_smooth = gaussian_filter1d(x, sigma=sigma)
        y_smooth = gaussian_filter1d(y, sigma=sigma)

        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        return curvature

    def detect_corner_candidates(
        self,
        curvature: np.ndarray,
        min_distance: int = 50,
        prominence: float = 0.001
    ) -> np.ndarray:
        logger.info("Detecting corner candidates from curvature peaks")

        peaks, properties = find_peaks(
            curvature,
            distance=min_distance,
            prominence=prominence
        )

        logger.info(f"Found {len(peaks)} corner candidates")
        return peaks

    def label_corner_phases(
        self,
        df: pd.DataFrame,
        corner_peaks: np.ndarray,
        entry_distance: int = 30,
        exit_distance: int = 30
    ) -> pd.DataFrame:
        logger.info("Labeling corner entry, apex, and exit phases")

        df = df.copy()
        df['corner_id'] = -1
        df['corner_phase'] = 'straight'

        for corner_num, peak_idx in enumerate(corner_peaks):
            entry_idx = max(0, peak_idx - entry_distance)
            exit_idx = min(len(df) - 1, peak_idx + exit_distance)

            df.loc[entry_idx:peak_idx, 'corner_id'] = corner_num
            df.loc[entry_idx:peak_idx, 'corner_phase'] = 'entry'

            df.loc[peak_idx, 'corner_id'] = corner_num
            df.loc[peak_idx, 'corner_phase'] = 'apex'

            df.loc[peak_idx:exit_idx, 'corner_id'] = corner_num
            df.loc[peak_idx:exit_idx, 'corner_phase'] = 'exit'

        logger.info(f"Labeled {len(corner_peaks)} corners with entry/apex/exit phases")
        return df

    def extract_corner_metadata(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Extracting corner metadata")

        corners = []

        for corner_id in df['corner_id'].unique():
            if corner_id == -1:
                continue

            corner_data = df[df['corner_id'] == corner_id]

            apex_data = corner_data[corner_data['corner_phase'] == 'apex']
            entry_data = corner_data[corner_data['corner_phase'] == 'entry']
            exit_data = corner_data[corner_data['corner_phase'] == 'exit']

            if apex_data.empty:
                continue

            apex_idx = apex_data.index[0]

            corner_info = {
                'corner_id': corner_id,
                'apex_arc_length': df.loc[apex_idx, 'arc_length'] if 'arc_length' in df.columns else np.nan,
                'apex_x': df.loc[apex_idx, 'x'] if 'x' in df.columns else np.nan,
                'apex_y': df.loc[apex_idx, 'y'] if 'y' in df.columns else np.nan,
                'apex_speed': df.loc[apex_idx, 'speed'] if 'speed' in df.columns else np.nan,
                'entry_speed': entry_data['speed'].iloc[0] if not entry_data.empty and 'speed' in entry_data.columns else np.nan,
                'exit_speed': exit_data['speed'].iloc[-1] if not exit_data.empty and 'speed' in exit_data.columns else np.nan,
                'min_speed': corner_data['speed'].min() if 'speed' in corner_data.columns else np.nan,
                'max_brake': corner_data['brake_total'].max() if 'brake_total' in corner_data.columns else np.nan,
                'max_lateral_g': corner_data['accy_can'].abs().max() if 'accy_can' in corner_data.columns else np.nan,
            }

            corners.append(corner_info)

        corners_df = pd.DataFrame(corners)
        logger.info(f"Extracted metadata for {len(corners_df)} corners")

        return corners_df

    def segment_track(
        self,
        baseline_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Segmenting track: {self.track_name}")

        curvature = self.compute_curvature(baseline_df)
        baseline_df['curvature'] = curvature

        corner_peaks = self.detect_corner_candidates(curvature)

        baseline_df = self.label_corner_phases(baseline_df, corner_peaks)

        corner_metadata = self.extract_corner_metadata(baseline_df)

        segmented_path = self.models_dir / "track_segmentation.parquet"
        baseline_df.to_parquet(segmented_path, index=False)
        logger.info(f"Saved track segmentation to {segmented_path}")

        corners_path = self.models_dir / "corners_metadata.parquet"
        corner_metadata.to_parquet(corners_path, index=False)
        logger.info(f"Saved corner metadata to {corners_path}")

        return baseline_df, corner_metadata


def main():
    from champion_baseline import ChampionBaseline
    from config import TRD_TRACKS

    for track in TRD_TRACKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Segmenting track: {track}")
        logger.info(f"{'='*50}\n")

        try:
            baseline_path = MODELS_DIR / track / "champion_baseline.parquet"

            if not baseline_path.exists():
                logger.warning(f"Baseline not found for {track}, building it first...")
                baseline_builder = ChampionBaseline(track)
                baseline_builder.build_and_save_baseline()

            baseline_df = pd.read_parquet(baseline_path)

            if 'x' not in baseline_df.columns or 'y' not in baseline_df.columns:
                logger.warning(f"Missing x/y coordinates, reconstructing from first lap...")
                track_dir = PROCESSED_DATA_DIR / track
                first_file = list(track_dir.glob("*.parquet"))[0]
                lap_data = pd.read_parquet(first_file)
                first_lap = lap_data[lap_data['lap_number'] == lap_data['lap_number'].min()]

                baseline_df = baseline_df.merge(
                    first_lap[['arc_length', 'x', 'y']],
                    on='arc_length',
                    how='left'
                )

            segmenter = CornerSegmentation(track)
            segmented_df, corners_df = segmenter.segment_track(baseline_df)

            logger.info(f"Successfully segmented {track}")
            logger.info(f"Found {len(corners_df)} corners\n")

        except Exception as e:
            logger.error(f"Error segmenting {track}: {e}\n")
            continue


if __name__ == "__main__":
    main()
