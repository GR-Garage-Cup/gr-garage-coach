import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import interpolate
import logging

from config import MODELS_DIR, ARC_LENGTH_RESOLUTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoachingEngine:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.models_dir = MODELS_DIR / track_name

        self.baseline = self.load_baseline()
        self.corners = self.load_corners()
        self.segmentation = self.load_segmentation()

    def load_baseline(self) -> pd.DataFrame:
        baseline_path = self.models_dir / "champion_baseline.parquet"

        if not baseline_path.exists():
            logger.error(f"Baseline not found: {baseline_path}")
            return pd.DataFrame()

        baseline = pd.read_parquet(baseline_path)
        logger.info(f"Loaded baseline with {len(baseline)} points")
        return baseline

    def load_corners(self) -> pd.DataFrame:
        corners_path = self.models_dir / "corners_metadata.parquet"

        if not corners_path.exists():
            logger.warning(f"Corners metadata not found: {corners_path}")
            return pd.DataFrame()

        corners = pd.read_parquet(corners_path)
        logger.info(f"Loaded {len(corners)} corners")
        return corners

    def load_segmentation(self) -> pd.DataFrame:
        seg_path = self.models_dir / "track_segmentation.parquet"

        if not seg_path.exists():
            logger.warning(f"Track segmentation not found: {seg_path}")
            return pd.DataFrame()

        seg = pd.read_parquet(seg_path)
        logger.info(f"Loaded track segmentation")
        return seg

    def align_lap_to_baseline(
        self,
        lap_df: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Aligning lap to baseline arc length")

        if 'arc_length' not in lap_df.columns:
            logger.error("No arc_length in lap data")
            return lap_df

        min_arc = self.baseline['arc_length'].min()
        max_arc = self.baseline['arc_length'].max()

        aligned_arc = np.arange(min_arc, max_arc, ARC_LENGTH_RESOLUTION)

        aligned_data = {'arc_length': aligned_arc}

        numeric_cols = lap_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'arc_length']

        for col in numeric_cols:
            valid_mask = lap_df[col].notna() & lap_df['arc_length'].notna()

            if valid_mask.sum() > 1:
                try:
                    f = interpolate.interp1d(
                        lap_df.loc[valid_mask, 'arc_length'],
                        lap_df.loc[valid_mask, col],
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    aligned_data[col] = f(aligned_arc)
                except:
                    aligned_data[col] = np.full(len(aligned_arc), np.nan)

        aligned_df = pd.DataFrame(aligned_data)
        return aligned_df

    def compute_delta(
        self,
        lap_df: pd.DataFrame
    ) -> pd.DataFrame:
        logger.info("Computing delta against baseline")

        aligned_lap = self.align_lap_to_baseline(lap_df)

        merged = pd.merge(
            self.baseline,
            aligned_lap,
            on='arc_length',
            how='inner',
            suffixes=('_baseline', '_user')
        )

        if 'speed_mean' in merged.columns and 'speed_user' in merged.columns:
            merged['speed_delta'] = merged['speed_user'] - merged['speed_mean']

        if 'brake_total_mean' in merged.columns and 'brake_total_user' in merged.columns:
            merged['brake_delta'] = merged['brake_total_user'] - merged['brake_total_mean']

        if 'ath_mean' in merged.columns and 'ath_user' in merged.columns:
            merged['throttle_delta'] = merged['ath_user'] - merged['ath_mean']

        return merged

    def identify_brake_point_issues(
        self,
        delta_df: pd.DataFrame,
        corner_id: int
    ) -> Dict:
        logger.info(f"Analyzing brake points for corner {corner_id}")

        corner_data = delta_df[delta_df['corner_id'] == corner_id]

        if corner_data.empty:
            return {}

        entry_data = corner_data[corner_data['corner_phase'] == 'entry']

        if entry_data.empty or 'brake_total_user' not in entry_data.columns:
            return {}

        brake_threshold = 50

        user_brake = entry_data[entry_data['brake_total_user'] > brake_threshold]
        baseline_brake = entry_data[entry_data['brake_total_mean'] > brake_threshold]

        if user_brake.empty or baseline_brake.empty:
            return {}

        user_brake_point = user_brake['arc_length'].iloc[0]
        baseline_brake_point = baseline_brake['arc_length'].iloc[0]

        brake_delta = user_brake_point - baseline_brake_point

        return {
            'corner_id': corner_id,
            'user_brake_point': user_brake_point,
            'baseline_brake_point': baseline_brake_point,
            'brake_delta_meters': brake_delta,
            'issue': 'early' if brake_delta < -5 else 'late' if brake_delta > 5 else 'optimal'
        }

    def identify_throttle_issues(
        self,
        delta_df: pd.DataFrame,
        corner_id: int
    ) -> Dict:
        logger.info(f"Analyzing throttle application for corner {corner_id}")

        corner_data = delta_df[delta_df['corner_id'] == corner_id]

        if corner_data.empty:
            return {}

        exit_data = corner_data[corner_data['corner_phase'] == 'exit']

        if exit_data.empty or 'ath_user' not in exit_data.columns:
            return {}

        throttle_threshold = 30

        user_throttle = exit_data[exit_data['ath_user'] > throttle_threshold]
        baseline_throttle = exit_data[exit_data['ath_mean'] > throttle_threshold]

        if user_throttle.empty or baseline_throttle.empty:
            return {}

        user_throttle_point = user_throttle['arc_length'].iloc[0]
        baseline_throttle_point = baseline_throttle['arc_length'].iloc[0]

        throttle_delta = user_throttle_point - baseline_throttle_point

        return {
            'corner_id': corner_id,
            'user_throttle_point': user_throttle_point,
            'baseline_throttle_point': baseline_throttle_point,
            'throttle_delta_meters': throttle_delta,
            'issue': 'late' if throttle_delta > 3 else 'early' if throttle_delta < -3 else 'optimal'
        }

    def analyze_lap(
        self,
        lap_df: pd.DataFrame
    ) -> Dict:
        logger.info("Performing full lap analysis")

        delta_df = self.compute_delta(lap_df)

        if self.segmentation is not None and not self.segmentation.empty:
            delta_df = delta_df.merge(
                self.segmentation[['arc_length', 'corner_id', 'corner_phase']],
                on='arc_length',
                how='left'
            )

        corner_issues = []

        if not self.corners.empty and 'corner_id' in delta_df.columns:
            for corner_id in self.corners['corner_id'].unique():
                brake_analysis = self.identify_brake_point_issues(delta_df, corner_id)
                throttle_analysis = self.identify_throttle_issues(delta_df, corner_id)

                if brake_analysis or throttle_analysis:
                    corner_issues.append({
                        'corner_id': corner_id,
                        'brake_analysis': brake_analysis,
                        'throttle_analysis': throttle_analysis
                    })

        if 'speed_delta' in delta_df.columns:
            avg_speed_delta = delta_df['speed_delta'].mean()
        else:
            avg_speed_delta = 0.0

        return {
            'track': self.track_name,
            'average_speed_delta': avg_speed_delta,
            'corner_issues': corner_issues,
            'delta_data': delta_df
        }

    def generate_coaching_recommendations(
        self,
        analysis: Dict
    ) -> List[str]:
        logger.info("Generating coaching recommendations")

        recommendations = []

        for corner_issue in analysis.get('corner_issues', []):
            corner_id = corner_issue['corner_id']
            brake_analysis = corner_issue.get('brake_analysis', {})
            throttle_analysis = corner_issue.get('throttle_analysis', {})

            if brake_analysis.get('issue') == 'early':
                delta = abs(brake_analysis['brake_delta_meters'])
                rec = (
                    f"Corner {corner_id}: You are braking {delta:.1f}m too early. "
                    f"Champions brake at {brake_analysis['baseline_brake_point']:.1f}m. "
                    f"Estimated time gain: {delta * 0.02:.2f}s."
                )
                recommendations.append(rec)

            elif brake_analysis.get('issue') == 'late':
                delta = abs(brake_analysis['brake_delta_meters'])
                rec = (
                    f"Corner {corner_id}: You are braking {delta:.1f}m too late. "
                    f"This may cause overshoot. Champions brake at {brake_analysis['baseline_brake_point']:.1f}m."
                )
                recommendations.append(rec)

            if throttle_analysis.get('issue') == 'late':
                delta = abs(throttle_analysis['throttle_delta_meters'])
                rec = (
                    f"Corner {corner_id}: Throttle application is {delta:.1f}m late. "
                    f"Apply throttle at {throttle_analysis['baseline_throttle_point']:.1f}m. "
                    f"Estimated time gain: {delta * 0.03:.2f}s."
                )
                recommendations.append(rec)

            elif throttle_analysis.get('issue') == 'early':
                delta = abs(throttle_analysis['throttle_delta_meters'])
                rec = (
                    f"Corner {corner_id}: Throttle application is {delta:.1f}m early. "
                    f"Risk of oversteer or understeer. Wait until {throttle_analysis['baseline_throttle_point']:.1f}m."
                )
                recommendations.append(rec)

        if not recommendations:
            recommendations.append("Your lap is very close to the baseline. Focus on consistency.")

        return recommendations


def main():
    logger.info("Coaching engine module loaded successfully")


if __name__ == "__main__":
    main()
