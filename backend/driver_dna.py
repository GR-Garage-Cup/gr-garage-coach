import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import pickle

from config import PROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriverDNA:

    def __init__(self, track_name: str = None):
        self.track_name = track_name
        self.scaler = StandardScaler()
        self.kmeans = None
        self.archetype_names = [
            "Late Braker",
            "Smooth Operator",
            "Throttle Hero",
            "Line Perfect",
            "Tire Whisperer",
            "Fearless"
        ]

    def extract_driver_features(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        logger.info("Extracting driver features from telemetry")

        features = {}

        if 'brake_total' in df.columns:
            brake_data = df[df['brake_total'] > 10]
            if not brake_data.empty:
                features['avg_brake_pressure'] = brake_data['brake_total'].mean()
                features['max_brake_pressure'] = brake_data['brake_total'].max()
                features['brake_variance'] = brake_data['brake_total'].std()
            else:
                features['avg_brake_pressure'] = 0.0
                features['max_brake_pressure'] = 0.0
                features['brake_variance'] = 0.0

        if 'ath' in df.columns:
            throttle_data = df[df['ath'] > 5]
            if not throttle_data.empty:
                features['avg_throttle'] = throttle_data['ath'].mean()
                features['throttle_aggression'] = (throttle_data['ath'] > 80).sum() / len(throttle_data)

                throttle_changes = np.abs(np.diff(df['ath'].values))
                features['throttle_smoothness'] = 1.0 / (1.0 + throttle_changes.mean())
            else:
                features['avg_throttle'] = 0.0
                features['throttle_aggression'] = 0.0
                features['throttle_smoothness'] = 1.0

        if 'steering_angle' in df.columns:
            steering_data = df[df['steering_angle'].abs() > 5]
            if not steering_data.empty:
                features['avg_steering_angle'] = steering_data['steering_angle'].abs().mean()
                features['max_steering_angle'] = steering_data['steering_angle'].abs().max()

                steering_changes = np.abs(np.diff(df['steering_angle'].values))
                features['steering_smoothness'] = 1.0 / (1.0 + steering_changes.mean())
            else:
                features['avg_steering_angle'] = 0.0
                features['max_steering_angle'] = 0.0
                features['steering_smoothness'] = 1.0

        if 'speed' in df.columns:
            features['avg_speed'] = df['speed'].mean()
            features['max_speed'] = df['speed'].max()
            features['min_speed'] = df['speed'].min()
            features['speed_variance'] = df['speed'].std()

        if 'accy_can' in df.columns:
            features['avg_lateral_g'] = df['accy_can'].abs().mean()
            features['max_lateral_g'] = df['accy_can'].abs().max()

        if 'accx_can' in df.columns:
            features['avg_longitudinal_g'] = df['accx_can'].abs().mean()
            features['max_longitudinal_g'] = df['accx_can'].abs().max()

        if 'acc_magnitude' in df.columns:
            features['avg_combined_g'] = df['acc_magnitude'].mean()
            features['max_combined_g'] = df['acc_magnitude'].max()

        logger.info(f"Extracted {len(features)} features")
        return features

    def compute_talent_score(
        self,
        lap_time: float,
        all_lap_times: List[float],
        consistency_std: float
    ) -> Dict[str, float]:
        logger.info("Computing talent score")

        if not all_lap_times or len(all_lap_times) < 2:
            return {
                'raw_pace_percentile': 0.0,
                'consistency_score': 0.0,
                'talent_score': 0.0
            }

        sorted_times = sorted(all_lap_times)
        percentile = (sorted_times.index(min(sorted_times, key=lambda x: abs(x - lap_time))) + 1) / len(sorted_times)
        raw_pace_percentile = 1.0 - percentile

        max_std = np.std(all_lap_times)
        consistency_score = 1.0 - min(consistency_std / max_std, 1.0) if max_std > 0 else 1.0

        talent_score = (0.6 * raw_pace_percentile + 0.4 * consistency_score) * 100

        return {
            'raw_pace_percentile': raw_pace_percentile * 100,
            'consistency_score': consistency_score * 100,
            'talent_score': talent_score
        }

    def train_clustering_model(
        self,
        track_name: str,
        n_clusters: int = 6
    ) -> None:
        logger.info(f"Training DNA clustering model for {track_name}")

        track_dir = PROCESSED_DATA_DIR / track_name

        if not track_dir.exists():
            logger.error(f"Track directory not found: {track_dir}")
            return

        all_features = []
        driver_ids = []

        for parquet_file in track_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)

            for lap_num in df['lap_number'].unique():
                lap_df = df[df['lap_number'] == lap_num]

                if len(lap_df) < 10:
                    continue

                features = self.extract_driver_features(lap_df)

                if features:
                    all_features.append(features)
                    driver_ids.append(f"{parquet_file.stem}_lap{lap_num}")

        if not all_features:
            logger.error("No features extracted")
            return

        features_df = pd.DataFrame(all_features)
        features_df.fillna(0, inplace=True)

        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)

        models_dir = MODELS_DIR / track_name
        models_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = models_dir / "dna_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        kmeans_path = models_dir / "dna_kmeans.pkl"
        with open(kmeans_path, 'wb') as f:
            pickle.dump(self.kmeans, f)

        features_path = models_dir / "dna_feature_names.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(list(features_df.columns), f)

        logger.info(f"Saved DNA models to {models_dir}")
        logger.info(f"Trained on {len(all_features)} laps from {len(set([d.split('_lap')[0] for d in driver_ids]))} drivers")

    def load_clustering_model(self, track_name: str) -> bool:
        logger.info(f"Loading DNA clustering model for {track_name}")

        models_dir = MODELS_DIR / track_name

        scaler_path = models_dir / "dna_scaler.pkl"
        kmeans_path = models_dir / "dna_kmeans.pkl"

        if not scaler_path.exists() or not kmeans_path.exists():
            logger.error("DNA models not found")
            return False

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)

        logger.info("DNA models loaded successfully")
        return True

    def classify_driver(
        self,
        lap_df: pd.DataFrame,
        track_name: str
    ) -> Dict:
        logger.info("Classifying driver archetype")

        if not self.load_clustering_model(track_name):
            logger.error("Cannot classify without trained model")
            return {}

        features = self.extract_driver_features(lap_df)

        if not features:
            return {}

        features_df = pd.DataFrame([features])
        features_df.fillna(0, inplace=True)

        X = features_df.values
        X_scaled = self.scaler.transform(X)

        cluster_id = self.kmeans.predict(X_scaled)[0]
        archetype_name = self.archetype_names[cluster_id] if cluster_id < len(self.archetype_names) else f"Archetype {cluster_id}"

        return {
            'archetype_id': int(cluster_id),
            'archetype_name': archetype_name,
            'features': features
        }


def main():
    from config import TRD_TRACKS

    dna_engine = DriverDNA()

    for track in TRD_TRACKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training DNA model for: {track}")
        logger.info(f"{'='*50}\n")

        dna_engine.train_clustering_model(track, n_clusters=6)


if __name__ == "__main__":
    main()
