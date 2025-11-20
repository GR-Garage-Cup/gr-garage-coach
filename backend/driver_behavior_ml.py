import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

    class DummyModule:
        pass

    class Dataset:
        pass

    class Module(DummyModule):
        pass

    nn = type('nn', (), {'Module': Module})()
    DataLoader = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriverStyle:
    archetype: str
    confidence: float
    characteristics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]


class TelemetrySequenceDataset(Dataset):

    def __init__(
        self,
        telemetry_sequences: List[np.ndarray],
        labels: Optional[List[int]] = None,
        sequence_length: int = 100
    ):
        self.sequences = telemetry_sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if len(seq) > self.sequence_length:
            start_idx = np.random.randint(0, len(seq) - self.sequence_length)
            seq = seq[start_idx:start_idx + self.sequence_length]
        elif len(seq) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(seq), seq.shape[1]))
            seq = np.vstack([seq, padding])

        seq_tensor = torch.FloatTensor(seq)

        if self.labels is not None:
            return seq_tensor, self.labels[idx]
        return seq_tensor


class DriverBehaviorLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super(DriverBehaviorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        pooled = torch.mean(attn_out, dim=1)

        output = self.fc_layers(pooled)

        return output


class DriverBehaviorEncoder(nn.Module):

    def __init__(
        self,
        input_size: int,
        latent_dim: int = 32,
        hidden_size: int = 128
    ):
        super(DriverBehaviorEncoder, self).__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)

        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=1)

        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)

        return mu, logvar


class AdvancedDriverProfiler:

    ARCHETYPES = {
        0: "Smooth Operator",
        1: "Late Braker",
        2: "Trail Braker",
        3: "Point and Shoot",
        4: "Aggressive Defender",
        5: "Conservative Steady"
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = 'cpu'
    ):
        if HAS_TORCH:
            self.device = torch.device(device)
            self.model = None
            self.encoder = None

            if model_path and model_path.exists():
                self.load_model(model_path)
        else:
            self.device = None
            self.model = None
            self.encoder = None
            logger.warning("PyTorch not available, will use heuristic classification only")

    def extract_behavioral_features(
        self,
        telemetry_df: pd.DataFrame,
        window_size: int = 50
    ) -> np.ndarray:

        features = []

        required_cols = [
            'speed', 'brake_total', 'ath', 'steering_angle',
            'accx_can', 'accy_can'
        ]

        for col in required_cols:
            if col not in telemetry_df.columns:
                telemetry_df[col] = 0.0

        speed = telemetry_df['speed'].values
        brake = telemetry_df['brake_total'].values if 'brake_total' in telemetry_df.columns else telemetry_df.get('brake_front', pd.Series(0)).values
        throttle = telemetry_df['ath'].values
        steering = telemetry_df['steering_angle'].values
        accel_x = telemetry_df['accx_can'].values
        accel_y = telemetry_df['accy_can'].values

        brake_gradient = np.gradient(brake)
        throttle_gradient = np.gradient(throttle)
        steering_gradient = np.gradient(steering)
        speed_gradient = np.gradient(speed)

        brake_smoothness = self._compute_rolling_std(brake_gradient, window_size)
        throttle_smoothness = self._compute_rolling_std(throttle_gradient, window_size)
        steering_smoothness = self._compute_rolling_std(steering_gradient, window_size)

        combined_accel = np.sqrt(accel_x**2 + accel_y**2)
        traction_usage = combined_accel / 1.5

        feature_matrix = np.column_stack([
            speed / 200.0,
            brake / 100.0,
            throttle / 100.0,
            steering / 900.0,
            accel_x / 2.0,
            accel_y / 2.0,
            brake_gradient / 50.0,
            throttle_gradient / 50.0,
            steering_gradient / 100.0,
            speed_gradient / 10.0,
            brake_smoothness,
            throttle_smoothness,
            steering_smoothness,
            traction_usage
        ])

        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        return feature_matrix

    def _compute_rolling_std(
        self,
        data: np.ndarray,
        window_size: int
    ) -> np.ndarray:

        result = np.zeros_like(data)

        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2)
            result[i] = np.std(data[start:end])

        return result

    def train_classifier(
        self,
        training_sequences: List[np.ndarray],
        training_labels: List[int],
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):

        logger.info("Training driver behavior classifier")

        input_size = training_sequences[0].shape[1]

        self.model = DriverBehaviorLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            num_classes=len(self.ARCHETYPES),
            dropout=0.3
        ).to(self.device)

        dataset = TelemetrySequenceDataset(
            training_sequences,
            training_labels,
            sequence_length=100
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = torch.LongTensor(labels).to(self.device)

                optimizer.zero_grad()

                outputs = self.model(sequences)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total

            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        logger.info("Training complete")

    def classify_driver_style(
        self,
        telemetry_df: pd.DataFrame
    ) -> DriverStyle:

        if self.model is None:
            logger.warning("No model loaded, using heuristic classification")
            return self._heuristic_classification(telemetry_df)

        self.model.eval()

        features = self.extract_behavioral_features(telemetry_df)

        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(sequence)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        archetype = self.ARCHETYPES[predicted_class]

        characteristics = self._extract_driving_characteristics(telemetry_df)
        strengths, weaknesses = self._identify_strengths_weaknesses(
            predicted_class,
            characteristics
        )

        return DriverStyle(
            archetype=archetype,
            confidence=confidence,
            characteristics=characteristics,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _heuristic_classification(
        self,
        telemetry_df: pd.DataFrame
    ) -> DriverStyle:

        characteristics = self._extract_driving_characteristics(telemetry_df)

        scores = np.zeros(len(self.ARCHETYPES))

        if characteristics['brake_aggressiveness'] > 0.7:
            scores[1] += 2.0
        if characteristics['brake_smoothness'] > 0.8:
            scores[0] += 2.0
        if characteristics['throttle_aggressiveness'] > 0.7:
            scores[3] += 1.5
        if characteristics['trail_braking_index'] > 0.6:
            scores[2] += 2.0
        if characteristics['steering_smoothness'] > 0.8:
            scores[0] += 1.0
        if characteristics['consistency_score'] > 0.85:
            scores[5] += 1.5
        if characteristics['traction_usage'] > 0.9:
            scores[4] += 1.0

        predicted_class = np.argmax(scores)
        archetype = self.ARCHETYPES[predicted_class]
        confidence = scores[predicted_class] / (np.sum(scores) + 1e-6)

        strengths, weaknesses = self._identify_strengths_weaknesses(
            predicted_class,
            characteristics
        )

        return DriverStyle(
            archetype=archetype,
            confidence=float(confidence),
            characteristics=characteristics,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _extract_driving_characteristics(
        self,
        telemetry_df: pd.DataFrame
    ) -> Dict[str, float]:

        n = len(telemetry_df)

        if 'brake_total' in telemetry_df.columns:
            brake = telemetry_df['brake_total'].values
        elif 'brake_front' in telemetry_df.columns:
            brake = telemetry_df['brake_front'].values
        else:
            brake = np.zeros(n)

        throttle = telemetry_df['throttle'].values if 'throttle' in telemetry_df.columns else np.zeros(n)
        steering = telemetry_df['steering_angle'].values if 'steering_angle' in telemetry_df.columns else np.zeros(n)
        speed = telemetry_df['speed'].values if 'speed' in telemetry_df.columns else np.ones(n) * 100
        accel_x = telemetry_df['accel_x'].values if 'accel_x' in telemetry_df.columns else np.zeros(n)
        accel_y = telemetry_df['accel_y'].values if 'accel_y' in telemetry_df.columns else np.zeros(n)

        if len(brake) < 2:
            brake = np.zeros(max(n, 2))
        if len(throttle) < 2:
            throttle = np.zeros(max(n, 2))
        if len(steering) < 2:
            steering = np.zeros(max(n, 2))

        brake_gradient = np.abs(np.gradient(brake))
        throttle_gradient = np.abs(np.gradient(throttle))
        steering_gradient = np.abs(np.gradient(steering))

        brake_aggressiveness = np.percentile(brake_gradient, 95) / 50.0
        throttle_aggressiveness = np.percentile(throttle_gradient, 95) / 50.0

        brake_smoothness = 1.0 - (np.std(brake_gradient) / (np.mean(brake_gradient) + 1e-6))
        throttle_smoothness = 1.0 - (np.std(throttle_gradient) / (np.mean(throttle_gradient) + 1e-6))
        steering_smoothness = 1.0 - (np.std(steering_gradient) / (np.mean(steering_gradient) + 1e-6))

        brake_on = brake > 20
        throttle_on = throttle > 20
        trail_braking = brake_on & throttle_on
        trail_braking_index = np.sum(trail_braking) / (np.sum(brake_on) + 1)

        combined_accel = np.sqrt(accel_x**2 + accel_y**2)
        traction_usage = np.mean(combined_accel) / 1.5

        speed_variance = np.std(speed) / (np.mean(speed) + 1)
        consistency_score = 1.0 - min(speed_variance, 1.0)

        return {
            'brake_aggressiveness': float(np.clip(brake_aggressiveness, 0, 1)),
            'throttle_aggressiveness': float(np.clip(throttle_aggressiveness, 0, 1)),
            'brake_smoothness': float(np.clip(brake_smoothness, 0, 1)),
            'throttle_smoothness': float(np.clip(throttle_smoothness, 0, 1)),
            'steering_smoothness': float(np.clip(steering_smoothness, 0, 1)),
            'trail_braking_index': float(np.clip(trail_braking_index, 0, 1)),
            'traction_usage': float(np.clip(traction_usage, 0, 1)),
            'consistency_score': float(np.clip(consistency_score, 0, 1))
        }

    def _identify_strengths_weaknesses(
        self,
        archetype_class: int,
        characteristics: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:

        archetype = self.ARCHETYPES[archetype_class]
        strengths = []
        weaknesses = []

        if archetype == "Smooth Operator":
            if characteristics['steering_smoothness'] > 0.8:
                strengths.append("Excellent steering smoothness")
            if characteristics['consistency_score'] > 0.85:
                strengths.append("High consistency")
            if characteristics['brake_aggressiveness'] < 0.5:
                weaknesses.append("Could brake harder for better lap times")

        elif archetype == "Late Braker":
            if characteristics['brake_aggressiveness'] > 0.7:
                strengths.append("Strong late braking ability")
            if characteristics['trail_braking_index'] < 0.3:
                weaknesses.append("Limited trail braking usage")
            if characteristics['throttle_smoothness'] < 0.6:
                weaknesses.append("Throttle application could be smoother")

        elif archetype == "Trail Braker":
            if characteristics['trail_braking_index'] > 0.6:
                strengths.append("Advanced trail braking technique")
            if characteristics['brake_smoothness'] > 0.7:
                strengths.append("Good brake modulation")
            if characteristics['consistency_score'] < 0.7:
                weaknesses.append("Consistency needs improvement")

        elif archetype == "Point and Shoot":
            if characteristics['throttle_aggressiveness'] > 0.7:
                strengths.append("Aggressive throttle application")
            if characteristics['brake_aggressiveness'] > 0.6:
                strengths.append("Decisive braking")
            if characteristics['steering_smoothness'] < 0.6:
                weaknesses.append("Steering inputs could be smoother")

        elif archetype == "Aggressive Defender":
            if characteristics['traction_usage'] > 0.9:
                strengths.append("High traction circle usage")
            if characteristics['consistency_score'] < 0.7:
                weaknesses.append("Over-aggressive at times, reducing consistency")

        elif archetype == "Conservative Steady":
            if characteristics['consistency_score'] > 0.85:
                strengths.append("Excellent consistency")
            if characteristics['traction_usage'] < 0.7:
                weaknesses.append("Not using full grip potential")
            if characteristics['brake_aggressiveness'] < 0.5:
                weaknesses.append("Too conservative on brakes")

        return strengths, weaknesses

    def save_model(self, path: Path):
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'archetypes': self.ARCHETYPES
            }, path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model = DriverBehaviorLSTM(
            input_size=14,
            hidden_size=128,
            num_layers=3,
            num_classes=len(self.ARCHETYPES)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from {path}")


def main():
    logger.info("Driver Behavior ML Module initialized")

    profiler = AdvancedDriverProfiler()

    sample_data = pd.DataFrame({
        'speed': np.random.uniform(50, 200, 1000),
        'brake_total': np.random.uniform(0, 100, 1000),
        'ath': np.random.uniform(0, 100, 1000),
        'steering_angle': np.random.uniform(-200, 200, 1000),
        'accx_can': np.random.uniform(-1.5, 1.5, 1000),
        'accy_can': np.random.uniform(-1.5, 1.5, 1000)
    })

    driver_style = profiler.classify_driver_style(sample_data)

    logger.info(f"Archetype: {driver_style.archetype}")
    logger.info(f"Confidence: {driver_style.confidence:.2f}")
    logger.info(f"Strengths: {driver_style.strengths}")
    logger.info(f"Weaknesses: {driver_style.weaknesses}")


if __name__ == "__main__":
    main()
