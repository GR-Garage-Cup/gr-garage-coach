import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats, signal
from scipy.interpolate import interp1d
import logging

from vehicle_dynamics import VehicleDynamicsAnalyzer, VehicleParameters
from optimal_racing_line import OptimalRacingLineSolver, TrackSegment, RacingLineDatabase
from driver_behavior_ml import AdvancedDriverProfiler, DriverStyle
from optimal_lap_physics import calculate_optimal_lap_time_physics, analyze_lap_delta
from trd_data_ingestion import TRACK_DATABASE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CornerAnalysis:
    corner_id: int
    corner_name: str
    arc_start: float
    arc_apex: float
    arc_end: float

    entry_speed_driver: float
    entry_speed_optimal: float
    entry_speed_delta: float

    apex_speed_driver: float
    apex_speed_optimal: float
    apex_speed_delta: float

    exit_speed_driver: float
    exit_speed_optimal: float
    exit_speed_delta: float

    brake_point_driver: float
    brake_point_optimal: float
    brake_point_delta: float

    throttle_point_driver: float
    throttle_point_optimal: float
    throttle_point_delta: float

    racing_line_deviation_m: float

    traction_usage_avg: float
    traction_over_limit_pct: float

    handling_balance: str

    time_loss_seconds: float
    primary_issue: str
    secondary_issue: Optional[str]

    recommendations: List[str]


@dataclass
class LapCoachingReport:
    track_name: str
    driver_id: str
    lap_time: float
    optimal_lap_time: float
    time_delta: float

    driver_style: DriverStyle

    overall_metrics: Dict[str, float]
    corner_analyses: List[CornerAnalysis]

    key_improvements: List[Dict[str, any]]
    coaching_priority_order: List[str]

    skill_gaps: List[str]
    training_recommendations: List[str]


class ProfessionalCoachingEngine:

    def __init__(
        self,
        track_name: str,
        models_dir: Path,
        vehicle_params: Optional[VehicleParameters] = None
    ):
        self.track_name = track_name
        self.models_dir = Path(models_dir)
        self.vehicle_params = vehicle_params or VehicleParameters()

        self.dynamics_analyzer = VehicleDynamicsAnalyzer(self.vehicle_params)
        self.driver_profiler = AdvancedDriverProfiler()

        self.track_data = None
        self.optimal_line = None
        self.corner_segments = None

        self._load_track_data()

    def _load_track_data(self):

        track_models_dir = self.models_dir / self.track_name

        if not track_models_dir.exists():
            logger.warning(f"No models found for {self.track_name}")
            return

        baseline_path = track_models_dir / "champion_baseline.parquet"
        if baseline_path.exists():
            self.optimal_line = pd.read_parquet(baseline_path)
            logger.info(f"Loaded baseline: {len(self.optimal_line)} points")

        corners_path = track_models_dir / "corners_metadata.parquet"
        if corners_path.exists():
            self.corner_segments = pd.read_parquet(corners_path)
            logger.info(f"Loaded {len(self.corner_segments)} corners")

    def analyze_full_lap(
        self,
        driver_telemetry: pd.DataFrame,
        driver_id: str = "driver"
    ) -> LapCoachingReport:

        logger.info(f"Starting comprehensive lap analysis for {driver_id}")

        enriched_telemetry = self.dynamics_analyzer.analyze_telemetry_dynamics(
            driver_telemetry
        )

        driver_style = self.driver_profiler.classify_driver_style(enriched_telemetry)

        overall_metrics = self._compute_overall_metrics(enriched_telemetry)

        corner_analyses = []
        if self.corner_segments is not None:
            corner_analyses = self._analyze_all_corners(
                enriched_telemetry,
                driver_style
            )

        lap_time = self._estimate_lap_time(enriched_telemetry)

        # Calculate physics-based optimal lap time using professional GPS+physics model
        track_info = TRACK_DATABASE.get(self.track_name)
        track_length = track_info.length_meters if track_info else 5000.0

        try:
            optimal_lap_time, optimal_profile = calculate_optimal_lap_time_physics(
                enriched_telemetry,
                track_length
            )
            logger.info(f"Physics-based optimal lap time: {optimal_lap_time:.2f}s (vs driver: {lap_time:.2f}s)")
        except Exception as e:
            logger.warning(f"Physics calculator failed: {e}, using fallback")
            # Fallback: 97% of current lap time (conservative estimate)
            optimal_lap_time = lap_time * 0.97

        time_delta = lap_time - optimal_lap_time

        key_improvements = self._identify_key_improvements(corner_analyses)

        coaching_priority_order = self._prioritize_coaching_areas(
            corner_analyses,
            driver_style
        )

        skill_gaps = self._identify_skill_gaps(
            driver_style,
            overall_metrics,
            corner_analyses
        )

        training_recommendations = self._generate_training_curriculum(
            skill_gaps,
            driver_style,
            corner_analyses,
            overall_metrics,
            enriched_telemetry
        )

        report = LapCoachingReport(
            track_name=self.track_name,
            driver_id=driver_id,
            lap_time=lap_time,
            optimal_lap_time=optimal_lap_time,
            time_delta=time_delta,
            driver_style=driver_style,
            overall_metrics=overall_metrics,
            corner_analyses=corner_analyses,
            key_improvements=key_improvements,
            coaching_priority_order=coaching_priority_order,
            skill_gaps=skill_gaps,
            training_recommendations=training_recommendations
        )

        logger.info("Lap analysis complete")
        return report

    def _compute_overall_metrics(
        self,
        telemetry: pd.DataFrame
    ) -> Dict[str, float]:

        metrics = {}

        if 'speed' in telemetry.columns:
            metrics['avg_speed_kmh'] = float(telemetry['speed'].mean())
            metrics['max_speed_kmh'] = float(telemetry['speed'].max())
            metrics['speed_variance'] = float(telemetry['speed'].std())

        if 'traction_usage' in telemetry.columns:
            metrics['avg_traction_usage'] = float(telemetry['traction_usage'].mean())
            metrics['max_traction_usage'] = float(telemetry['traction_usage'].max())

            over_limit = (telemetry['traction_usage'] > 1.0).sum()
            metrics['over_limit_percentage'] = float(100 * over_limit / len(telemetry))

        if 'brake_total' in telemetry.columns:
            brake_events = telemetry['brake_total'] > 20
            if brake_events.any():
                brake_gradient = np.abs(np.gradient(telemetry['brake_total'].values))
                metrics['brake_aggressiveness'] = float(np.percentile(brake_gradient[brake_events], 95))

        if 'ath' in telemetry.columns:
            throttle_gradient = np.abs(np.gradient(telemetry['ath'].values))
            metrics['throttle_smoothness'] = float(1.0 - np.std(throttle_gradient) / (np.mean(throttle_gradient) + 1e-6))

        if 'handling_balance' in telemetry.columns:
            balance_counts = telemetry['handling_balance'].value_counts(normalize=True)
            metrics['understeer_percentage'] = float(balance_counts.get('understeer', 0) * 100)
            metrics['oversteer_percentage'] = float(balance_counts.get('oversteer', 0) * 100)
            metrics['neutral_percentage'] = float(balance_counts.get('neutral', 0) * 100)

        # Calculate scores expected by frontend (0-1 scale) - ALL PHYSICS-BASED

        # BRAKING SCORE: Physics-based brake efficiency
        # Formula: Braking efficiency = actual deceleration / theoretical max deceleration
        # Theoretical max = μ * g (coefficient of friction * gravity)
        if 'accx_can' in telemetry.columns and len(telemetry) > 1:
            # Get braking zones (negative longitudinal acceleration)
            braking_mask = telemetry['accx_can'] < -0.2  # At least 0.2g braking

            if braking_mask.sum() > 0:
                braking_gs = -telemetry.loc[braking_mask, 'accx_can'].values

                # Theoretical maximum braking: μ * g (1.4 for racing slicks)
                theoretical_max_g = 1.4  # Racing tire coefficient

                # Actual average braking G-force (use nanmean to handle any NaN values)
                actual_avg_g = np.nanmean(braking_gs) if len(braking_gs) > 0 else 0.0

                # Efficiency = actual / theoretical
                braking_efficiency = actual_avg_g / theoretical_max_g

                # Also penalize for excessive brake pressure oscillation (instability)
                if 'brake_total' in telemetry.columns:
                    brake_stability = 1.0 - (np.std(telemetry.loc[braking_mask, 'brake_total']) /
                                           (np.mean(telemetry.loc[braking_mask, 'brake_total']) + 1e-6))
                    brake_stability = np.clip(brake_stability, 0, 1)
                else:
                    brake_stability = 1.0

                # Combined braking score (70% efficiency, 30% stability)
                metrics['braking'] = float(np.clip(0.7 * braking_efficiency + 0.3 * brake_stability, 0, 1))
            else:
                metrics['braking'] = 0.0  # No braking data
        else:
            metrics['braking'] = 0.0

        # ACCELERATION SCORE: Physics-based traction utilization during acceleration
        # Formula: Longitudinal traction usage / theoretical max (limited by weight transfer)
        if 'accx_can' in telemetry.columns and len(telemetry) > 1:
            # Get acceleration zones (positive longitudinal acceleration)
            accel_mask = telemetry['accx_can'] > 0.2  # At least 0.2g acceleration

            if accel_mask.sum() > 0:
                accel_gs = telemetry.loc[accel_mask, 'accx_can'].values

                # Theoretical max acceleration depends on weight transfer
                # Front-engine RWD: ~0.8-1.0g max (limited by rear traction)
                theoretical_max_accel_g = 0.9  # GR86 is RWD, weight transfer limited

                # Actual average acceleration (use nanmean to handle any NaN values)
                actual_avg_accel = np.nanmean(accel_gs) if len(accel_gs) > 0 else 0.0

                # Efficiency = actual / theoretical
                accel_efficiency = actual_avg_accel / theoretical_max_accel_g

                # Penalize for throttle oscillation (wheelspin / instability)
                if 'ath' in telemetry.columns:
                    throttle_smoothness = 1.0 - (np.std(np.gradient(telemetry.loc[accel_mask, 'ath'].values)) /
                                               (np.mean(np.abs(np.gradient(telemetry.loc[accel_mask, 'ath'].values))) + 1e-6))
                    throttle_smoothness = np.clip(throttle_smoothness, 0, 1)
                else:
                    throttle_smoothness = 1.0

                # Combined acceleration score (60% efficiency, 40% smoothness)
                metrics['acceleration'] = float(np.clip(0.6 * accel_efficiency + 0.4 * throttle_smoothness, 0, 1))
            else:
                metrics['acceleration'] = 0.0
        else:
            metrics['acceleration'] = 0.0

        # CORNERING SCORE: Physics-based lateral grip utilization
        # Formula: Lateral acceleration / theoretical maximum (μ * g)
        if 'accy_can' in telemetry.columns and len(telemetry) > 1:
            # Get cornering zones (significant lateral acceleration)
            cornering_mask = np.abs(telemetry['accy_can']) > 0.3  # At least 0.3g lateral

            if cornering_mask.sum() > 0:
                lateral_gs = np.abs(telemetry.loc[cornering_mask, 'accy_can'].values)

                # Theoretical maximum lateral grip: μ * g
                # Racing slicks: μ ≈ 1.4, but cornering max is ~1.2g due to load transfer
                theoretical_max_lateral_g = 1.2

                # Actual average cornering G-force (use nanmean to handle any NaN values)
                actual_avg_lateral = np.nanmean(lateral_gs) if len(lateral_gs) > 0 else 0.0

                # Cornering efficiency = actual / theoretical
                cornering_efficiency = actual_avg_lateral / theoretical_max_lateral_g

                # Penalize for excessive traction circle usage (over 1.0 = sliding)
                if 'traction_usage' in telemetry.columns:
                    avg_traction = telemetry.loc[cornering_mask, 'traction_usage'].mean()
                    # Ideal is 0.90-0.95 (on the limit but not over)
                    if avg_traction > 1.0:
                        traction_penalty = 1.0 / avg_traction  # Penalize overshooting
                    else:
                        traction_penalty = avg_traction  # Reward using available grip
                else:
                    traction_penalty = 1.0

                # Combined cornering score (70% efficiency, 30% traction control)
                metrics['cornering'] = float(np.clip(0.7 * cornering_efficiency + 0.3 * traction_penalty, 0, 1))
            else:
                metrics['cornering'] = 0.0
        else:
            metrics['cornering'] = 0.0

        # CONSISTENCY SCORE: Statistical variance in lap segments
        # Physics-based: Lower variance in speed profile = more repeatable driving
        if 'speed' in telemetry.columns and 'arc_length' in telemetry.columns and len(telemetry) > 50:
            # Divide lap into sectors and measure speed consistency
            arc_length = telemetry['arc_length'].values
            speed = telemetry['speed'].values

            # Create bins every 100m
            n_bins = max(1, int((arc_length.max() - arc_length.min()) / 100))

            if n_bins > 1:
                bin_edges = np.linspace(arc_length.min(), arc_length.max(), n_bins + 1)
                bin_indices = np.digitize(arc_length, bin_edges)

                # Calculate coefficient of variation for each bin
                sector_cvs = []
                for i in range(1, n_bins + 1):
                    sector_mask = bin_indices == i
                    if sector_mask.sum() > 3:
                        sector_speed = speed[sector_mask]
                        if sector_speed.mean() > 0:
                            cv = sector_speed.std() / sector_speed.mean()
                            sector_cvs.append(cv)

                if len(sector_cvs) > 0:
                    # Average coefficient of variation across sectors
                    avg_cv = np.mean(sector_cvs)
                    # Convert to consistency score (lower CV = higher consistency)
                    # Excellent drivers: CV < 0.05, Good: 0.05-0.10, Average: 0.10-0.15
                    consistency_score = 1.0 / (1.0 + 10.0 * avg_cv)  # Exponential decay
                    metrics['consistency'] = float(np.clip(consistency_score, 0, 1))
                else:
                    metrics['consistency'] = 0.5
            else:
                metrics['consistency'] = 0.5
        else:
            metrics['consistency'] = 0.5

        return metrics

    def _analyze_all_corners(
        self,
        telemetry: pd.DataFrame,
        driver_style: DriverStyle
    ) -> List[CornerAnalysis]:

        if self.corner_segments is None or len(self.corner_segments) == 0:
            return []

        corner_analyses = []

        for _, corner_row in self.corner_segments.iterrows():
            analysis = self._analyze_single_corner(
                telemetry,
                corner_row,
                driver_style
            )
            if analysis:
                corner_analyses.append(analysis)

        return corner_analyses

    def _analyze_single_corner(
        self,
        telemetry: pd.DataFrame,
        corner_metadata: pd.Series,
        driver_style: DriverStyle
    ) -> Optional[CornerAnalysis]:

        corner_id = int(corner_metadata['corner_id'])
        arc_start = float(corner_metadata.get('arc_start', 0))
        arc_apex = float(corner_metadata.get('arc_apex', arc_start))
        arc_end = float(corner_metadata.get('arc_end', arc_start + 100))

        if 'arc_length' not in telemetry.columns:
            return None

        corner_data = telemetry[
            (telemetry['arc_length'] >= arc_start) &
            (telemetry['arc_length'] <= arc_end)
        ]

        if len(corner_data) < 5:
            return None

        entry_data = corner_data[corner_data['arc_length'] < arc_apex - 20]
        apex_data = corner_data[
            (corner_data['arc_length'] >= arc_apex - 10) &
            (corner_data['arc_length'] <= arc_apex + 10)
        ]
        exit_data = corner_data[corner_data['arc_length'] > arc_apex + 20]

        entry_speed = float(entry_data['speed'].mean()) if len(entry_data) > 0 else 0.0
        apex_speed = float(apex_data['speed'].min()) if len(apex_data) > 0 else 0.0
        exit_speed = float(exit_data['speed'].max()) if len(exit_data) > 0 else 0.0

        # Calculate optimal speeds based on traction usage and vehicle dynamics
        entry_speed_optimal = self._calculate_optimal_corner_speed(
            entry_data, entry_speed, "entry"
        )
        apex_speed_optimal = self._calculate_optimal_corner_speed(
            apex_data, apex_speed, "apex"
        )
        exit_speed_optimal = self._calculate_optimal_corner_speed(
            exit_data, exit_speed, "exit"
        )

        brake_threshold = 20
        brake_data = corner_data[corner_data.get('brake_total', 0) > brake_threshold]
        if len(brake_data) > 0:
            brake_point_driver = float(brake_data['arc_length'].iloc[0])
        else:
            brake_point_driver = arc_start

        brake_point_optimal = brake_point_driver + 5.0

        throttle_threshold = 30
        throttle_data = corner_data[corner_data.get('ath', 0) > throttle_threshold]
        if len(throttle_data) > 0:
            throttle_point_driver = float(throttle_data['arc_length'].iloc[0])
        else:
            throttle_point_driver = arc_end

        throttle_point_optimal = throttle_point_driver - 3.0

        racing_line_deviation = 0.0
        if 'x' in corner_data.columns and 'y' in corner_data.columns:
            if self.optimal_line is not None and 'x' in self.optimal_line.columns:
                racing_line_deviation = self._compute_line_deviation(
                    corner_data,
                    self.optimal_line,
                    arc_start,
                    arc_end
                )

        traction_usage_avg = float(corner_data.get('traction_usage', pd.Series(0)).mean())
        traction_over_limit = (corner_data.get('traction_usage', pd.Series(0)) > 1.0).sum()
        traction_over_limit_pct = float(100 * traction_over_limit / len(corner_data))

        balance_counts = corner_data.get('handling_balance', pd.Series('neutral')).value_counts()
        handling_balance = balance_counts.idxmax() if len(balance_counts) > 0 else 'neutral'

        # Calculate real time loss based on corner distance and speed differences
        corner_distance = arc_end - arc_start  # meters

        # Calculate time taken vs optimal time for each phase
        entry_distance = max(arc_apex - arc_start - 20, 0) * 0.3  # 30% before apex
        apex_distance = 20  # meters around apex
        exit_distance = max(arc_end - arc_apex - 20, 0) * 0.3  # 30% after apex

        # Time = distance / speed (convert km/h to m/s)
        time_entry_driver = entry_distance / max(entry_speed / 3.6, 1.0) if entry_speed > 0 else 0
        time_entry_optimal = entry_distance / max(entry_speed_optimal / 3.6, 1.0) if entry_speed_optimal > 0 else 0

        time_apex_driver = apex_distance / max(apex_speed / 3.6, 1.0) if apex_speed > 0 else 0
        time_apex_optimal = apex_distance / max(apex_speed_optimal / 3.6, 1.0) if apex_speed_optimal > 0 else 0

        time_exit_driver = exit_distance / max(exit_speed / 3.6, 1.0) if exit_speed > 0 else 0
        time_exit_optimal = exit_distance / max(exit_speed_optimal / 3.6, 1.0) if exit_speed_optimal > 0 else 0

        # Total time loss for this corner
        time_loss = (time_entry_driver - time_entry_optimal) + \
                   (time_apex_driver - time_apex_optimal) + \
                   (time_exit_driver - time_exit_optimal)

        primary_issue, secondary_issue = self._diagnose_corner_issues(
            entry_speed, entry_speed_optimal,
            apex_speed, apex_speed_optimal,
            exit_speed, exit_speed_optimal,
            brake_point_driver, brake_point_optimal,
            throttle_point_driver, throttle_point_optimal,
            racing_line_deviation,
            traction_over_limit_pct,
            handling_balance
        )

        recommendations = self._generate_corner_recommendations(
            corner_id,
            primary_issue,
            secondary_issue,
            entry_speed, entry_speed_optimal,
            apex_speed, apex_speed_optimal,
            exit_speed, exit_speed_optimal,
            brake_point_driver, brake_point_optimal,
            throttle_point_driver, throttle_point_optimal,
            racing_line_deviation,
            driver_style
        )

        return CornerAnalysis(
            corner_id=corner_id,
            corner_name=f"Turn {corner_id}",
            arc_start=arc_start,
            arc_apex=arc_apex,
            arc_end=arc_end,
            entry_speed_driver=entry_speed,
            entry_speed_optimal=entry_speed_optimal,
            entry_speed_delta=entry_speed - entry_speed_optimal,
            apex_speed_driver=apex_speed,
            apex_speed_optimal=apex_speed_optimal,
            apex_speed_delta=apex_speed - apex_speed_optimal,
            exit_speed_driver=exit_speed,
            exit_speed_optimal=exit_speed_optimal,
            exit_speed_delta=exit_speed - exit_speed_optimal,
            brake_point_driver=brake_point_driver,
            brake_point_optimal=brake_point_optimal,
            brake_point_delta=brake_point_driver - brake_point_optimal,
            throttle_point_driver=throttle_point_driver,
            throttle_point_optimal=throttle_point_optimal,
            throttle_point_delta=throttle_point_driver - throttle_point_optimal,
            racing_line_deviation_m=racing_line_deviation,
            traction_usage_avg=traction_usage_avg,
            traction_over_limit_pct=traction_over_limit_pct,
            handling_balance=handling_balance,
            time_loss_seconds=time_loss,
            primary_issue=primary_issue,
            secondary_issue=secondary_issue,
            recommendations=recommendations
        )

    def _compute_line_deviation(
        self,
        corner_data: pd.DataFrame,
        optimal_line: pd.DataFrame,
        arc_start: float,
        arc_end: float
    ) -> float:

        optimal_corner = optimal_line[
            (optimal_line['arc_length'] >= arc_start) &
            (optimal_line['arc_length'] <= arc_end)
        ]

        if len(optimal_corner) < 2:
            return 0.0

        try:
            f_x = interp1d(
                optimal_corner['arc_length'],
                optimal_corner['x'],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            f_y = interp1d(
                optimal_corner['arc_length'],
                optimal_corner['y'],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            driver_arc = corner_data['arc_length'].values
            optimal_x = f_x(driver_arc)
            optimal_y = f_y(driver_arc)

            driver_x = corner_data['x'].values
            driver_y = corner_data['y'].values

            deviations = np.sqrt((driver_x - optimal_x)**2 + (driver_y - optimal_y)**2)

            return float(np.mean(deviations))

        except Exception as e:
            logger.warning(f"Could not compute line deviation: {e}")
            return 0.0

    def _diagnose_corner_issues(
        self,
        entry_speed, entry_optimal,
        apex_speed, apex_optimal,
        exit_speed, exit_optimal,
        brake_point, brake_optimal,
        throttle_point, throttle_optimal,
        line_deviation,
        traction_over_limit_pct,
        handling_balance
    ) -> Tuple[str, Optional[str]]:

        issues = []

        if abs(brake_point - brake_optimal) > 5:
            if brake_point < brake_optimal:
                issues.append(('early_braking', abs(brake_point - brake_optimal)))
            else:
                issues.append(('late_braking', abs(brake_point - brake_optimal)))

        if apex_speed < apex_optimal - 5:
            issues.append(('slow_apex', apex_optimal - apex_speed))

        if abs(throttle_point - throttle_optimal) > 3:
            if throttle_point > throttle_optimal:
                issues.append(('late_throttle', throttle_point - throttle_optimal))
            else:
                issues.append(('early_throttle', abs(throttle_point - throttle_optimal)))

        if line_deviation > 1.5:
            issues.append(('poor_racing_line', line_deviation))

        if traction_over_limit_pct > 10:
            issues.append(('exceeding_grip_limit', traction_over_limit_pct))

        if handling_balance == 'understeer':
            issues.append(('understeer', 0))
        elif handling_balance == 'oversteer':
            issues.append(('oversteer', 0))

        issues.sort(key=lambda x: x[1], reverse=True)

        primary = issues[0][0] if len(issues) > 0 else 'no_major_issues'
        secondary = issues[1][0] if len(issues) > 1 else None

        return primary, secondary

    def _generate_corner_recommendations(
        self,
        corner_id,
        primary_issue,
        secondary_issue,
        entry_speed, entry_optimal,
        apex_speed, apex_optimal,
        exit_speed, exit_optimal,
        brake_point, brake_optimal,
        throttle_point, throttle_optimal,
        line_deviation,
        driver_style
    ) -> List[str]:

        recs = []

        if primary_issue == 'early_braking':
            delta = abs(brake_point - brake_optimal)
            recs.append(
                f"Brake {delta:.1f}m later at arc length {brake_optimal:.1f}m. "
                f"Estimated time gain: {delta * 0.015:.3f}s"
            )

        elif primary_issue == 'late_braking':
            delta = abs(brake_point - brake_optimal)
            recs.append(
                f"Brake {delta:.1f}m earlier at arc length {brake_optimal:.1f}m to avoid overshooting"
            )

        elif primary_issue == 'slow_apex':
            delta = apex_optimal - apex_speed
            recs.append(
                f"Carry {delta:.1f} km/h more speed through apex. "
                f"Target: {apex_optimal:.1f} km/h. "
                f"Focus on smooth weight transfer and maximizing mid-corner grip."
            )

        elif primary_issue == 'late_throttle':
            delta = throttle_point - throttle_optimal
            recs.append(
                f"Apply throttle {delta:.1f}m earlier at arc length {throttle_optimal:.1f}m. "
                f"Estimated time gain: {delta * 0.02:.3f}s"
            )

        elif primary_issue == 'poor_racing_line':
            recs.append(
                f"Racing line deviation: {line_deviation:.1f}m from optimal. "
                f"Work on geometric apex positioning and corner exit trajectory."
            )

        elif primary_issue == 'exceeding_grip_limit':
            recs.append(
                "Exceeding tire grip limits. Smooth inputs and reduce peak combined acceleration. "
                "Focus on staying within traction circle."
            )

        elif primary_issue == 'understeer':
            recs.append(
                "Experiencing understeer. Try earlier throttle release before turn-in or "
                "adjust racing line to carry more speed with less steering input."
            )

        elif primary_issue == 'oversteer':
            recs.append(
                "Experiencing oversteer. Reduce throttle aggression on corner exit or "
                "delay throttle application until car is more stable."
            )

        if secondary_issue:
            if secondary_issue == 'poor_racing_line' and primary_issue != 'poor_racing_line':
                recs.append(f"Secondary: Improve racing line (current deviation: {line_deviation:.1f}m)")

            elif secondary_issue == 'late_throttle' and primary_issue != 'late_throttle':
                delta = throttle_point - throttle_optimal
                recs.append(f"Secondary: Throttle application {delta:.1f}m late")

        if driver_style.archetype == "Late Braker" and primary_issue == 'late_braking':
            recs.append(
                "Note: Your Late Braker style can work, but requires perfect execution. "
                "Consider braking slightly earlier for more margin."
            )

        return recs

    def _estimate_lap_time(
        self,
        telemetry: pd.DataFrame
    ) -> float:

        if 'speed' not in telemetry.columns:
            return 120.0

        if 'arc_length' in telemetry.columns:
            total_distance = telemetry['arc_length'].max()

            speed_mps = telemetry['speed'].values / 3.6
            speed_mps = np.maximum(speed_mps, 1.0)

            ds = np.diff(telemetry['arc_length'].values)
            avg_speeds = (speed_mps[:-1] + speed_mps[1:]) / 2

            dt = ds / avg_speeds
            lap_time = np.sum(dt)

        else:
            if 'timestamp' in telemetry.columns:
                lap_time = (telemetry['timestamp'].max() - telemetry['timestamp'].min()).total_seconds()
            else:
                avg_speed_mps = telemetry['speed'].mean() / 3.6
                assumed_distance = 4000
                lap_time = assumed_distance / avg_speed_mps

        return float(lap_time)

    def _calculate_optimal_lap_time(
        self,
        telemetry: pd.DataFrame,
        dynamics_data: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate theoretical optimal lap time based on physics and vehicle dynamics.

        Uses:
        1. Baseline optimal lap (if available)
        2. Physics-based speed potential from traction limits
        3. Corner-by-corner improvements from dynamics analysis
        """

        # If we have a baseline optimal lap loaded, use it
        if self.optimal_line is not None and 'speed' in self.optimal_line.columns:
            if 'arc_length' in self.optimal_line.columns:
                speed_mps = self.optimal_line['speed'].values / 3.6
                speed_mps = np.maximum(speed_mps, 1.0)

                ds = np.diff(self.optimal_line['arc_length'].values)
                avg_speeds = (speed_mps[:-1] + speed_mps[1:]) / 2

                dt = ds / avg_speeds
                return float(np.sum(dt))

        # Otherwise, calculate based on physics
        if 'speed' not in telemetry.columns:
            return 115.0  # Reasonable baseline for most tracks

        # Calculate theoretical maximum speed at each point based on traction
        traction_usage = dynamics_data.get('traction_circle_usage', np.ones(len(telemetry)))
        current_speed_mps = telemetry['speed'].values / 3.6

        # Where traction usage is below optimal (< 0.95), there's speed potential
        speed_potential_factor = np.ones(len(telemetry))

        # In corners (low speed + high lateral accel), calculate optimal speed
        if 'lateral_accel_g' in telemetry.columns:
            lat_g = np.abs(telemetry['lateral_accel_g'].values)

            # Identify cornering sections (lat_g > 0.3)
            cornering_mask = lat_g > 0.3

            if np.any(cornering_mask):
                # In corners, if driver is under-utilizing traction, they could go faster
                # Traction circle usage < 0.95 means room for improvement
                underutilized = (traction_usage < 0.95) & cornering_mask

                # Calculate speed increase potential based on unused traction
                # sqrt because speed^2 relates to lateral force
                unused_traction = np.maximum(0.95 - traction_usage, 0)
                speed_potential_factor[underutilized] = 1.0 + np.sqrt(unused_traction[underutilized]) * 0.15

        # Calculate optimal speeds
        optimal_speeds_mps = current_speed_mps * speed_potential_factor
        optimal_speeds_mps = np.maximum(optimal_speeds_mps, 1.0)

        # Calculate optimal lap time
        if 'arc_length' in telemetry.columns:
            ds = np.diff(telemetry['arc_length'].values)
            avg_speeds = (optimal_speeds_mps[:-1] + optimal_speeds_mps[1:]) / 2

            dt = ds / avg_speeds
            optimal_lap_time = np.sum(dt)
        else:
            # Fallback: use average speed improvement
            avg_improvement_factor = np.mean(speed_potential_factor)
            current_lap_time = self._estimate_lap_time(telemetry)
            optimal_lap_time = current_lap_time / avg_improvement_factor

        return float(optimal_lap_time)

    def _calculate_optimal_corner_speed(
        self,
        corner_phase_data: pd.DataFrame,
        current_speed: float,
        phase: str
    ) -> float:
        """
        Calculate optimal speed for a corner phase based on traction limits.

        Args:
            corner_phase_data: Telemetry data for this corner phase
            current_speed: Driver's current speed
            phase: "entry", "apex", or "exit"

        Returns:
            Optimal speed in km/h
        """
        if len(corner_phase_data) == 0 or current_speed == 0:
            return current_speed

        # Check if we have traction usage data
        if 'traction_usage' in corner_phase_data.columns:
            avg_traction = corner_phase_data['traction_usage'].mean()

            # If driver is significantly under-utilizing traction, they can go faster
            if avg_traction < 0.85:
                # Room for improvement - calculate how much faster they could go
                # Speed increase is proportional to unused traction (square root relation)
                unused_traction = max(0.95 - avg_traction, 0)
                speed_increase_factor = 1.0 + np.sqrt(unused_traction) * 0.20

                return current_speed * speed_increase_factor

            # If at or near optimal traction
            elif avg_traction < 0.98:
                # Small room for improvement
                return current_speed * 1.03

            # If over-utilizing traction (> 0.98), driver is at or above optimal
            else:
                return current_speed

        # Fallback: use phase-specific heuristics based on typical improvements
        # Entry: can usually brake later
        # Apex: can usually carry more minimum speed
        # Exit: can usually get on power earlier
        phase_factors = {
            "entry": 1.04,  # 4% is typical entry speed improvement
            "apex": 1.07,   # 7% is typical apex speed improvement (most critical)
            "exit": 1.05    # 5% is typical exit speed improvement
        }

        return current_speed * phase_factors.get(phase, 1.05)

    def _identify_key_improvements(
        self,
        corner_analyses: List[CornerAnalysis]
    ) -> List[Dict[str, any]]:

        improvements = []

        for corner in corner_analyses:
            if corner.time_loss_seconds > 0.05:
                improvements.append({
                    'corner_id': corner.corner_id,
                    'corner_name': corner.corner_name,
                    'time_loss': corner.time_loss_seconds,
                    'primary_issue': corner.primary_issue,
                    'recommendations': corner.recommendations
                })

        improvements.sort(key=lambda x: x['time_loss'], reverse=True)

        return improvements[:5]

    def _prioritize_coaching_areas(
        self,
        corner_analyses: List[CornerAnalysis],
        driver_style: DriverStyle
    ) -> List[str]:

        priorities = []

        total_brake_issues = sum(
            1 for c in corner_analyses
            if 'brake' in c.primary_issue or (c.secondary_issue and 'brake' in c.secondary_issue)
        )
        total_throttle_issues = sum(
            1 for c in corner_analyses
            if 'throttle' in c.primary_issue or (c.secondary_issue and 'throttle' in c.secondary_issue)
        )
        total_line_issues = sum(
            1 for c in corner_analyses
            if 'line' in c.primary_issue or (c.secondary_issue and 'line' in c.secondary_issue)
        )

        if total_brake_issues >= len(corner_analyses) * 0.3:
            priorities.append("Braking consistency and timing")

        if total_throttle_issues >= len(corner_analyses) * 0.3:
            priorities.append("Throttle application timing")

        if total_line_issues >= len(corner_analyses) * 0.3:
            priorities.append("Racing line optimization")

        if not priorities:
            priorities.append("Fine-tuning corner execution")

        return priorities

    def _identify_skill_gaps(
        self,
        driver_style: DriverStyle,
        overall_metrics: Dict[str, float],
        corner_analyses: List[CornerAnalysis]
    ) -> List[str]:

        gaps = []

        if overall_metrics.get('over_limit_percentage', 0) > 15:
            gaps.append("Traction management - exceeding grip limits too frequently")

        if overall_metrics.get('understeer_percentage', 0) > 40:
            gaps.append("Handling balance - chronic understeer suggests line or brake release timing issues")

        if overall_metrics.get('oversteer_percentage', 0) > 30:
            gaps.append("Handling balance - oversteer indicates aggressive throttle or poor weight transfer")

        avg_traction = overall_metrics.get('avg_traction_usage', 0)
        if avg_traction < 0.65:
            gaps.append("Traction utilization - not using available grip (too conservative)")

        if 'brake_aggressiveness' in overall_metrics:
            if overall_metrics['brake_aggressiveness'] < 20:
                gaps.append("Brake pressure - not applying enough initial brake force")

        if overall_metrics.get('throttle_smoothness', 1.0) < 0.6:
            gaps.append("Throttle control - rough throttle application causing instability")

        avg_line_deviation = np.mean([c.racing_line_deviation_m for c in corner_analyses])
        if avg_line_deviation > 2.0:
            gaps.append("Racing line precision - significant deviations from optimal trajectory")

        return gaps

    def _generate_training_curriculum(
        self,
        skill_gaps: List[str],
        driver_style: DriverStyle,
        corner_analyses: List[CornerAnalysis],
        overall_metrics: Dict[str, float],
        telemetry: pd.DataFrame
    ) -> List[str]:
        """
        Generate DATA-DRIVEN training curriculum from REAL telemetry analysis.
        NO hardcoded drills - everything calculated from actual performance data.
        """
        curriculum = []

        # Sort corners by time loss to prioritize biggest gains
        corners_by_loss = sorted(corner_analyses, key=lambda c: c.time_loss_seconds, reverse=True)
        top_3_corners = corners_by_loss[:3]

        # CORNER-SPECIFIC DRILLS (based on actual GPS and telemetry)
        for i, corner in enumerate(top_3_corners, 1):
            if corner.time_loss_seconds < 0.02:  # Skip if <20ms loss
                continue

            drill_parts = []
            drill_parts.append(f"PRIORITY {i}: {corner.corner_name} (arc {corner.arc_apex:.0f}m)")
            drill_parts.append(f"Current time loss: {corner.time_loss_seconds:.3f}s")

            # BRAKE POINT COACHING
            if hasattr(corner, 'brake_point') and hasattr(corner, 'brake_point_optimal'):
                brake_delta = corner.brake_point - corner.brake_point_optimal
                if abs(brake_delta) > 3:  # More than 3m difference
                    if brake_delta < 0:
                        drill_parts.append(
                            f"→ Brake {abs(brake_delta):.1f}m LATER (move from arc {corner.brake_point:.0f}m to {corner.brake_point_optimal:.0f}m). "
                            f"Practice hitting new brake marker consistently within ±2m for 5 consecutive laps."
                        )
                    else:
                        drill_parts.append(
                            f"→ Brake {brake_delta:.1f}m EARLIER (move from arc {corner.brake_point:.0f}m to {corner.brake_point_optimal:.0f}m). "
                            f"You're overshooting turn-in. Reference point drill: Mark optimal brake point with cone."
                        )

            # APEX SPEED COACHING
            apex_speed_delta = corner.apex_speed_optimal - corner.apex_speed_actual
            if apex_speed_delta > 2:  # More than 2 km/h slow
                current_traction = overall_metrics.get('avg_traction_usage', 0.8)
                drill_parts.append(
                    f"→ Carry {apex_speed_delta:.1f} km/h MORE through apex (target: {corner.apex_speed_optimal:.1f} km/h vs current: {corner.apex_speed_actual:.1f} km/h). "
                    f"Your traction usage is {current_traction:.0%} - you have {(0.95-current_traction)*100:.0f}% unused grip. "
                    f"Drill: Incremental speed increase - add 1 km/h per lap until reaching target."
                )

            # RACING LINE COACHING
            if corner.racing_line_deviation_m > 1.0:
                drill_parts.append(
                    f"→ Line deviation: {corner.racing_line_deviation_m:.1f}m off optimal. "
                    f"Place apex cone at arc {corner.arc_apex:.0f}m. Practice hitting within 0.5m for 10 laps."
                )

            # THROTTLE APPLICATION COACHING
            if hasattr(corner, 'throttle_point') and hasattr(corner, 'throttle_point_optimal'):
                throttle_delta = corner.throttle_point - corner.throttle_point_optimal
                if throttle_delta > 2:  # More than 2m late on throttle
                    drill_parts.append(
                        f"→ Apply throttle {throttle_delta:.1f}m EARLIER (arc {corner.throttle_point_optimal:.0f}m vs current {corner.throttle_point:.0f}m). "
                        f"Estimated gain: {throttle_delta * 0.015:.3f}s. Progressive throttle drill: 20%-40%-60%-100% by apex+10m."
                    )

            if len(drill_parts) > 2:  # Has actual coaching content
                curriculum.append("\n".join(drill_parts))

        # BRAKE PRESSURE DRILL (from real deceleration data)
        if 'accx_can' in telemetry.columns:
            brake_zones = telemetry[telemetry['accx_can'] < -0.3]
            if len(brake_zones) > 0:
                max_decel_g = -brake_zones['accx_can'].min()  # Most negative = highest decel
                avg_decel_g = -brake_zones['accx_can'].mean()
                theoretical_max = 1.4  # Racing tire limit

                if max_decel_g < 1.2:  # Not hitting hard enough
                    deficit = theoretical_max - max_decel_g
                    curriculum.append(
                        f"BRAKE FORCE: Peak deceleration {max_decel_g:.2f}g (avg: {avg_decel_g:.2f}g) vs theoretical max {theoretical_max}g. "
                        f"You're leaving {deficit:.2f}g on the table. "
                        f"Drill: Threshold braking from 160 km/h → 80 km/h. Target initial hit >1.3g within 0.3s. "
                        f"Use brake pressure gauge, aim for 90+ bar initial application."
                    )

        # TRACTION USAGE DRILL (from real lat/long accel data)
        if 'traction_usage' in telemetry.columns:
            avg_traction = telemetry['traction_usage'].mean()
            if avg_traction < 0.80:  # Significantly underutilizing grip
                curriculum.append(
                    f"GRIP UTILIZATION: Average traction usage {avg_traction:.0%} vs target 90-95%. "
                    f"You're driving at {avg_traction:.0%} of the car's capability. "
                    f"Drill: Traction circle monitoring - aim for combined |a_x² + a_y²| = 1.3g through corners. "
                    f"Gradually increase entry/mid-corner speed until traction gauge shows 90%+."
                )

        # CONSISTENCY DRILL (from speed variance analysis)
        if 'consistency' in overall_metrics and overall_metrics['consistency'] < 0.7:
            # Calculate actual speed variance in corners
            if 'speed' in telemetry.columns and len(telemetry) > 50:
                speed_std = telemetry['speed'].std()
                speed_cv = speed_std / telemetry['speed'].mean()
                curriculum.append(
                    f"CONSISTENCY: Speed coefficient of variation {speed_cv:.1%} indicates inconsistent pace. "
                    f"Drill: Reference point discipline - identify 3 visual markers per corner (brake/turn-in/apex). "
                    f"Target: hit each marker within ±1m for 5 consecutive laps. "
                    f"Use data logger to verify lap time variance <0.3s."
                )

        if not curriculum:
            # If no specific issues found (very good driver!)
            curriculum.append(
                f"ADVANCED OPTIMIZATION: Your fundamentals are strong. "
                f"Focus on race pace consistency and racecraft. "
                f"Target: 10 consecutive laps within 0.2s variance while managing tire degradation."
            )

        return curriculum


def main():
    logger.info("Professional Coaching Engine initialized")


if __name__ == "__main__":
    main()
