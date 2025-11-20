"""
Professional-grade vehicle dynamics model for racing analysis.

This module implements the physics-based vehicle dynamics calculations used in
professional racing telemetry systems (Bosch, AiM, Motec). It computes:
- Tire slip angles and slip ratios
- Load transfer (longitudinal and lateral)
- Tire force estimation using Pacejka Magic Formula
- Aerodynamic downforce effects
- Optimal cornering speed envelopes
- Traction circle analysis

These calculations enable physics-based root cause analysis of driver performance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VehicleParameters:
    """Physical parameters for GR Cup car (Toyota GR86 Cup Car)"""

    # Mass properties
    mass: float = 1240.0  # kg (GR86 Cup car mass)
    weight_dist_front: float = 0.53  # Front weight distribution

    # Dimensions
    wheelbase: float = 2.575  # meters
    track_front: float = 1.520  # meters
    track_rear: float = 1.540  # meters
    cg_height: float = 0.460  # meters (center of gravity height)

    # Aerodynamics
    frontal_area: float = 2.0  # m^2
    drag_coeff: float = 0.29  # Cd
    downforce_coeff_front: float = 0.15  # Clf
    downforce_coeff_rear: float = 0.20  # Clr

    # Tire properties (Michelin Pilot Sport Cup 2)
    tire_radius: float = 0.325  # meters

    # Pacejka Magic Formula coefficients (simplified)
    # These are approximations - real values would come from tire testing
    pacejka_b: float = 10.0  # Stiffness factor
    pacejka_c: float = 1.9   # Shape factor
    pacejka_d: float = 1.0   # Peak value
    pacejka_e: float = 0.97  # Curvature factor

    # Suspension
    roll_stiffness: float = 80000.0  # N/rad
    ride_height: float = 0.120  # meters


class VehicleDynamicsAnalyzer:
    """
    Analyzes vehicle dynamics from telemetry data using racing physics principles.

    This class implements the same calculations used in professional racing
    telemetry systems to understand vehicle behavior at the limits of grip.
    """

    def __init__(self, params: Optional[VehicleParameters] = None):
        self.params = params or VehicleParameters()

        # Pre-compute static properties
        self.mass_front = self.params.mass * self.params.weight_dist_front
        self.mass_rear = self.params.mass * (1 - self.params.weight_dist_front)

        # Gravity
        self.g = 9.81  # m/s^2

        # Air density at sea level
        self.rho_air = 1.225  # kg/m^3

    def compute_aerodynamic_forces(
        self,
        speed_mps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute aerodynamic drag and downforce.

        Args:
            speed_mps: Vehicle speed in m/s

        Returns:
            drag_force: Aerodynamic drag in Newtons
            downforce_front: Front axle downforce in Newtons
            downforce_rear: Rear axle downforce in Newtons
        """
        # Dynamic pressure
        q = 0.5 * self.rho_air * speed_mps**2

        # Drag force
        drag_force = q * self.params.frontal_area * self.params.drag_coeff

        # Downforce
        downforce_front = q * self.params.frontal_area * self.params.downforce_coeff_front
        downforce_rear = q * self.params.frontal_area * self.params.downforce_coeff_rear

        return drag_force, downforce_front, downforce_rear

    def compute_load_transfer(
        self,
        accel_long: np.ndarray,
        accel_lat: np.ndarray,
        speed_mps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute load transfer to each tire due to acceleration and aerodynamics.

        This is critical for understanding grip limits - a tire can only generate
        force proportional to its vertical load.

        Args:
            accel_long: Longitudinal acceleration in G's
            accel_lat: Lateral acceleration in G's
            speed_mps: Vehicle speed in m/s

        Returns:
            load_fl: Front-left tire vertical load (N)
            load_fr: Front-right tire vertical load (N)
            load_rl: Rear-left tire vertical load (N)
            load_rr: Rear-right tire vertical load (N)
        """
        # Convert acceleration to m/s^2
        accel_long_ms2 = accel_long * self.g
        accel_lat_ms2 = accel_lat * self.g

        # Aerodynamic forces
        _, downforce_front, downforce_rear = self.compute_aerodynamic_forces(speed_mps)

        # Static loads
        static_front = self.mass_front * self.g
        static_rear = self.mass_rear * self.g

        # Longitudinal load transfer (braking/acceleration)
        # Transfer = (mass * accel * cg_height) / wheelbase
        long_transfer = (self.params.mass * accel_long_ms2 * self.params.cg_height) / self.params.wheelbase

        # Front/rear loads with longitudinal transfer
        load_front_total = static_front - long_transfer + downforce_front
        load_rear_total = static_rear + long_transfer + downforce_rear

        # Lateral load transfer (cornering)
        # Transfer across track width
        lat_transfer_front = (self.mass_front * accel_lat_ms2 * self.params.cg_height) / self.params.track_front
        lat_transfer_rear = (self.mass_rear * accel_lat_ms2 * self.params.cg_height) / self.params.track_rear

        # Individual tire loads
        # Positive lateral accel = right turn, loads left tires
        load_fl = (load_front_total / 2) + lat_transfer_front
        load_fr = (load_front_total / 2) - lat_transfer_front
        load_rl = (load_rear_total / 2) + lat_transfer_rear
        load_rr = (load_rear_total / 2) - lat_transfer_rear

        # Ensure non-negative (tire can't pull)
        load_fl = np.maximum(load_fl, 0)
        load_fr = np.maximum(load_fr, 0)
        load_rl = np.maximum(load_rl, 0)
        load_rr = np.maximum(load_rr, 0)

        return load_fl, load_fr, load_rl, load_rr

    def compute_slip_angle(
        self,
        speed_mps: np.ndarray,
        lateral_accel_g: np.ndarray,
        is_front: bool = True
    ) -> np.ndarray:
        """
        Estimate tire slip angle from lateral acceleration.

        Slip angle is the angle between tire heading and actual travel direction.
        It's a critical parameter for understanding cornering performance.

        Args:
            speed_mps: Vehicle speed in m/s
            lateral_accel_g: Lateral acceleration in G's
            is_front: True for front axle, False for rear

        Returns:
            slip_angle: Slip angle in degrees
        """
        # Avoid division by zero
        speed_mps = np.maximum(speed_mps, 0.1)

        # Lateral velocity
        lateral_vel = lateral_accel_g * self.g * 1.0  # Simplified

        # Slip angle (small angle approximation for now)
        # alpha ≈ atan(v_lat / v_long)
        slip_angle_rad = np.arctan2(lateral_vel, speed_mps)

        # Apply understeer/oversteer gradient
        # Front tires typically have higher slip angles in understeer
        if is_front:
            slip_angle_rad *= 1.2

        return np.degrees(slip_angle_rad)

    def pacejka_tire_force(
        self,
        slip_angle_deg: np.ndarray,
        vertical_load_n: np.ndarray
    ) -> np.ndarray:
        """
        Compute lateral tire force using Pacejka Magic Formula.

        This is the industry-standard tire model used in professional racing.
        F(α) = D * sin(C * arctan(B*α - E*(B*α - arctan(B*α))))

        Args:
            slip_angle_deg: Tire slip angle in degrees
            vertical_load_n: Vertical load on tire in Newtons

        Returns:
            lateral_force: Lateral force in Newtons
        """
        # Normalize vertical load (peak force occurs around 1.0-1.2x static load)
        static_load = self.params.mass * self.g / 4  # Per tire
        load_factor = vertical_load_n / static_load

        # Magic Formula parameters
        B = self.params.pacejka_b / load_factor
        C = self.params.pacejka_c
        D = self.params.pacejka_d * vertical_load_n
        E = self.params.pacejka_e

        # Convert slip angle to radians
        alpha = np.radians(slip_angle_deg)

        # Magic Formula
        lateral_force = D * np.sin(
            C * np.arctan(
                B * alpha - E * (B * alpha - np.arctan(B * alpha))
            )
        )

        return lateral_force

    def compute_traction_circle_usage(
        self,
        accel_long_g: np.ndarray,
        accel_lat_g: np.ndarray
    ) -> np.ndarray:
        """
        Compute how much of the tire's traction circle is being used.

        The traction circle represents the maximum combined longitudinal and
        lateral grip available. Professional drivers maximize this usage.

        Args:
            accel_long_g: Longitudinal acceleration in G's
            accel_lat_g: Lateral acceleration in G's

        Returns:
            usage: Traction circle usage (0-1, values >1 indicate loss of grip)
        """
        # Maximum grip (typically ~1.4-1.6G for racing slicks)
        max_grip = 1.5

        # Combined acceleration magnitude
        combined_accel = np.sqrt(accel_long_g**2 + accel_lat_g**2)

        # Usage as fraction of maximum
        usage = combined_accel / max_grip

        return usage

    def analyze_telemetry_dynamics(
        self,
        telemetry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add comprehensive vehicle dynamics analysis to telemetry data.

        This computes all physics-based metrics needed for professional-grade
        performance analysis.

        Args:
            telemetry_df: DataFrame with columns:
                - speed: km/h
                - accx_can: longitudinal accel (G)
                - accy_can: lateral accel (G)

        Returns:
            Enhanced DataFrame with dynamics columns added
        """
        logger.info("Computing vehicle dynamics analysis")

        df = telemetry_df.copy()

        # Convert speed to m/s
        if 'speed' in df.columns:
            df['speed_mps'] = df['speed'] / 3.6
        else:
            logger.error("No speed column in telemetry")
            return df

        # Get acceleration data
        accel_long = df.get('accx_can', pd.Series(np.zeros(len(df))))
        accel_lat = df.get('accy_can', pd.Series(np.zeros(len(df))))
        speed_mps = df['speed_mps'].values

        # Aerodynamic forces
        drag, downforce_f, downforce_r = self.compute_aerodynamic_forces(speed_mps)
        df['aero_drag_n'] = drag
        df['aero_downforce_front_n'] = downforce_f
        df['aero_downforce_rear_n'] = downforce_r

        # Load transfer
        load_fl, load_fr, load_rl, load_rr = self.compute_load_transfer(
            accel_long.values,
            accel_lat.values,
            speed_mps
        )
        df['load_front_left_n'] = load_fl
        df['load_front_right_n'] = load_fr
        df['load_rear_left_n'] = load_rl
        df['load_rear_right_n'] = load_rr

        # Total axle loads
        df['load_front_total_n'] = load_fl + load_fr
        df['load_rear_total_n'] = load_rl + load_rr

        # Slip angles
        df['slip_angle_front_deg'] = self.compute_slip_angle(
            speed_mps,
            accel_lat.values,
            is_front=True
        )
        df['slip_angle_rear_deg'] = self.compute_slip_angle(
            speed_mps,
            accel_lat.values,
            is_front=False
        )

        # Traction circle usage
        df['traction_usage'] = self.compute_traction_circle_usage(
            accel_long.values,
            accel_lat.values
        )

        # Identify over-limit situations
        df['over_limit'] = df['traction_usage'] > 1.0

        # Understeer/oversteer detection
        # If front slip angle >> rear slip angle = understeer
        # If rear slip angle >> front slip angle = oversteer
        slip_diff = df['slip_angle_front_deg'] - df['slip_angle_rear_deg']
        df['handling_balance'] = np.select(
            [slip_diff > 2, slip_diff < -2],
            ['understeer', 'oversteer'],
            default='neutral'
        )

        logger.info("Vehicle dynamics analysis complete")
        return df

    def compute_optimal_corner_speed(
        self,
        corner_radius_m: float,
        banking_deg: float = 0.0
    ) -> float:
        """
        Compute theoretical optimal speed through a corner.

        This uses the physics of circular motion with banked turns.

        Args:
            corner_radius_m: Corner radius in meters
            banking_deg: Track banking angle in degrees

        Returns:
            optimal_speed_mps: Optimal speed in m/s
        """
        # Maximum lateral grip
        mu = 1.5  # Racing slick coefficient of friction

        # Banking angle in radians
        theta = np.radians(banking_deg)

        # Optimal speed from circular motion
        # v = sqrt(r * g * (mu + tan(theta)) / (1 - mu*tan(theta)))
        numerator = corner_radius_m * self.g * (mu + np.tan(theta))
        denominator = 1 - mu * np.tan(theta)

        if denominator <= 0:
            # Banking too steep or mu too high
            optimal_speed_mps = np.sqrt(corner_radius_m * self.g * mu)
        else:
            optimal_speed_mps = np.sqrt(numerator / denominator)

        return optimal_speed_mps

    def analyze_braking_performance(
        self,
        telemetry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detailed braking performance analysis.

        Args:
            telemetry_df: DataFrame with speed, brake pressure, acceleration

        Returns:
            DataFrame with braking analysis
        """
        logger.info("Analyzing braking performance")

        df = telemetry_df.copy()

        # Identify braking zones (negative longitudinal accel)
        if 'accx_can' in df.columns:
            df['is_braking'] = df['accx_can'] < -0.3  # Threshold for braking

            # Compute braking efficiency
            # Theoretical max decel with 1.5G grip limit
            max_decel_g = 1.5
            df['braking_efficiency'] = np.abs(df['accx_can']) / max_decel_g
            df['braking_efficiency'] = df['braking_efficiency'].clip(0, 1)

        # Brake pressure analysis
        if 'brake_front' in df.columns and 'brake_rear' in df.columns:
            df['brake_total'] = df['brake_front'] + df['brake_rear']
            df['brake_bias'] = df['brake_front'] / (df['brake_total'] + 1e-6)

        return df


def main():
    """Test vehicle dynamics calculations"""
    logger.info("Vehicle Dynamics Analyzer initialized")

    # Example: compute optimal speed for Turn 1 at Road America
    analyzer = VehicleDynamicsAnalyzer()

    # Turn 1 specs (example)
    corner_radius = 80  # meters
    banking = 0  # degrees

    optimal_speed = analyzer.compute_optimal_corner_speed(corner_radius, banking)
    logger.info(f"Optimal speed for {corner_radius}m radius corner: {optimal_speed:.1f} m/s ({optimal_speed * 3.6:.1f} km/h)")

    # Example traction circle
    accel_long = np.array([0.0, 0.5, 1.0, 0.0, -1.0])
    accel_lat = np.array([0.0, 0.0, 0.0, 1.2, 0.0])

    usage = analyzer.compute_traction_circle_usage(accel_long, accel_lat)
    logger.info(f"Traction usage: {usage}")


if __name__ == "__main__":
    main()
