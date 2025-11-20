"""
Professional-Grade Optimal Lap Time Calculator for GR Cup Racing
Based on physics models used by motorsport engineering teams.

This implements the fundamental equations of vehicle dynamics:
- Curvature-speed relationship (friction circle)
- Weight transfer effects on grip
- Brake/acceleration modeling with load transfer
- Minimum time integration using velocity profile optimization

Author: Physics-based racing optimization
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from dataclasses import dataclass


@dataclass
class VehicleParams:
    """GR86 Cup Car physical parameters"""
    mass: float = 1270.0  # kg (with driver)
    cg_height: float = 0.46  # m (center of gravity height)
    wheelbase: float = 2.57  # m
    track_front: float = 1.52  # m
    track_rear: float = 1.54  # m
    weight_dist_front: float = 0.53  # 53% front weight distribution

    # Tire and grip parameters
    tire_mu: float = 1.35  # Racing slicks on warm track (peak)
    tire_mu_longitudinal: float = 1.40  # Slightly higher for braking

    # Performance limits
    max_brake_decel: float = 1.4  # g (limited by tire grip + aero)
    max_accel: float = 0.9  # g (RWD, traction limited)

    # Aerodynamics (minimal for GR86)
    drag_coeff_area: float = 0.65  # Cd*A in m²
    downforce_coeff_area: float = 0.15  # Cl*A in m² (small wing)

    # Engine/drivetrain
    power_hp: float = 230.0  # hp (Cup spec)
    max_rpm: float = 7400.0

    @property
    def power_watts(self) -> float:
        return self.power_hp * 745.7  # Convert hp to watts

    @property
    def weight_n(self) -> float:
        return self.mass * 9.81


def calculate_curvature(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Calculate track curvature κ(s) = |x'y'' - y'x''| / (x'² + y'²)^(3/2)

    Uses central differences for numerical derivatives with smoothing.

    Args:
        x: X coordinates (meters)
        y: Y coordinates (meters)
        s: Arc length (meters)

    Returns:
        curvature: κ in 1/meters (positive = radius of curvature)
    """
    # Smooth coordinates to reduce GPS noise
    window = min(51, len(x) // 10)
    if window % 2 == 0:
        window += 1
    if window < 5:
        window = 5

    x_smooth = savgol_filter(x, window, 3)
    y_smooth = savgol_filter(y, window, 3)

    # First derivatives (dx/ds, dy/ds)
    dx = np.gradient(x_smooth, s)
    dy = np.gradient(y_smooth, s)

    # Second derivatives (d²x/ds², d²y/ds²)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # Curvature formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)

    curvature = numerator / denominator

    # Limit minimum radius to something reasonable (e.g., 5m)
    max_curvature = 1.0 / 5.0
    curvature = np.minimum(curvature, max_curvature)

    return curvature


def calculate_max_lateral_accel(speed_mps: float, vehicle: VehicleParams) -> float:
    """
    Calculate maximum lateral acceleration including weight transfer and aero.

    For steady-state cornering, the limiting factor is the friction circle
    modified by weight transfer and aerodynamic downforce.

    Args:
        speed_mps: Vehicle speed in m/s
        vehicle: Vehicle parameters

    Returns:
        max_lat_accel: Maximum lateral acceleration in m/s²
    """
    g = 9.81

    # Aerodynamic downforce (increases with speed²)
    aero_downforce = 0.5 * 1.225 * vehicle.downforce_coeff_area * speed_mps**2

    # Total normal force
    total_normal_force = vehicle.weight_n + aero_downforce

    # Weight transfer reduces peak lateral grip (simplified model)
    # Under pure lateral load, inside tires unload
    weight_transfer_factor = 0.92  # Accounts for ~8% grip loss from weight transfer

    # Maximum lateral force
    max_lat_force = vehicle.tire_mu * total_normal_force * weight_transfer_factor

    # Lateral acceleration
    max_lat_accel = max_lat_force / vehicle.mass

    return max_lat_accel


def calculate_corner_speed(curvature: float, vehicle: VehicleParams,
                          min_speed: float = 15.0) -> float:
    """
    Calculate maximum speed through corner based on curvature and grip.

    v_max = sqrt(a_lat_max / κ)

    Args:
        curvature: Track curvature (1/radius) in 1/m
        vehicle: Vehicle parameters
        min_speed: Minimum corner speed (m/s) - safety margin

    Returns:
        max_speed: Maximum corner speed in m/s
    """
    if curvature < 1e-6:  # Straight
        return 999.0  # Effectively unlimited (will be limited by power)

    # Iterative solution since aero depends on speed
    speed_guess = 30.0  # m/s initial guess

    for _ in range(5):  # Converges quickly
        max_lat_accel = calculate_max_lateral_accel(speed_guess, vehicle)
        speed_new = np.sqrt(max_lat_accel / curvature)
        speed_guess = speed_new

    return max(speed_guess, min_speed)


def calculate_brake_decel(speed_mps: float, vehicle: VehicleParams) -> float:
    """
    Calculate maximum braking deceleration with weight transfer.

    Under braking, weight transfers to front wheels, increasing front grip
    and decreasing rear grip. This model accounts for this.

    Args:
        speed_mps: Current speed in m/s
        vehicle: Vehicle parameters

    Returns:
        brake_decel: Maximum braking deceleration in m/s²
    """
    g = 9.81

    # Aerodynamic downforce and drag
    aero_downforce = 0.5 * 1.225 * vehicle.downforce_coeff_area * speed_mps**2
    aero_drag_force = 0.5 * 1.225 * vehicle.drag_coeff_area * speed_mps**2

    # Total normal force
    total_normal_force = vehicle.weight_n + aero_downforce

    # Weight transfer under braking increases front axle load
    # For maximum braking, weight transfer enhances front grip
    # This is a simplified model - full model requires iterative solution
    weight_transfer_factor = 1.05  # Front axle benefits from weight transfer

    # Maximum braking force (limited by front tire grip)
    max_brake_force = vehicle.tire_mu_longitudinal * total_normal_force * weight_transfer_factor

    # Add aero drag contribution
    max_brake_force += aero_drag_force

    # Brake deceleration
    brake_decel = max_brake_force / vehicle.mass

    # Limit to specified maximum
    return min(brake_decel, vehicle.max_brake_decel * g)


def calculate_accel_force(speed_mps: float, vehicle: VehicleParams) -> float:
    """
    Calculate maximum acceleration force (traction or power limited).

    At low speeds: traction limited (RWD weight transfer)
    At high speeds: power limited

    Args:
        speed_mps: Current speed in m/s
        vehicle: Vehicle parameters

    Returns:
        accel_force: Maximum acceleration force in N
    """
    g = 9.81

    # Power-limited acceleration
    if speed_mps < 1.0:
        speed_mps = 1.0  # Avoid division by zero

    power_limited_force = vehicle.power_watts / speed_mps

    # Traction-limited acceleration (RWD, rear weight bias under accel)
    aero_downforce = 0.5 * 1.225 * vehicle.downforce_coeff_area * speed_mps**2
    total_normal_force = vehicle.weight_n + aero_downforce

    # Under acceleration, weight transfers to rear (benefits RWD)
    rear_weight_fraction = 1.0 - vehicle.weight_dist_front
    weight_transfer_benefit = 1.15  # Rear axle gains ~15% load under accel

    rear_normal_force = total_normal_force * rear_weight_fraction * weight_transfer_benefit
    traction_limited_force = vehicle.tire_mu * rear_normal_force

    # Take minimum (limiting factor)
    max_accel_force = min(power_limited_force, traction_limited_force)

    # Also limit by specified max
    max_accel_force = min(max_accel_force, vehicle.max_accel * g * vehicle.mass)

    return max_accel_force


def forward_integrate_braking(s: np.ndarray, v_max_corner: np.ndarray,
                               vehicle: VehicleParams) -> np.ndarray:
    """
    Forward integration: Calculate required braking from high speed to corner entry.

    Starting from the end (high speed section), propagate braking requirements
    backward through the track.

    Args:
        s: Arc length array (meters)
        v_max_corner: Maximum corner speeds (m/s)
        vehicle: Vehicle parameters

    Returns:
        v_brake: Speed profile considering braking (m/s)
    """
    n = len(s)
    v_brake = np.copy(v_max_corner)

    # Iterate backward through track
    for i in range(n - 2, -1, -1):
        ds = s[i + 1] - s[i]
        if ds <= 0:
            continue

        # Speed at next point
        v_next = v_brake[i + 1]

        # Maximum brake deceleration at next point
        brake_decel = calculate_brake_decel(v_next, vehicle)

        # Using v² = v₀² + 2as, solve for v₀
        # v_next² = v_current² - 2 * brake_decel * ds
        # v_current = sqrt(v_next² + 2 * brake_decel * ds)

        v_brake_required = np.sqrt(v_next**2 + 2 * brake_decel * ds)

        # Take minimum (can't exceed corner limit)
        v_brake[i] = min(v_brake[i], v_brake_required)

    return v_brake


def backward_integrate_acceleration(s: np.ndarray, v_brake: np.ndarray,
                                    vehicle: VehicleParams) -> np.ndarray:
    """
    Backward integration: Calculate acceleration from corner exit.

    Starting from beginning, propagate acceleration forward through track.

    Args:
        s: Arc length array (meters)
        v_brake: Speed profile from braking pass (m/s)
        vehicle: Vehicle parameters

    Returns:
        v_optimal: Optimal speed profile (m/s)
    """
    n = len(s)
    v_optimal = np.copy(v_brake)

    # Iterate forward through track
    for i in range(1, n):
        ds = s[i] - s[i - 1]
        if ds <= 0:
            continue

        # Speed at previous point
        v_prev = v_optimal[i - 1]

        # Maximum acceleration force at previous point
        accel_force = calculate_accel_force(v_prev, vehicle)
        accel = accel_force / vehicle.mass

        # Aero drag at previous speed
        drag_force = 0.5 * 1.225 * vehicle.drag_coeff_area * v_prev**2
        drag_decel = drag_force / vehicle.mass

        # Net acceleration
        net_accel = accel - drag_decel

        # Using v² = v₀² + 2as
        v_accel_possible = np.sqrt(v_prev**2 + 2 * net_accel * ds)

        # Take minimum (can't exceed brake-limited speed)
        v_optimal[i] = min(v_optimal[i], v_accel_possible)

    return v_optimal


def calculate_throttle_brake(v_actual: np.ndarray, v_optimal: np.ndarray,
                             s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate throttle and brake inputs based on speed delta.

    Args:
        v_actual: Current speed profile (m/s)
        v_optimal: Optimal speed profile (m/s)
        s: Arc length (meters)

    Returns:
        throttle: Throttle percentage (0-100)
        brake: Brake percentage (0-100)
    """
    n = len(v_actual)
    throttle = np.zeros(n)
    brake = np.zeros(n)

    for i in range(1, n):
        ds = s[i] - s[i - 1]
        if ds <= 0:
            continue

        # Required acceleration
        dv = v_optimal[i] - v_actual[i - 1]
        dt = ds / max(v_actual[i - 1], 1.0)
        required_accel = dv / dt

        if required_accel > 0.5:  # Accelerating
            throttle[i] = min(100.0, (required_accel / 0.9 / 9.81) * 100)
            brake[i] = 0.0
        elif required_accel < -0.5:  # Braking
            brake[i] = min(100.0, (abs(required_accel) / 1.4 / 9.81) * 100)
            throttle[i] = 0.0
        else:  # Coasting or maintenance
            throttle[i] = 10.0  # Maintenance throttle
            brake[i] = 0.0

    return throttle, brake


def calculate_optimal_lap_time_physics(
    telemetry: pd.DataFrame,
    track_length_m: float,
    vehicle: VehicleParams = None
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate theoretical optimal lap time using professional racing physics.

    This implements the quasi-steady-state lap time simulation method used by
    motorsport engineers, accounting for:
    - Track curvature and corner speed limits
    - Weight transfer effects on grip
    - Aerodynamic forces (downforce and drag)
    - Power and traction limits
    - Optimal brake points and acceleration zones

    Algorithm:
    1. Calculate track curvature from GPS coordinates
    2. Determine maximum corner speeds (grip-limited)
    3. Forward integrate braking zones
    4. Backward integrate acceleration zones
    5. Integrate lap time from velocity profile

    Args:
        telemetry: DataFrame with columns:
            - arc_length: Distance along track (meters)
            - x, y: GPS coordinates (meters)
            - speed: Current speed (m/s) - for reference only
            - accx_can, accy_can: Current accelerations - for reference
        track_length_m: Total track length in meters
        vehicle: Vehicle parameters (uses GR86 Cup defaults if None)

    Returns:
        optimal_lap_time: Theoretical best lap time in seconds
        optimal_profile: DataFrame with:
            - arc_length: Distance (m)
            - curvature: Track curvature (1/m)
            - v_max_corner: Max corner speed (m/s)
            - v_optimal: Optimal speed (m/s)
            - throttle: Throttle % (0-100)
            - brake: Brake % (0-100)
            - lap_time: Cumulative time (s)
    """
    if vehicle is None:
        vehicle = VehicleParams()

    # Extract data
    s = telemetry['arc_length'].values
    x = telemetry['x'].values
    y = telemetry['y'].values
    v_actual = telemetry['speed'].values

    # Ensure monotonic arc length
    s = np.maximum.accumulate(s)

    # Step 1: Calculate track curvature
    curvature = calculate_curvature(x, y, s)

    # Step 2: Calculate maximum corner speeds (grip-limited)
    v_max_corner = np.array([
        calculate_corner_speed(k, vehicle) for k in curvature
    ])

    # Step 3: Forward integrate - braking zones
    v_brake = forward_integrate_braking(s, v_max_corner, vehicle)

    # Step 4: Backward integrate - acceleration zones
    v_optimal = backward_integrate_acceleration(s, v_brake, vehicle)

    # Step 5: Calculate throttle and brake
    throttle, brake = calculate_throttle_brake(v_actual, v_optimal, s)

    # Step 6: Integrate lap time
    lap_time_cumulative = np.zeros(len(s))
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        v_avg = (v_optimal[i] + v_optimal[i - 1]) / 2
        v_avg = max(v_avg, 1.0)  # Avoid division by zero

        dt = ds / v_avg
        lap_time_cumulative[i] = lap_time_cumulative[i - 1] + dt

    optimal_lap_time = lap_time_cumulative[-1]

    # Create output DataFrame
    optimal_profile = pd.DataFrame({
        'arc_length': s,
        'curvature': curvature,
        'radius_m': 1.0 / np.maximum(curvature, 1e-6),
        'v_max_corner': v_max_corner,
        'v_optimal': v_optimal,
        'v_actual': v_actual,
        'throttle_pct': throttle,
        'brake_pct': brake,
        'lap_time': lap_time_cumulative,
        'time_delta': lap_time_cumulative[-1] - lap_time_cumulative,
    })

    return optimal_lap_time, optimal_profile


def analyze_lap_delta(telemetry: pd.DataFrame, track_length_m: float,
                      current_lap_time: float) -> Dict:
    """
    Analyze potential lap time improvement and identify key areas.

    Args:
        telemetry: Telemetry DataFrame
        track_length_m: Track length in meters
        current_lap_time: Driver's current lap time in seconds

    Returns:
        analysis: Dictionary with:
            - optimal_time: Theoretical best (s)
            - current_time: Current lap time (s)
            - potential_gain: Time improvement possible (s)
            - gain_percentage: Improvement %
            - corner_analysis: DataFrame with per-corner analysis
            - top_opportunities: List of biggest time gain opportunities
    """
    optimal_time, profile = calculate_optimal_lap_time_physics(telemetry, track_length_m)

    potential_gain = current_lap_time - optimal_time
    gain_percentage = (potential_gain / current_lap_time) * 100

    # Identify corners (high curvature zones)
    profile['is_corner'] = profile['curvature'] > 0.02  # Radius < 50m

    # Calculate speed deficit
    profile['speed_deficit_mps'] = profile['v_optimal'] - profile['v_actual']
    profile['speed_deficit_pct'] = (profile['speed_deficit_mps'] / profile['v_optimal']) * 100

    # Find time loss zones
    profile['time_loss_rate'] = np.gradient(
        profile['lap_time'].values - np.linspace(0, current_lap_time, len(profile))
    )

    # Identify top opportunities (corners with biggest time loss)
    corner_zones = []
    in_corner = False
    corner_start = 0

    for i, row in profile.iterrows():
        if row['is_corner'] and not in_corner:
            corner_start = i
            in_corner = True
        elif not row['is_corner'] and in_corner:
            corner_data = profile.iloc[corner_start:i]
            corner_zones.append({
                'corner_num': len(corner_zones) + 1,
                'start_distance': corner_data['arc_length'].iloc[0],
                'end_distance': corner_data['arc_length'].iloc[-1],
                'min_radius': corner_data['radius_m'].min(),
                'avg_speed_deficit': corner_data['speed_deficit_mps'].mean(),
                'max_speed_deficit': corner_data['speed_deficit_mps'].max(),
                'time_loss_est': (corner_data['arc_length'].iloc[-1] -
                                 corner_data['arc_length'].iloc[0]) /
                                corner_data['v_actual'].mean() -
                                (corner_data['arc_length'].iloc[-1] -
                                 corner_data['arc_length'].iloc[0]) /
                                corner_data['v_optimal'].mean()
            })
            in_corner = False

    corner_analysis = pd.DataFrame(corner_zones)
    if len(corner_analysis) > 0:
        corner_analysis = corner_analysis.sort_values('time_loss_est', ascending=False)
        top_opportunities = corner_analysis.head(5).to_dict('records')
    else:
        top_opportunities = []

    return {
        'optimal_time': optimal_time,
        'current_time': current_lap_time,
        'potential_gain': potential_gain,
        'gain_percentage': gain_percentage,
        'corner_analysis': corner_analysis,
        'top_opportunities': top_opportunities,
        'speed_profile': profile
    }


# Example usage and validation
if __name__ == "__main__":
    """
    Example: Generate synthetic track data and calculate optimal lap time
    """

    # Create synthetic track (1.5km with various corners)
    n_points = 1000
    track_length = 1500  # meters
    s = np.linspace(0, track_length, n_points)

    # Synthetic GPS coordinates (track with straights and corners)
    theta = s / track_length * 2 * np.pi  # One lap
    radius_base = 200  # base radius

    # Add corners
    x = radius_base * np.cos(theta) + 30 * np.sin(4 * theta)
    y = radius_base * np.sin(theta) + 30 * np.cos(4 * theta)

    # Synthetic speed (driver not optimal)
    speed = 25 + 15 * np.cos(4 * theta)  # 10-40 m/s

    # Create telemetry DataFrame
    telemetry = pd.DataFrame({
        'arc_length': s,
        'x': x,
        'y': y,
        'speed': speed,
        'accx_can': np.gradient(speed, s) * speed,
        'accy_can': np.zeros(n_points)
    })

    # Calculate optimal lap time
    optimal_time, profile = calculate_optimal_lap_time_physics(telemetry, track_length)

    # Current lap time (from speed profile)
    current_time = np.trapz(1.0 / speed, s)

    print("=" * 60)
    print("OPTIMAL LAP TIME ANALYSIS - GR86 Cup Car")
    print("=" * 60)
    print(f"Track Length: {track_length:.1f} m ({track_length/1000:.2f} km)")
    print(f"Current Lap Time: {current_time:.2f} seconds")
    print(f"Optimal Lap Time: {optimal_time:.2f} seconds")
    print(f"Potential Gain: {current_time - optimal_time:.2f} seconds")
    print(f"Improvement: {((current_time - optimal_time) / current_time * 100):.1f}%")
    print("=" * 60)

    # Show speed statistics
    print(f"\nSpeed Statistics:")
    print(f"  Current avg speed: {np.mean(speed):.1f} m/s ({np.mean(speed)*3.6:.1f} km/h)")
    print(f"  Optimal avg speed: {np.mean(profile['v_optimal']):.1f} m/s ({np.mean(profile['v_optimal'])*3.6:.1f} km/h)")
    print(f"  Max corner speed: {np.min(profile['v_max_corner']):.1f} m/s")
    print(f"  Tightest radius: {np.min(profile['radius_m']):.1f} m")

    # Full analysis
    analysis = analyze_lap_delta(telemetry, track_length, current_time)

    if len(analysis['top_opportunities']) > 0:
        print(f"\nTop 3 Improvement Opportunities:")
        for i, opp in enumerate(analysis['top_opportunities'][:3]):
            print(f"  {i+1}. Corner {opp['corner_num']}: "
                  f"{opp['time_loss_est']:.3f}s potential gain "
                  f"(R={opp['min_radius']:.1f}m, "
                  f"speed deficit={opp['avg_speed_deficit']:.1f} m/s)")
