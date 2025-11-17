"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_vehicle_dynamics_quality(
    sim_trajectories: np.ndarray,
    real_trajectories: np.ndarray,
    maneuver_type: str = 'general',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate vehicle dynamics and physics realism in simulation.
    
    Validates acceleration, braking, steering response, and overall motion characteristics
    compared to real-world vehicle dynamics. Critical for planning and control validation.
    
    Args:
        sim_trajectories: Simulated vehicle trajectories (N, T, D) where:
            - N: number of trajectories
            - T: timesteps
            - D: dimensions [x, y, velocity, acceleration, yaw_rate, steering_angle, ...]
            Minimum required: (N, T, 4) with [x, y, vx, vy] or [x, y, v, heading]
        real_trajectories: Real-world trajectories, same shape as sim_trajectories
        maneuver_type: Type of maneuver being validated. Options:
            - 'general': General driving validation
            - 'acceleration': Longitudinal acceleration validation
            - 'braking': Braking/deceleration validation
            - 'lane_change': Lateral dynamics validation
            - 'turning': Cornering and yaw dynamics
            - 'emergency': Emergency maneuvers
        metrics: List of metrics to compute. Options:
            - 'acceleration_profile': Acceleration curve matching
            - 'braking_distance': Braking distance accuracy
            - 'lateral_dynamics': Lateral acceleration/jerk
            - 'yaw_rate': Yaw rate distribution
            - 'trajectory_smoothness': Jerk and curvature
            - 'speed_distribution': Speed histogram matching
            - 'reaction_time': Response delay modeling
            
    Returns:
        Dictionary with requested metrics:
            - acceleration_mean_error: Mean absolute error in acceleration (m/s²)
            - acceleration_kl_divergence: KL divergence of acceleration distributions
            - braking_distance_error: Mean absolute error in braking distance (m)
            - deceleration_rate_ratio: Ratio of deceleration rates
            - lateral_accel_error: Lateral acceleration error (m/s²)
            - lateral_jerk_ratio: Ratio of lateral jerk magnitudes
            - yaw_rate_kl_divergence: KL divergence of yaw rate distributions
            - trajectory_smoothness: Curvature and jerk-based smoothness (0-1)
            - speed_kl_divergence: KL divergence of speed distributions
            - speed_mean_error: Mean absolute speed error (m/s)
            - reaction_time_error: Reaction time difference (seconds)
            - overall_dynamics_score: Combined physics quality score (0-100)
        
    Example:
        >>> # Acceleration validation
        >>> sim_traj = np.random.randn(50, 100, 4)  # 50 trajectories, 100 timesteps
        >>> real_traj = sim_traj + np.random.randn(*sim_traj.shape) * 0.1
        >>> quality = calculate_vehicle_dynamics_quality(
        ...     sim_traj, real_traj,
        ...     maneuver_type='acceleration',
        ...     metrics=['acceleration_profile', 'speed_distribution']
        ... )
        >>> print(f"Acceleration error: {quality['acceleration_mean_error']:.2f} m/s²")
        
    Notes:
        - Typical acceleration ranges:
            * Normal acceleration: 0-3 m/s²
            * Aggressive acceleration: 3-5 m/s²
            * Emergency: > 5 m/s²
        - Typical braking ranges:
            * Normal braking: 1-3 m/s²
            * Hard braking: 3-6 m/s²
            * Emergency: 6-9 m/s² (ABS limit)
        - Lateral acceleration limits:
            * Comfortable: < 0.3g (~3 m/s²)
            * Sporty: 0.3-0.5g (~3-5 m/s²)
            * Limit: 0.5-0.8g (~5-8 m/s²)
        - Overall dynamics score > 75 is considered good for production use
    """
    if metrics is None:
        metrics = [
            'acceleration_profile', 'speed_distribution',
            'trajectory_smoothness', 'lateral_dynamics'
        ]
    
    results = {}
    
    # Ensure trajectories have at least (N, T, 2) shape
    if len(sim_trajectories.shape) != 3 or len(real_trajectories.shape) != 3:
        raise ValueError(f"Trajectories must be 3D (N, T, D), got shapes {sim_trajectories.shape}, {real_trajectories.shape}")
    
    if sim_trajectories.shape != real_trajectories.shape:
        raise ValueError(f"Trajectory shape mismatch: {sim_trajectories.shape} vs {real_trajectories.shape}")
    
    N, T, D = sim_trajectories.shape
    
    # Extract positions (first 2 dimensions assumed to be x, y)
    sim_positions = sim_trajectories[:, :, :2]  # (N, T, 2)
    real_positions = real_trajectories[:, :, :2]
    
    # Calculate velocities if not provided (dimensions 2-3)
    if D >= 4:
        # Assume dimensions [x, y, vx, vy] or [x, y, v, heading]
        sim_velocities = sim_trajectories[:, :, 2:4]
        real_velocities = real_trajectories[:, :, 2:4]
    else:
        # Compute velocities from positions (finite difference)
        sim_velocities = np.diff(sim_positions, axis=1)  # (N, T-1, 2)
        real_velocities = np.diff(real_positions, axis=1)
        # Pad to maintain shape
        sim_velocities = np.concatenate([sim_velocities, sim_velocities[:, -1:, :]], axis=1)
        real_velocities = np.concatenate([real_velocities, real_velocities[:, -1:, :]], axis=1)
    
    # Calculate speeds (magnitude of velocity)
    sim_speeds = np.sqrt(np.sum(sim_velocities**2, axis=-1))  # (N, T)
    real_speeds = np.sqrt(np.sum(real_velocities**2, axis=-1))
    
    # Calculate accelerations (finite difference of velocities)
    sim_accels = np.diff(sim_velocities, axis=1)  # (N, T-1, 2)
    real_accels = np.diff(real_velocities, axis=1)
    
    # Longitudinal acceleration (along velocity direction)
    sim_accel_mag = np.sqrt(np.sum(sim_accels**2, axis=-1))  # (N, T-1)
    real_accel_mag = np.sqrt(np.sum(real_accels**2, axis=-1))
    
    # Acceleration profile validation
    if 'acceleration_profile' in metrics or maneuver_type == 'acceleration':
        sim_accel_flat = sim_accel_mag.flatten()
        real_accel_flat = real_accel_mag.flatten()
        
        # Remove outliers (> 10 m/s² likely noise)
        sim_accel_clean = sim_accel_flat[sim_accel_flat < 10]
        real_accel_clean = real_accel_flat[real_accel_flat < 10]
        
        if len(sim_accel_clean) > 0 and len(real_accel_clean) > 0:
            results['acceleration_mean_error'] = float(
                np.abs(sim_accel_clean.mean() - real_accel_clean.mean())
            )
            results['acceleration_std_ratio'] = float(
                sim_accel_clean.std() / (real_accel_clean.std() + 1e-6)
            )
            
            # KL divergence of acceleration distributions
            bins = np.linspace(0, min(sim_accel_clean.max(), real_accel_clean.max()), 30)
            if len(bins) > 1:
                sim_hist, _ = np.histogram(sim_accel_clean, bins=bins, density=True)
                real_hist, _ = np.histogram(real_accel_clean, bins=bins, density=True)
                
                sim_hist = sim_hist + 1e-10
                real_hist = real_hist + 1e-10
                sim_hist = sim_hist / sim_hist.sum()
                real_hist = real_hist / real_hist.sum()
                
                accel_kl = stats.entropy(sim_hist, real_hist)
                results['acceleration_kl_divergence'] = float(accel_kl)
    
    # Braking distance and deceleration validation
    if 'braking_distance' in metrics or maneuver_type == 'braking':
        # Identify braking events (negative acceleration)
        sim_decels = sim_accel_mag[sim_accel_mag > 1.0]  # Deceleration > 1 m/s²
        real_decels = real_accel_mag[real_accel_mag > 1.0]
        
        if len(sim_decels) > 0 and len(real_decels) > 0:
            results['deceleration_mean'] = float(sim_decels.mean())
            results['deceleration_real_mean'] = float(real_decels.mean())
            results['deceleration_rate_ratio'] = float(
                sim_decels.mean() / (real_decels.mean() + 1e-6)
            )
            
            # Estimate braking distance from speed and deceleration
            # d = v² / (2 * a)
            if sim_speeds.shape[1] > 1:
                sim_brake_dist = (sim_speeds[:, 0]**2) / (2 * sim_decels.mean() + 1e-6)
                real_brake_dist = (real_speeds[:, 0]**2) / (2 * real_decels.mean() + 1e-6)
                results['braking_distance_error'] = float(
                    np.abs(sim_brake_dist.mean() - real_brake_dist.mean())
                )
    
    # Lateral dynamics (for lane changes, turning)
    if 'lateral_dynamics' in metrics or maneuver_type in ['lane_change', 'turning']:
        # Lateral acceleration (perpendicular to velocity)
        if sim_accels.shape[1] > 1:
            # Simple approximation: use y-component of acceleration
            sim_lateral_accel = np.abs(sim_accels[:, :, 1])  # (N, T-1)
            real_lateral_accel = np.abs(real_accels[:, :, 1])
            
            results['lateral_accel_error'] = float(
                np.abs(sim_lateral_accel.mean() - real_lateral_accel.mean())
            )
            results['lateral_accel_max_sim'] = float(sim_lateral_accel.max())
            results['lateral_accel_max_real'] = float(real_lateral_accel.max())
            
            # Lateral jerk (rate of change of lateral acceleration)
            sim_lateral_jerk = np.abs(np.diff(sim_lateral_accel, axis=1))
            real_lateral_jerk = np.abs(np.diff(real_lateral_accel, axis=1))
            
            if sim_lateral_jerk.size > 0 and real_lateral_jerk.size > 0:
                results['lateral_jerk_ratio'] = float(
                    sim_lateral_jerk.mean() / (real_lateral_jerk.mean() + 1e-6)
                )
    
    # Yaw rate validation (if available in data)
    if 'yaw_rate' in metrics and D >= 5:
        # Assume dimension 4 is yaw rate
        sim_yaw_rate = sim_trajectories[:, :, 4].flatten()
        real_yaw_rate = real_trajectories[:, :, 4].flatten()
        
        # Remove outliers
        sim_yaw_clean = sim_yaw_rate[np.abs(sim_yaw_rate) < 1.0]  # < 1 rad/s
        real_yaw_clean = real_yaw_rate[np.abs(real_yaw_rate) < 1.0]
        
        if len(sim_yaw_clean) > 10 and len(real_yaw_clean) > 10:
            bins = np.linspace(-0.5, 0.5, 30)
            sim_hist, _ = np.histogram(sim_yaw_clean, bins=bins, density=True)
            real_hist, _ = np.histogram(real_yaw_clean, bins=bins, density=True)
            
            sim_hist = sim_hist + 1e-10
            real_hist = real_hist + 1e-10
            sim_hist = sim_hist / sim_hist.sum()
            real_hist = real_hist / real_hist.sum()
            
            yaw_kl = stats.entropy(sim_hist, real_hist)
            results['yaw_rate_kl_divergence'] = float(yaw_kl)
    
    # Trajectory smoothness (jerk and curvature)
    if 'trajectory_smoothness' in metrics:
        # Jerk: rate of change of acceleration
        sim_jerk = np.abs(np.diff(sim_accel_mag, axis=1))  # (N, T-2)
        real_jerk = np.abs(np.diff(real_accel_mag, axis=1))
        
        if sim_jerk.size > 0 and real_jerk.size > 0:
            results['jerk_mean_sim'] = float(sim_jerk.mean())
            results['jerk_mean_real'] = float(real_jerk.mean())
            results['jerk_ratio'] = float(
                sim_jerk.mean() / (real_jerk.mean() + 1e-6)
            )
            
            # Smoothness score: lower jerk = smoother
            # Normalize to 0-1 range (typical jerk < 5 m/s³)
            sim_smoothness = 1.0 - np.clip(sim_jerk.mean() / 5.0, 0, 1)
            real_smoothness = 1.0 - np.clip(real_jerk.mean() / 5.0, 0, 1)
            results['trajectory_smoothness'] = float((sim_smoothness + real_smoothness) / 2)
    
    # Speed distribution matching
    if 'speed_distribution' in metrics:
        sim_speeds_flat = sim_speeds.flatten()
        real_speeds_flat = real_speeds.flatten()
        
        # Remove very low speeds (< 0.5 m/s, likely stationary)
        sim_speeds_clean = sim_speeds_flat[sim_speeds_flat > 0.5]
        real_speeds_clean = real_speeds_flat[real_speeds_flat > 0.5]
        
        if len(sim_speeds_clean) > 0 and len(real_speeds_clean) > 0:
            results['speed_mean_error'] = float(
                np.abs(sim_speeds_clean.mean() - real_speeds_clean.mean())
            )
            results['speed_mean_sim'] = float(sim_speeds_clean.mean())
            results['speed_mean_real'] = float(real_speeds_clean.mean())
            
            # Speed distribution KL divergence
            max_speed = min(sim_speeds_clean.max(), real_speeds_clean.max(), 50)  # Cap at 50 m/s
            bins = np.linspace(0, max_speed, 30)
            if len(bins) > 1:
                sim_hist, _ = np.histogram(sim_speeds_clean, bins=bins, density=True)
                real_hist, _ = np.histogram(real_speeds_clean, bins=bins, density=True)
                
                sim_hist = sim_hist + 1e-10
                real_hist = real_hist + 1e-10
                sim_hist = sim_hist / sim_hist.sum()
                real_hist = real_hist / real_hist.sum()
                
                speed_kl = stats.entropy(sim_hist, real_hist)
                results['speed_kl_divergence'] = float(speed_kl)
    
    # Reaction time estimation (if maneuver has clear response)
    if 'reaction_time' in metrics or maneuver_type == 'emergency':
        # Find time to maximum acceleration/deceleration
        if sim_accel_mag.shape[1] > 5:
            sim_response_time = np.argmax(sim_accel_mag, axis=1).mean()
            real_response_time = np.argmax(real_accel_mag, axis=1).mean()
            
            results['reaction_time_error'] = float(
                np.abs(sim_response_time - real_response_time)
            )
            results['reaction_time_sim'] = float(sim_response_time)
            results['reaction_time_real'] = float(real_response_time)
    
    # Overall dynamics quality score (0-100)
    if len(results) > 0:
        score_components = []
        
        # Acceleration accuracy (lower error = better)
        if 'acceleration_mean_error' in results:
            # Typical acceleration error < 1 m/s² is good
            accel_score = 100 * (1 - np.clip(results['acceleration_mean_error'] / 2.0, 0, 1))
            score_components.append(accel_score)
        
        # Speed distribution matching
        if 'speed_kl_divergence' in results:
            speed_score = 100 * np.exp(-results['speed_kl_divergence'])
            score_components.append(speed_score)
        
        # Trajectory smoothness
        if 'trajectory_smoothness' in results:
            smooth_score = 100 * results['trajectory_smoothness']
            score_components.append(smooth_score)
        
        # Lateral dynamics (lower error = better)
        if 'lateral_accel_error' in results:
            # Typical lateral error < 0.5 m/s² is good
            lateral_score = 100 * (1 - np.clip(results['lateral_accel_error'] / 1.0, 0, 1))
            score_components.append(lateral_score)
        
        if score_components:
            results['overall_dynamics_score'] = float(np.mean(score_components))
    
    return results


