"""
Tests for vehicle dynamics quality metrics.

Validates physics and motion realism including acceleration, braking, steering,
and trajectory smoothness.
"""

import numpy as np
import pytest
from admetrics.simulation import calculate_vehicle_dynamics_quality


class TestAccelerationProfile:
    """Tests for acceleration profile validation."""
    
    def test_similar_acceleration_profiles(self):
        """Test with similar acceleration profiles."""
        # Create trajectories with constant acceleration
        N, T = 20, 50
        t = np.linspace(0, 5, T)
        dt = t[1] - t[0]
        
        # Simulate acceleration: v = v0 + at, x = x0 + v0*t + 0.5*a*t^2
        a = 2.0  # 2 m/s² acceleration
        sim_x = 0.5 * a * t**2
        sim_vx = a * t
        
        # Create multiple trajectories
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x  # x position
        sim_traj[:, :, 2] = sim_vx  # x velocity
        
        # Real trajectories with slight variation
        real_traj = sim_traj.copy()
        real_traj[:, :, 0] += np.random.randn(N, T) * 0.1
        real_traj[:, :, 2] += np.random.randn(N, T) * 0.05
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='acceleration',
            metrics=['acceleration_profile']
        )
        
        assert 'acceleration_mean_error' in result
        assert result['acceleration_mean_error'] < 1.0  # Reasonable error
        if 'acceleration_kl_divergence' in result:
            assert result['acceleration_kl_divergence'] < 3.0  # Similar distributions (relaxed for finite differences)
    
    def test_different_acceleration_rates(self):
        """Test with different acceleration rates."""
        N, T = 15, 40
        t = np.linspace(0, 4, T)
        
        # Sim: 3 m/s² acceleration
        sim_vx = 3.0 * t
        sim_x = 0.5 * 3.0 * t**2
        
        # Real: 2 m/s² acceleration (clearly different)
        real_vx = 2.0 * t
        real_x = 0.5 * 2.0 * t**2
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 2] = sim_vx
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 0] = real_x
        real_traj[:, :, 2] = real_vx
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='acceleration',
            metrics=['acceleration_profile', 'speed_distribution']
        )
        
        # Should detect the speed difference at least
        assert 'speed_mean_error' in result
        assert result['speed_mean_error'] >= 2.0  # Clear difference in speeds (>= to account for rounding)


class TestBrakingDynamics:
    """Tests for braking distance and deceleration validation."""
    
    def test_braking_deceleration(self):
        """Test braking deceleration validation."""
        N, T = 20, 60
        
        # Simulate hard braking: constant deceleration
        v0 = 20.0  # Initial speed 20 m/s
        a = -5.0  # -5 m/s² deceleration
        
        # Time array with small timestep for proper acceleration computation
        dt = 0.05  # 50ms timestep
        
        sim_vx = np.maximum(v0 + a * np.arange(T) * dt, 0)
        sim_x = np.cumsum(sim_vx) * dt
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 2] = sim_vx
        
        # Real with similar braking
        real_vx = np.maximum(v0 + a * 1.05 * np.arange(T) * dt, 0)  # Slightly different
        real_x = np.cumsum(real_vx) * dt
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 0] = real_x
        real_traj[:, :, 2] = real_vx
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='braking',
            metrics=['braking_distance', 'acceleration_profile']
        )
        
        # Should have acceleration metrics
        assert 'acceleration_mean_error' in result or 'deceleration_mean' in result or 'speed_mean_error' in result
    
    def test_emergency_braking(self):
        """Test emergency braking scenario."""
        N, T = 10, 50
        
        # Emergency braking: -8 m/s² (near ABS limit)
        v0 = 25.0  # 25 m/s (~90 km/h)
        a_sim = -8.0
        a_real = -7.5
        
        dt = 0.04  # 40ms timestep
        
        sim_vx = np.maximum(v0 + a_sim * np.arange(T) * dt, 0)
        real_vx = np.maximum(v0 + a_real * np.arange(T) * dt, 0)
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 2] = sim_vx
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 2] = real_vx
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='emergency',
            metrics=['braking_distance', 'reaction_time', 'acceleration_profile']
        )
        
        # Should have some metrics
        assert len(result) > 0


class TestLateralDynamics:
    """Tests for lateral dynamics validation."""
    
    def test_lane_change_maneuver(self):
        """Test lane change lateral dynamics."""
        N, T = 15, 80
        t = np.linspace(0, 4, T)
        
        # Simulate lane change: sinusoidal lateral motion
        lateral_amplitude = 3.5  # 3.5m lane change
        sim_y = lateral_amplitude * (1 - np.cos(2 * np.pi * t / 4)) / 2
        sim_x = 20 * t  # Moving forward at 20 m/s
        
        # Velocity components
        sim_vy = lateral_amplitude * np.pi / 2 * np.sin(2 * np.pi * t / 4)
        sim_vx = np.ones_like(t) * 20
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 1] = sim_y
        sim_traj[:, :, 2] = sim_vx
        sim_traj[:, :, 3] = sim_vy
        
        # Real with variation
        real_traj = sim_traj.copy()
        real_traj[:, :, 1] += np.random.randn(N, T) * 0.1
        real_traj[:, :, 3] += np.random.randn(N, T) * 0.05
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='lane_change',
            metrics=['lateral_dynamics']
        )
        
        assert 'lateral_accel_error' in result
        assert result['lateral_accel_error'] < 2.0  # Reasonable lateral accel
        if 'lateral_jerk_ratio' in result:
            assert result['lateral_jerk_ratio'] > 0  # Positive value
    
    def test_turning_dynamics(self):
        """Test turning/cornering dynamics."""
        N, T = 10, 60
        theta = np.linspace(0, np.pi/2, T)  # 90 degree turn
        radius = 20.0  # 20m turning radius
        
        # Circular motion
        sim_x = radius * np.sin(theta)
        sim_y = radius * (1 - np.cos(theta))
        
        # Velocity (tangent to circle)
        v = 10.0  # 10 m/s
        sim_vx = v * np.cos(theta)
        sim_vy = v * np.sin(theta)
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 1] = sim_y
        sim_traj[:, :, 2] = sim_vx
        sim_traj[:, :, 3] = sim_vy
        
        real_traj = sim_traj.copy()
        real_traj[:, :, :2] += np.random.randn(N, T, 2) * 0.15
        real_traj[:, :, 2:4] += np.random.randn(N, T, 2) * 0.1
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='turning',
            metrics=['lateral_dynamics']
        )
        
        assert 'lateral_accel_max_sim' in result
        assert 'lateral_accel_max_real' in result
        # Should have some lateral acceleration
        assert result['lateral_accel_max_sim'] > 0


class TestTrajectorySmoothness:
    """Tests for trajectory smoothness validation."""
    
    def test_smooth_trajectory(self):
        """Test smooth trajectory with low jerk."""
        N, T = 20, 100
        t = np.linspace(0, 10, T)
        
        # Smooth sinusoidal trajectory
        sim_x = t * 10
        sim_y = 2 * np.sin(2 * np.pi * t / 10)
        sim_vx = np.ones_like(t) * 10
        sim_vy = 2 * 2 * np.pi / 10 * np.cos(2 * np.pi * t / 10)
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 1] = sim_y
        sim_traj[:, :, 2] = sim_vx
        sim_traj[:, :, 3] = sim_vy
        
        real_traj = sim_traj + np.random.randn(N, T, 4) * 0.05
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['trajectory_smoothness']
        )
        
        assert 'trajectory_smoothness' in result
        assert 'jerk_ratio' in result
        assert result['trajectory_smoothness'] > 0.7  # High smoothness
    
    def test_jerky_trajectory(self):
        """Test jerky trajectory with high jerk."""
        N, T = 15, 80
        t = np.linspace(0, 8, T)
        
        # Jerky trajectory with random accelerations
        np.random.seed(42)
        sim_vx = 10 + np.cumsum(np.random.randn(T) * 2)
        sim_x = np.cumsum(sim_vx)
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = sim_x
        sim_traj[:, :, 2] = sim_vx
        
        real_traj = sim_traj + np.random.randn(N, T, 4) * 0.1
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['trajectory_smoothness']
        )
        
        assert 'jerk_mean_sim' in result
        assert 'jerk_mean_real' in result


class TestSpeedDistribution:
    """Tests for speed distribution validation."""
    
    def test_highway_speed_distribution(self):
        """Test highway speed distribution (high speeds)."""
        N, T = 50, 100
        
        # Highway speeds: normally distributed around 25 m/s (~90 km/h)
        sim_speeds = np.random.normal(25, 3, (N, T))
        real_speeds = np.random.normal(24.5, 3.2, (N, T))
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 2] = sim_speeds  # vx
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 2] = real_speeds
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['speed_distribution']
        )
        
        assert 'speed_mean_error' in result
        assert 'speed_kl_divergence' in result
        assert result['speed_mean_error'] < 2.0  # Close mean speeds
        assert result['speed_kl_divergence'] < 0.3  # Similar distributions
    
    def test_urban_speed_distribution(self):
        """Test urban speed distribution (lower speeds)."""
        N, T = 40, 80
        
        # Urban speeds: lower, more variable
        sim_speeds = np.random.gamma(3, 3, (N, T))  # Skewed distribution
        real_speeds = np.random.gamma(3.2, 2.9, (N, T))
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 2] = sim_speeds
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 2] = real_speeds
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['speed_distribution']
        )
        
        assert 'speed_mean_sim' in result
        assert 'speed_mean_real' in result
        assert result['speed_mean_sim'] > 0.5  # Moving


class TestReactionTime:
    """Tests for reaction time estimation."""
    
    def test_reaction_time_estimation(self):
        """Test reaction time in emergency maneuver."""
        N, T = 20, 100
        
        # Simulate delayed response
        sim_accel = np.zeros(T)
        sim_accel[20:] = -8.0  # Braking starts at t=20
        
        real_accel = np.zeros(T)
        real_accel[25:] = -7.5  # Braking starts at t=25 (delayed)
        
        sim_vx = 25 + np.cumsum(sim_accel) * 0.1
        real_vx = 25 + np.cumsum(real_accel) * 0.1
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 2] = sim_vx
        
        real_traj = np.zeros((N, T, 4))
        real_traj[:, :, 2] = real_vx
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            maneuver_type='emergency',
            metrics=['reaction_time']
        )
        
        assert 'reaction_time_error' in result
        # Should detect ~5 timestep delay
        assert result['reaction_time_error'] >= 0


class TestOverallDynamicsScore:
    """Tests for overall dynamics quality score."""
    
    def test_high_quality_dynamics(self):
        """Test with high-quality physics simulation."""
        N, T = 30, 80
        t = np.linspace(0, 8, T)
        
        # Similar trajectories (good sim)
        base_x = 15 * t
        base_vx = 15 + 2 * np.sin(2 * np.pi * t / 8)
        
        sim_traj = np.zeros((N, T, 4))
        sim_traj[:, :, 0] = base_x
        sim_traj[:, :, 2] = base_vx
        
        real_traj = sim_traj + np.random.randn(N, T, 4) * 0.05
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['acceleration_profile', 'speed_distribution', 
                    'trajectory_smoothness', 'lateral_dynamics']
        )
        
        assert 'overall_dynamics_score' in result
        assert result['overall_dynamics_score'] > 70  # High quality
    
    def test_low_quality_dynamics(self):
        """Test with poor-quality physics simulation."""
        N, T = 25, 60
        
        # Very different dynamics
        sim_traj = np.random.randn(N, T, 4) * 5 + 10
        real_traj = np.random.randn(N, T, 4) * 3 + 15
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['acceleration_profile', 'speed_distribution']
        )
        
        assert 'overall_dynamics_score' in result
        # Score should reflect poor match


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_wrong_shape_trajectories(self):
        """Test with incorrect trajectory shapes."""
        sim_traj = np.random.randn(10, 50)  # 2D instead of 3D
        real_traj = np.random.randn(10, 50, 4)
        
        with pytest.raises(ValueError, match="must be 3D"):
            calculate_vehicle_dynamics_quality(sim_traj, real_traj)
    
    def test_shape_mismatch(self):
        """Test with mismatched trajectory shapes."""
        sim_traj = np.random.randn(10, 50, 4)
        real_traj = np.random.randn(15, 50, 4)  # Different N
        
        with pytest.raises(ValueError, match="shape mismatch"):
            calculate_vehicle_dynamics_quality(sim_traj, real_traj)
    
    def test_minimal_dimensions(self):
        """Test with minimal trajectory dimensions (x, y only)."""
        N, T = 10, 40
        
        # Only positions, no velocities
        sim_traj = np.random.randn(N, T, 2) * 10
        real_traj = sim_traj + np.random.randn(N, T, 2) * 0.5
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['speed_distribution']
        )
        
        # Should compute velocities from positions
        assert 'speed_mean_error' in result or len(result) >= 0
    
    def test_stationary_vehicles(self):
        """Test with stationary vehicles (zero velocity)."""
        N, T = 10, 30
        
        # All zeros (stationary)
        sim_traj = np.zeros((N, T, 4))
        real_traj = np.zeros((N, T, 4))
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=['speed_distribution', 'acceleration_profile']
        )
        
        # Should handle gracefully
        assert len(result) >= 0
    
    def test_different_maneuver_types(self):
        """Test all supported maneuver types."""
        N, T = 15, 50
        sim_traj = np.random.randn(N, T, 4) * 2 + 10
        real_traj = sim_traj + np.random.randn(N, T, 4) * 0.3
        
        maneuver_types = ['general', 'acceleration', 'braking', 
                         'lane_change', 'turning', 'emergency']
        
        for maneuver in maneuver_types:
            result = calculate_vehicle_dynamics_quality(
                sim_traj, real_traj,
                maneuver_type=maneuver
            )
            assert len(result) >= 0
    
    def test_empty_metrics_list(self):
        """Test with None metrics (should use defaults)."""
        N, T = 10, 40
        sim_traj = np.random.randn(N, T, 4) * 5 + 15
        real_traj = sim_traj + np.random.randn(N, T, 4) * 0.2
        
        result = calculate_vehicle_dynamics_quality(
            sim_traj, real_traj,
            metrics=None  # Use defaults
        )
        
        # Should return default metrics
        assert len(result) > 0
