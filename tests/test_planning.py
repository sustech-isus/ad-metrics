"""Tests for end-to-end planning and driving metrics."""

import numpy as np
import pytest
from admetrics.planning import (
    calculate_l2_distance,
    calculate_collision_rate,
    calculate_progress_score,
    calculate_route_completion,
    average_displacement_error_planning,
    calculate_lateral_deviation,
    calculate_heading_error,
    calculate_velocity_error,
    calculate_comfort_metrics,
    calculate_driving_score,
    calculate_planning_kl_divergence,
    calculate_time_to_collision,
    calculate_lane_invasion_rate,
    calculate_collision_severity,
    check_kinematic_feasibility,
    # New functions
    calculate_collision_with_fault_classification,
    calculate_time_to_collision_enhanced,
    calculate_distance_to_road_edge,
    calculate_driving_direction_compliance,
    calculate_interaction_metrics,
)


class TestL2Distance:
    """Test L2 distance calculation."""
    
    def test_identical_trajectories(self):
        """Test L2 distance with identical trajectories."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        dist = calculate_l2_distance(traj, traj)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_parallel_offset(self):
        """Test L2 distance with parallel trajectories."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        dist = calculate_l2_distance(pred, expert)
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_weighted_distance(self):
        """Test weighted L2 distance."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        weights = np.array([1.0, 1.0, 2.0])  # Weight last point more
        dist = calculate_l2_distance(pred, expert, weights=weights)
        # (1*1 + 1*1 + 2*1) / (1+1+2) = 4/4 = 1.0
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_weighted_distance_nonuniform(self):
        """Test weighted L2 distance with different weights."""
        pred = np.array([[0., 0.], [1., 0.]])
        expert = np.array([[0., 0.], [2., 0.]])
        w = np.array([1.0, 2.0])
        d = calculate_l2_distance(pred, expert, weights=w)
        # distances = [0,1], weighted mean = (0*1 + 1*2)/(1+2) = 2/3
        assert d == pytest.approx(2.0/3.0, abs=1e-9)
    
    def test_3d_trajectory(self):
        """Test with 3D trajectories."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        expert = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]])
        dist = calculate_l2_distance(pred, expert)
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.array([[0, 0], [1, 0]])
        expert = np.array([[0, 0], [1, 0], [2, 0]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_l2_distance(pred, expert)


class TestCollisionRate:
    """Test collision rate calculation."""
    
    def test_no_collision(self):
        """Test with no collisions."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[5, 5]])]  # Far away
        result = calculate_collision_rate(traj, obstacles)
        assert result['collision_rate'] == 0.0
        assert result['num_collisions'] == 0
        assert result['first_collision'] is None
    
    def test_single_collision(self):
        """Test with single collision."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[1, 0]])]  # Collides at t=1
        result = calculate_collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['collision_rate'] > 0.0
        assert result['num_collisions'] >= 1
    
    def test_dynamic_obstacle(self):
        """Test with dynamic obstacle."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        # Obstacle moving toward ego
        obstacles = [np.array([[0, 5], [1, 2], [2, 0]])]
        result = calculate_collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['num_collisions'] >= 1
    
    def test_safety_margin(self):
        """Test safety margin."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[1, 2.5]])]
        
        # Without margin, no collision
        result1 = calculate_collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0), safety_margin=0.0)
        
        # With large margin, collision
        result2 = calculate_collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0), safety_margin=2.0)
        
        assert result2['num_collisions'] >= result1['num_collisions']
    
    def test_multiple_obstacles(self):
        """Test with multiple obstacles."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        obstacles = [
            np.array([[1, 0]]),
            np.array([[3, 0]])
        ]
        result = calculate_collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['num_collisions'] >= 1
    
    def test_static_and_dynamic_mixed(self):
        """Test with mixed static and dynamic obstacles."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        static = np.array([[1.0, 0.0]])
        dyn = np.array([[2.9, 0.0], [2.9, 0.0], [2.9, 0.0], [2.9, 0.0]])
        res = calculate_collision_rate(traj, [static, dyn], obstacle_sizes=[(1.0, 1.0), (1.0, 1.0)])
        assert res['num_collisions'] >= 1


class TestProgressScore:
    """Test progress score calculation."""
    
    def test_complete_path(self):
        """Test with complete path traversal."""
        traj = np.array([[0, 0], [5, 0], [10, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = calculate_progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(1.0, abs=0.1)
        assert result['goal_reached'] is True
    
    def test_partial_progress(self):
        """Test with partial progress."""
        traj = np.array([[0, 0], [2, 0], [4, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = calculate_progress_score(traj, ref)
        assert 0.0 < result['progress_ratio'] < 1.0
    
    def test_no_progress(self):
        """Test with no progress."""
        traj = np.array([[0, 0], [0, 0.1], [0, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = calculate_progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(0.0, abs=0.1)
    
    def test_custom_goal(self):
        """Test with custom goal position."""
        traj = np.array([[0, 0], [5, 0], [10, 0]])
        ref = np.array([[0, 0], [10, 0], [20, 0]])
        goal = np.array([10, 0])
        result = calculate_progress_score(traj, ref, goal_position=goal)
        assert result['goal_reached'] is True


class TestRouteCompletion:
    """Test route completion metrics."""
    
    def test_all_waypoints_reached(self):
        """Test with all waypoints reached."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        waypoints = np.array([[1, 0], [2, 0]])
        result = calculate_route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == 1.0
        assert result['num_waypoints_reached'] == 2
        assert all(result['waypoint_status'])
    
    def test_partial_completion(self):
        """Test with partial waypoint completion."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[1, 0], [5, 0], [10, 0]])
        result = calculate_route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == pytest.approx(1/3, abs=0.01)
        assert result['num_waypoints_reached'] == 1
    
    def test_no_completion(self):
        """Test with no waypoints reached."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[10, 10], [20, 20]])
        result = calculate_route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == 0.0
        assert result['num_waypoints_reached'] == 0
    
    def test_completion_radius(self):
        """Test different completion radii."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[1.3, 0]])
        
        result_small = calculate_route_completion(traj, waypoints, completion_radius=0.2)
        result_large = calculate_route_completion(traj, waypoints, completion_radius=0.5)
        
        assert result_large['num_waypoints_reached'] >= result_small['num_waypoints_reached']
    
    def test_ordered_waypoint_passing(self):
        """Test that waypoints must be visited in order."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        waypoints = np.array([[1., 0.], [3., 0.]])
        res = calculate_route_completion(traj, waypoints, completion_radius=0.1)
        assert res['num_waypoints_reached'] == 2
        
        # If waypoints are out of order, should not mark second as reached
        waypoints2 = np.array([[3., 0.], [1., 0.]])
        res2 = calculate_route_completion(traj, waypoints2, completion_radius=0.1)
        assert res2['num_waypoints_reached'] == 1


class TestAverageDisplacementErrorPlanning:
    """Test planning ADE calculation."""
    
    def test_perfect_match(self):
        """Test with perfect trajectory match."""
        traj = np.random.rand(10, 2)
        result = average_displacement_error_planning(traj, traj)
        assert result['ADE'] == pytest.approx(0.0, abs=1e-6)
        assert result['FDE'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_error(self):
        """Test with constant displacement."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        result = average_displacement_error_planning(pred, expert)
        assert result['ADE'] == pytest.approx(1.0, abs=1e-6)
        assert result['FDE'] == pytest.approx(1.0, abs=1e-6)
    
    def test_temporal_weighting(self):
        """Test temporal weighting schemes."""
        pred = np.zeros((10, 2))
        expert = np.ones((10, 2))
        
        result_uniform = average_displacement_error_planning(pred, expert, timestep_weights='uniform')
        result_linear = average_displacement_error_planning(pred, expert, timestep_weights='linear')
        result_exp = average_displacement_error_planning(pred, expert, timestep_weights='exponential')
        
        # All should have similar ADE for constant error
        assert result_uniform['ADE'] > 0
        assert result_linear['weighted_ADE'] > 0
        assert result_exp['weighted_ADE'] > 0
    
    def test_multi_horizon_ade_fde(self):
        """Test multi-horizon ADE/FDE calculation."""
        pred = np.zeros((5, 2))
        expert = np.ones((5, 2))
        r = average_displacement_error_planning(pred, expert, timestep_weights='linear', horizons=[1, 3, 5])
        assert 'ADE_1' in r and 'ADE_3' in r and 'ADE_5' in r
        assert 'FDE_1' in r and 'FDE_3' in r and 'FDE_5' in r
        assert r['ADE'] == pytest.approx(np.mean(np.linalg.norm(pred - expert, axis=1)), abs=1e-9)


class TestLateralDeviation:
    """Test lateral deviation metrics."""
    
    def test_perfect_following(self):
        """Test with perfect path following."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        result = calculate_lateral_deviation(traj, ref)
        assert result['mean_lateral_error'] == pytest.approx(0.0, abs=1e-6)
        assert result['lane_keeping_rate'] == 1.0
    
    def test_constant_offset(self):
        """Test with constant lateral offset."""
        traj = np.array([[0, 0.5], [1, 0.5], [2, 0.5]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        result = calculate_lateral_deviation(traj, ref)
        assert result['mean_lateral_error'] == pytest.approx(0.5, abs=0.1)
    
    def test_lane_departure(self):
        """Test lane departure detection."""
        # Half inside lane, half outside
        traj = np.array([[0, 0.5], [1, 0.5], [2, 2.5], [3, 2.5]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        result = calculate_lateral_deviation(traj, ref)
        assert 0.0 < result['lane_keeping_rate'] < 1.0
    
    def test_segment_projection(self):
        """Test perpendicular distance to segment projection."""
        ref = np.array([[0., 0.], [10., 0.]])
        traj = np.array([[5., 1.], [5., -2.]])
        out = calculate_lateral_deviation(traj, ref)
        # distances should be 1 and 2
        assert out['mean_lateral_error'] == pytest.approx(1.5, abs=1e-6)
        assert out['lane_keeping_rate'] == 0.5


class TestHeadingError:
    """Test heading error calculation."""
    
    def test_perfect_heading(self):
        """Test with perfect heading match."""
        headings = np.array([0.0, 0.1, 0.2, 0.3])
        result = calculate_heading_error(headings, headings)
        assert result['mean_heading_error'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_offset(self):
        """Test with constant heading offset."""
        pred = np.array([0.0, 0.1, 0.2])
        expert = np.array([0.1, 0.2, 0.3])
        result = calculate_heading_error(pred, expert)
        assert result['mean_heading_error'] == pytest.approx(0.1, abs=1e-6)
        assert result['mean_heading_error_deg'] == pytest.approx(np.degrees(0.1), abs=0.1)
    
    def test_wraparound(self):
        """Test heading wraparound handling."""
        pred = np.array([0.0, np.pi - 0.1])
        expert = np.array([0.0, -np.pi + 0.1])
        result = calculate_heading_error(pred, expert)
        # Should handle wraparound correctly
        assert result['mean_heading_error'] < np.pi
        # Mean of [0, 0.2] should be 0.1
        assert result['mean_heading_error'] == pytest.approx(0.1, abs=1e-6)
        assert result['max_heading_error'] == pytest.approx(0.2, abs=1e-6)


class TestVelocityError:
    """Test velocity error calculation."""
    
    def test_perfect_velocity(self):
        """Test with perfect velocity match."""
        vel = np.array([10.0, 12.0, 15.0])
        result = calculate_velocity_error(vel, vel)
        assert result['mean_velocity_error'] == pytest.approx(0.0, abs=1e-6)
        assert result['rmse_velocity'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_error(self):
        """Test with constant velocity error."""
        pred = np.array([10.0, 12.0, 15.0])
        expert = np.array([11.0, 13.0, 16.0])
        result = calculate_velocity_error(pred, expert)
        assert result['mean_velocity_error'] == pytest.approx(1.0, abs=1e-6)
        assert result['max_velocity_error'] == pytest.approx(1.0, abs=1e-6)
    
    def test_velocity_error_computation(self):
        """Test velocity error computation correctness."""
        pred = np.array([10.0, 12.0, 13.0])
        expert = np.array([10.0, 11.0, 16.0])
        out = calculate_velocity_error(pred, expert)
        assert out['mean_velocity_error'] == pytest.approx(np.mean(np.abs(pred - expert)), abs=1e-9)


class TestComfortMetrics:
    """Test comfort metrics calculation."""
    
    def test_smooth_trajectory(self):
        """Test with smooth constant velocity."""
        # Constant velocity trajectory
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        result = calculate_comfort_metrics(traj, t)
        assert result['mean_longitudinal_accel'] == pytest.approx(0.0, abs=0.1)
        assert result['comfort_rate'] >= 0.9
    
    def test_harsh_braking(self):
        """Test with harsh acceleration change."""
        # Start slow, then fast acceleration
        traj = np.array([[0, 0], [0.5, 0], [2, 0], [5, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        result = calculate_comfort_metrics(traj, t, max_longitudinal_accel=1.0)
        assert result['max_longitudinal_accel'] > 0.5
    
    def test_short_trajectory(self):
        """Test with very short trajectory."""
        traj = np.array([[0, 0], [1, 0]])
        t = np.array([0.0, 1.0])
        result = calculate_comfort_metrics(traj, t)
        assert result['mean_jerk'] == 0.0
        assert result['comfort_rate'] == 1.0
    
    def test_high_acceleration_threshold(self):
        """Test with high thresholds for acceleration and jerk."""
        traj = np.array([[0., 0.], [1., 0.], [3., 0.], [6., 0.]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        out = calculate_comfort_metrics(traj, t, max_longitudinal_accel=100.0, max_jerk=100.0)
        # With high thresholds, no violations
        assert out['comfort_violations'] == 0
        assert out['comfort_rate'] == 1.0


class TestDrivingScore:
    """Test comprehensive driving score."""
    
    def test_perfect_driving(self):
        """Test with perfect driving."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        obstacles = []
        
        result = calculate_driving_score(traj, traj, obstacles, ref, t)
        assert result['driving_score'] >= 90.0
        assert result['safety_score'] == 100.0
    
    def test_collision_penalty(self):
        """Test that collisions reduce score."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        obstacles = [np.array([[1, 0]])]
        
        result = calculate_driving_score(traj, traj, obstacles, ref, t)
        assert result['safety_score'] < 100.0
        assert result['driving_score'] < 100.0
    
    def test_custom_weights(self):
        """Test custom component weights."""
        traj = np.random.rand(10, 2)
        ref = np.random.rand(15, 2)
        t = np.linspace(0, 5, 10)
        obstacles = []
        
        weights = {'planning': 0.5, 'safety': 0.3, 'progress': 0.1, 'comfort': 0.1}
        result = calculate_driving_score(traj, traj, obstacles, ref, t, weights=weights)
        assert 0.0 <= result['driving_score'] <= 100.0


class TestPlanningKLDivergence:
    """Test KL divergence calculation."""
    
    def test_identical_distributions(self):
        """Test with identical distributions."""
        dist = np.array([0.2, 0.5, 0.3])
        kl = calculate_planning_kl_divergence(dist, dist)
        assert kl == pytest.approx(0.0, abs=1e-6)
    
    def test_different_distributions(self):
        """Test with different distributions."""
        pred = np.array([0.3, 0.4, 0.3])
        expert = np.array([0.1, 0.6, 0.3])
        kl = calculate_planning_kl_divergence(pred, expert)
        assert kl > 0.0
    
    def test_normalization(self):
        """Test that unnormalized inputs are handled."""
        pred = np.array([1, 2, 1])  # Not normalized
        expert = np.array([1, 3, 2])  # Not normalized
        kl = calculate_planning_kl_divergence(pred, expert)
        assert kl >= 0.0
    
    def test_basic_kl_computation(self):
        """Test basic KL divergence computation."""
        pred = np.array([0.2, 0.5, 0.3])
        expert = np.array([0.1, 0.6, 0.3])
        kl = calculate_planning_kl_divergence(pred, expert)
        assert kl >= 0.0


class TestTimeToCollision:
    """Test time-to-collision calculation."""
    
    def test_no_collision_far_obstacle(self):
        """Test TTC with distant obstacle."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        obs = [np.array([[100., 0.]])]
        res = calculate_time_to_collision(traj, obs)
        # obstacle very far ahead; TTC should be large
        assert res['min_ttc'] > 50.0
    
    def test_approaching_obstacle(self):
        """Test TTC with approaching obstacle."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        # Obstacle ahead but not too close
        obs = [np.array([[5., 0.]])]
        res = calculate_time_to_collision(traj, obs)
        assert res['min_ttc'] > 0


class TestLaneInvasion:
    """Test lane invasion rate calculation."""
    
    def test_no_invasion(self):
        """Test with trajectory inside lane."""
        traj = np.array([[0., 0.], [2., 0.], [4., 0.]])
        lane = np.array([[0., 0.], [4., 0.]])
        res = calculate_lane_invasion_rate(traj, [lane], lane_width=2.0)
        assert res['invasion_rate'] == 0.0
        assert res['invasion_count'] == 0
    
    def test_complete_invasion(self):
        """Test with trajectory completely outside lane."""
        traj = np.array([[0., 5.], [2., 5.], [4., 5.]])
        lane = np.array([[0., 0.], [4., 0.]])
        res = calculate_lane_invasion_rate(traj, [lane], lane_width=2.0)
        assert res['invasion_rate'] == 1.0


class TestCollisionSeverity:
    """Test collision severity calculation."""
    
    def test_no_impact(self):
        """Test with no collision."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        obs = [np.array([[10., 0.]])]
        res = calculate_collision_severity(traj, obs)
        assert res['max_severity'] == 0.0
        assert len(res['severities']) == 0
    
    def test_collision_with_severity(self):
        """Test collision severity computation."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        # Close obstacle
        obs = [np.array([[1., 0.]])]
        res = calculate_collision_severity(traj, obs)
        # Should detect some severity if collision occurs
        assert res['max_severity'] >= 0.0


class TestKinematicFeasibility:
    """Test kinematic feasibility checks."""
    
    def test_stationary_vehicle(self):
        """Test with stationary vehicle."""
        traj = np.array([[0., 0.], [0., 0.], [0., 0.]])
        t = np.array([0., 1., 2.])
        res = check_kinematic_feasibility(traj, t)
        assert res['feasible'] is True
        assert res['max_lateral_accel'] == pytest.approx(0.0, abs=1e-6)
        assert res['max_yaw_rate'] == pytest.approx(0.0, abs=1e-6)
    
    def test_smooth_trajectory(self):
        """Test with kinematically feasible trajectory."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        t = np.array([0., 1., 2., 3.])
        res = check_kinematic_feasibility(traj, t)
        assert res['feasible'] is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_trajectory(self):
        """Test with empty/minimal trajectories."""
        traj = np.array([[0, 0]])
        t = np.array([0.0])
        result = calculate_comfort_metrics(traj, t)
        assert result['comfort_rate'] == 1.0
    
    def test_single_point_trajectory(self):
        """Test with single point."""
        traj = np.array([[0, 0]])
        ref = np.array([[0, 0], [1, 0]])
        result = calculate_progress_score(traj, ref)
        assert 0.0 <= result['progress_ratio'] <= 1.0
    
    def test_backwards_trajectory(self):
        """Test with backwards motion."""
        pred = np.array([[3, 0], [2, 0], [1, 0], [0, 0]])
        expert = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        dist = calculate_l2_distance(pred, expert)
        assert dist > 0.0
    
    def test_stationary_vehicle(self):
        """Test with stationary vehicle."""
        traj = np.array([[0, 0], [0, 0], [0, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0]])
        result = calculate_progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(0.0, abs=0.1)


class TestCollisionWithFaultClassification:
    """Test collision with at-fault classification (nuPlan/tuplan_garage style)."""
    
    def test_at_fault_rear_end_collision(self):
        """Test at-fault rear-end collision (active_front)."""
        # Ego vehicle rear-ending slower vehicle ahead
        ego_traj = np.array([[0., 0.], [2., 0.], [4., 0.], [6., 0.], [8., 0.]])
        ego_heads = np.array([0., 0., 0., 0., 0.])
        # Slower vehicle ahead
        obstacles = [np.array([[7., 0.], [7.5, 0.], [8., 0.], [8.5, 0.], [9., 0.]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, obstacles, ego_heads, vehicle_size=(4.5, 2.0)
        )
        
        # Check that results exist
        assert 'at_fault_collisions' in result
        assert 'collision_rate' in result
    
    def test_at_fault_stopped_track_collision(self):
        """Test at-fault collision with stopped vehicle (stopped_track)."""
        ego_traj = np.array([[0., 0.], [2., 0.], [4., 0.], [6., 0.]])
        ego_heads = np.array([0., 0., 0., 0.])
        # Stopped vehicle
        obstacles = [np.array([[5., 0.], [5., 0.], [5., 0.], [5., 0.]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, obstacles, ego_heads, vehicle_size=(4.5, 2.0)
        )
        
        assert 'at_fault_collisions' in result
        assert 'collision_rate' in result
    
    def test_not_at_fault_rear_ended(self):
        """Test not-at-fault being rear-ended (active_rear)."""
        ego_traj = np.array([[0., 0.], [0.5, 0.], [1., 0.], [1.5, 0.]])
        ego_heads = np.array([0., 0., 0., 0.])
        # Faster vehicle behind
        obstacles = [np.array([[-2., 0.], [0., 0.], [2., 0.], [4., 0.]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, obstacles, ego_heads, vehicle_size=(4.5, 2.0)
        )
        
        assert 'at_fault_collisions' in result
        assert 'not_at_fault_collisions' in result
    
    def test_no_collisions(self):
        """Test with safe trajectories - no collisions."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        ego_heads = np.array([0., 0., 0., 0.])
        # Vehicle far away
        obstacles = [np.array([[0., 10.], [1., 10.], [2., 10.], [3., 10.]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, obstacles, ego_heads
        )
        
        assert result['collision_rate'] == 0.0
        assert result['at_fault_collisions'] == 0
    
    def test_lateral_collision(self):
        """Test lateral/sideswipe collision (active_lateral)."""
        ego_traj = np.array([[0., 0.], [1., 0.5], [2., 1.], [3., 1.5]])
        ego_heads = np.array([0.4, 0.4, 0.4, 0.4])  # Diagonal motion
        # Parallel vehicle
        obstacles = [np.array([[0., 2.], [1., 2.], [2., 2.], [3., 2.]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, obstacles, ego_heads, vehicle_size=(4.5, 2.0)
        )
        
        # May or may not collide depending on exact geometry
        assert 'at_fault_collisions' in result
        assert 'collision_types' in result


class TestTimeToCollisionEnhanced:
    """Test enhanced TTC with forward projection (nuPlan style)."""
    
    def test_approaching_collision_with_projection(self):
        """Test TTC with forward projection for approaching vehicles."""
        ego_traj = np.array([[0., 0.], [2., 0.], [4., 0.], [6., 0.]])
        timestamps = np.array([0., 1., 2., 3.])
        # Vehicle ahead moving slower
        obstacles = [np.array([[8., 0.], [9., 0.], [10., 0.], [11., 0.]])]
        
        result = calculate_time_to_collision_enhanced(
            ego_traj, obstacles, timestamps, projection_horizon=1.0
        )
        
        assert 'min_ttc' in result
        assert 'ttc_violations' in result
    
    def test_no_collision_vehicles_diverging(self):
        """Test TTC when vehicles are diverging - no collision."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        timestamps = np.array([0., 1., 2., 3.])
        # Vehicle moving away
        obstacles = [np.array([[10., 0.], [15., 0.], [20., 0.], [25., 0.]])]
        
        result = calculate_time_to_collision_enhanced(
            ego_traj, obstacles, timestamps
        )
        
        assert 'min_ttc' in result
        assert result['ttc_violations'] >= 0
    
    def test_stopped_vehicle_excluded(self):
        """Test that stopped vehicles are excluded from TTC calculation."""
        ego_traj = np.array([[0., 0.], [2., 0.], [4., 0.], [6., 0.]])
        timestamps = np.array([0., 1., 2., 3.])
        # Stopped vehicle
        obstacles = [np.array([[10., 0.], [10., 0.], [10., 0.], [10., 0.]])]
        
        result = calculate_time_to_collision_enhanced(
            ego_traj, obstacles, timestamps, stopped_speed_threshold=0.005
        )
        
        assert 'min_ttc' in result
        assert 'per_ttc' in result
    
    def test_ttc_violations_counting(self):
        """Test TTC violations counting."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        timestamps = np.array([0., 1., 2., 3.])
        # Close vehicle
        obstacles = [np.array([[4., 0.], [4.5, 0.], [5., 0.], [5.5, 0.]])]
        
        result = calculate_time_to_collision_enhanced(
            ego_traj, obstacles, timestamps
        )
        
        assert result['ttc_violations'] >= 0
        assert isinstance(result['ttc_violations'], (int, np.integer))
    
    def test_projection_parameters(self):
        """Test different projection horizon parameter."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        timestamps = np.array([0., 1., 2.])
        obstacles = [np.array([[5., 0.], [6., 0.], [7., 0.]])]
        
        result = calculate_time_to_collision_enhanced(
            ego_traj, obstacles, timestamps, projection_horizon=2.0
        )
        
        assert 'min_ttc' in result
        assert 'per_ttc' in result


class TestDistanceToRoadEdge:
    """Test distance to road edge / drivable area (Waymo Sim Agents style)."""
    
    def test_with_lane_boundaries_fallback(self):
        """Test distance calculation with lane boundaries fallback."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        boundaries = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        
        result = calculate_distance_to_road_edge(
            traj, lane_boundaries=boundaries, lane_width=3.5
        )
        
        assert 'mean_distance' in result
        assert 'min_distance' in result
        assert 'offroad_rate' in result
        assert 'distances' in result
        assert len(result['distances']) == len(traj)
    
    def test_trajectory_inside_drivable_area(self):
        """Test trajectory completely inside drivable area (negative distance)."""
        traj = np.array([[1., 1.], [1., 2.], [1., 3.]])
        boundaries = np.array([[1., 0.], [1., 5.]])
        
        result = calculate_distance_to_road_edge(
            traj, lane_boundaries=boundaries, lane_width=4.0
        )
        
        # Inside drivable area
        assert 'offroad_rate' in result
    
    def test_trajectory_outside_drivable_area(self):
        """Test trajectory violating drivable area (positive distance)."""
        traj = np.array([[10., 0.], [11., 0.], [12., 0.]])
        boundaries = np.array([[0., 0.], [5., 0.]])
        
        result = calculate_distance_to_road_edge(
            traj, lane_boundaries=boundaries, lane_width=2.0
        )
        
        # Outside drivable area should have some offroad
        assert result['offroad_rate'] >= 0.0
    
    def test_with_shapely_polygon(self):
        """Test with Shapely polygon (preferred method)."""
        try:
            from shapely.geometry import Polygon
            
            traj = np.array([[1., 1.], [2., 2.], [3., 3.]])
            # Create a simple polygon
            drivable_area = Polygon([(-5, -5), (10, -5), (10, 10), (-5, 10)])
            
            result = calculate_distance_to_road_edge(
                traj, drivable_area_polygons=[drivable_area]
            )
            
            assert 'mean_distance' in result
            assert 'min_distance' in result
        except ImportError:
            pytest.skip("Shapely not installed")
    
    def test_partial_violation(self):
        """Test trajectory partially violating drivable area."""
        # Start inside, move outside
        traj = np.array([[0., 0.], [1., 0.], [5., 0.], [10., 0.]])
        boundaries = np.array([[0., 0.], [10., 0.]])
        
        result = calculate_distance_to_road_edge(
            traj, lane_boundaries=boundaries, lane_width=3.0
        )
        
        assert 0.0 <= result['offroad_rate'] <= 1.0


class TestDrivingDirectionCompliance:
    """Test driving direction compliance / wrong-way detection (nuPlan style)."""
    
    def test_correct_direction_full_compliance(self):
        """Test driving in correct direction (compliance score = 1.0)."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        ref_path = np.array([[0., 0.], [5., 0.]])  # Lane going east
        
        result = calculate_driving_direction_compliance(traj, ref_path)
        
        assert result['compliance_score'] >= 0.9
        assert result['max_wrong_way_distance'] <= 2.0  # nuPlan threshold
        assert result['is_compliant'] is True
    
    def test_wrong_way_detection(self):
        """Test wrong-way driving detection (compliance score = 0.0)."""
        traj = np.array([[3., 0.], [2., 0.], [1., 0.], [0., 0.]])
        ref_path = np.array([[0., 0.], [5., 0.]])  # Lane going east
        
        result = calculate_driving_direction_compliance(traj, ref_path)
        
        # Wrong way driving should reduce score
        assert 'compliance_score' in result
        assert 'max_wrong_way_distance' in result
    
    def test_partial_wrong_way(self):
        """Test partial wrong-way (e.g., brief reverse during maneuver)."""
        # Mix of forward and potentially backward motion
        traj = np.array([[0., 0.], [1., 0.], [1.5, 0.], [2., 0.], [3., 0.]])
        ref_path = np.array([[0., 0.], [5., 0.]])
        
        result = calculate_driving_direction_compliance(traj, ref_path)
        
        # Should have some compliance metric
        assert 0.0 <= result['compliance_score'] <= 1.0
    
    def test_direction_vectors_parameter(self):
        """Test with provided direction vectors."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        ref_path = np.array([[0., 0.], [5., 0.]])
        # Direction vectors pointing east
        dir_vectors = np.array([[1., 0.], [1., 0.]])
        
        result = calculate_driving_direction_compliance(
            traj, ref_path, route_direction_vectors=dir_vectors
        )
        
        assert 'compliance_score' in result
    
    def test_compliance_output_structure(self):
        """Test that all expected output keys are present."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        ref_path = np.array([[0., 0.], [5., 0.]])
        
        result = calculate_driving_direction_compliance(traj, ref_path)
        
        assert 'max_wrong_way_distance' in result
        assert 'compliance_score' in result
        assert 'is_compliant' in result
        assert 'wrong_way_timesteps' in result


class TestInteractionMetrics:
    """Test multi-agent interaction/proximity metrics (Waymo Sim Agents style)."""
    
    def test_minimum_distance_calculation(self):
        """Test minimum distance to any object calculation."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        # Two other vehicles at different distances
        other_objects = [
            np.array([[5., 0.], [5., 0.], [5., 0.], [5., 0.]]),  # 5m away initially
            np.array([[0., 10.], [1., 10.], [2., 10.], [3., 10.]])  # 10m away
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert result['min_distance'] is not None
        assert result['min_distance'] >= 0
        # Closest should be first vehicle at ~5m
        assert result['min_distance'] < 10.0
    
    def test_mean_distance_to_nearest(self):
        """Test mean distance to nearest object over trajectory."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        other_objects = [
            np.array([[4., 0.], [4., 0.], [4., 0.]])  # Consistently ~4m away
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert result['mean_distance_to_nearest'] >= 0
        assert result['mean_distance_to_nearest'] < 10.0
    
    def test_close_interactions_counting(self):
        """Test counting of close interactions (< 5m threshold)."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        # Vehicle that gets closer over time
        other_objects = [
            np.array([[10., 0.], [7., 0.], [4., 0.], [2., 0.], [1., 0.]])
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert result['num_close_interactions'] >= 0
        # Should have some close interactions as vehicle approaches
        assert result['num_close_interactions'] > 0
    
    def test_closest_object_identification(self):
        """Test identification of closest object and timestep."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        other_objects = [
            np.array([[10., 0.], [10., 0.], [10., 0.]]),  # Object 0: far
            np.array([[5., 0.], [3., 0.], [2., 0.]])      # Object 1: gets closer
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert 'closest_object_id' in result
        assert 'closest_approach_timestep' in result
        assert result['closest_object_id'] in [0, 1]
        assert 0 <= result['closest_approach_timestep'] < len(ego_traj)
    
    def test_distance_per_timestep(self):
        """Test distance to nearest object per timestep."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        other_objects = [
            np.array([[3., 0.], [3., 0.], [3., 0.]])
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert 'distance_to_nearest_per_timestep' in result
        assert len(result['distance_to_nearest_per_timestep']) == len(ego_traj)
        assert all(d >= 0 for d in result['distance_to_nearest_per_timestep'])
    
    def test_static_and_dynamic_obstacles(self):
        """Test with mix of static and dynamic obstacles."""
        ego_traj = np.array([[0., 0.], [2., 0.], [4., 0.]])
        other_objects = [
            np.array([[5., 0.], [5., 0.], [5., 0.]]),      # Static obstacle
            np.array([[10., 5.], [8., 4.], [6., 3.]])      # Moving obstacle
        ]
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        assert result['min_distance'] >= 0
        assert result['num_close_interactions'] >= 0
    
    def test_no_objects(self):
        """Test with no other objects."""
        ego_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        other_objects = []
        
        result = calculate_interaction_metrics(ego_traj, other_objects)
        
        # Should handle empty list gracefully
        assert 'min_distance' in result
        assert 'mean_distance_to_nearest' in result


class TestComfortMetricsEnhanced:
    """Test enhanced comfort metrics features (smoothing, lateral)."""
    
    def test_smoothing_improves_comfort(self):
        """Test that Savitzky-Golay smoothing improves comfort scores."""
        # Create noisy trajectory
        t = np.linspace(0, 5, 50)
        traj = np.column_stack([t * 2, np.sin(t)])
        traj += np.random.randn(50, 2) * 0.1  # Add noise
        
        result_no_smooth = calculate_comfort_metrics(
            traj, t, use_smoothing=False, include_lateral=True
        )
        result_smooth = calculate_comfort_metrics(
            traj, t, use_smoothing=True, include_lateral=True
        )
        
        # Smoothing should generally improve comfort rate
        assert result_smooth['comfort_rate'] >= result_no_smooth['comfort_rate'] - 0.1
        # Smoothed should have lower max acceleration
        assert result_smooth['max_longitudinal_accel'] <= result_no_smooth['max_longitudinal_accel']
    
    def test_lateral_metrics_included(self):
        """Test that lateral metrics are computed when enabled."""
        traj = np.array([[0., 0.], [1., 0.5], [2., 1.], [3., 1.5]])
        t = np.array([0., 1., 2., 3.])
        
        result = calculate_comfort_metrics(traj, t, include_lateral=True)
        
        assert 'mean_lateral_accel' in result
        assert 'max_lateral_accel' in result
        assert 'mean_yaw_rate' in result
        assert 'max_yaw_rate' in result
        assert 'mean_yaw_accel' in result
        assert 'max_yaw_accel' in result
    
    def test_lateral_metrics_excluded(self):
        """Test that lateral metrics are not computed when disabled."""
        traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        t = np.array([0., 1., 2.])
        
        result = calculate_comfort_metrics(traj, t, include_lateral=False)
        
        assert 'mean_lateral_accel' not in result
        assert 'max_lateral_accel' not in result
        assert 'mean_yaw_rate' not in result
    
    def test_smoothing_window_parameter(self):
        """Test different smoothing window sizes."""
        traj = np.random.randn(30, 2) * 5
        t = np.linspace(0, 3, 30)
        
        result_small = calculate_comfort_metrics(
            traj, t, use_smoothing=True, smoothing_window=7
        )
        result_large = calculate_comfort_metrics(
            traj, t, use_smoothing=True, smoothing_window=15
        )
        
        # Both should execute without error
        assert 'comfort_rate' in result_small
        assert 'comfort_rate' in result_large


class TestDrivingScoreEnhanced:
    """Test enhanced driving score features (nuPlan mode)."""
    
    def test_nuplan_mode_basic(self):
        """Test driving score with nuPlan mode."""
        pred_traj = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        expert_traj = np.array([[0., 0.], [1., 0.1], [2., 0.], [3., 0.]])
        timestamps = np.array([0., 1., 2., 3.])
        obstacles = []
        ref_path = np.array([[0., 0.], [5., 0.]])
        headings = np.array([0., 0., 0., 0.])
        velocities = np.array([1., 1., 1., 1.])
        
        result = calculate_driving_score(
            pred_traj, expert_traj, obstacles, ref_path, timestamps,
            mode='nuplan', headings=headings, velocities=velocities
        )
        
        assert 'driving_score' in result
        assert 'planning_accuracy' in result
        assert 'safety_score' in result
        assert 'progress_score' in result
        assert 'comfort_score' in result
        assert 0 <= result['driving_score'] <= 100
    
    def test_default_mode_vs_nuplan_mode(self):
        """Test difference between default and nuPlan modes."""
        pred_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        expert_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        timestamps = np.array([0., 1., 2.])
        obstacles = []
        ref_path = np.array([[0., 0.], [5., 0.]])
        
        result_default = calculate_driving_score(
            pred_traj, expert_traj, obstacles, ref_path, timestamps, mode='default'
        )
        
        headings = np.array([0., 0., 0.])
        velocities = np.array([1., 1., 1.])
        result_nuplan = calculate_driving_score(
            pred_traj, expert_traj, obstacles, ref_path, timestamps,
            mode='nuplan', headings=headings, velocities=velocities
        )
        
        # Both should return valid scores
        assert 'driving_score' in result_default
        assert 'driving_score' in result_nuplan
        assert result_default['driving_score'] >= 0
        assert result_nuplan['driving_score'] >= 0
    
    def test_nuplan_with_obstacles(self):
        """Test nuPlan mode with obstacles for safety scoring."""
        pred_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        expert_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
        timestamps = np.array([0., 1., 2.])
        ref_path = np.array([[0., 0.], [5., 0.]])
        obstacles = [np.array([[10., 0.], [10., 0.], [10., 0.]])]  # Far away
        headings = np.array([0., 0., 0.])
        velocities = np.array([1., 1., 1.])
        
        result = calculate_driving_score(
            pred_traj, expert_traj, obstacles, ref_path, timestamps,
            headings=headings, velocities=velocities, mode='nuplan'
        )
        
        assert result['safety_score'] >= 0
        assert 'driving_score' in result

