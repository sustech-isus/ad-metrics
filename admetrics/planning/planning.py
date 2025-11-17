"""
End-to-End Planning Metrics for Autonomous Driving.

This module provides metrics for evaluating end-to-end autonomous driving models
that directly output planned trajectories or control commands from sensor inputs.
These metrics are commonly used in benchmarks like nuPlan, CARLA, and Waymo Open Dataset.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from scipy.stats import entropy


def calculate_l2_distance(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate L2 distance between predicted and expert trajectories.
    
    This is the primary metric for many end-to-end planning benchmarks,
    measuring the geometric distance between planned and expert paths.
    
    Args:
        predicted_trajectory: Predicted trajectory, shape (T, 2) or (T, 3) for [x, y] or [x, y, z]
        expert_trajectory: Expert/reference trajectory, same shape as predicted
        weights: Optional temporal weights, shape (T,). Later timesteps often weighted higher.
    
    Returns:
        Average L2 distance across all timesteps
    
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0]])
        >>> expert = np.array([[0, 0], [1, 1], [2, 2]])
    >>> dist = calculate_l2_distance(pred, expert)
    """
    if predicted_trajectory.shape != expert_trajectory.shape:
        raise ValueError(
            f"Shape mismatch: pred {predicted_trajectory.shape} vs expert {expert_trajectory.shape}"
        )
    
    # Calculate Euclidean distance at each timestep
    distances = np.linalg.norm(predicted_trajectory - expert_trajectory, axis=1)
    
    if weights is not None:
        if len(weights) != len(distances):
            raise ValueError(f"Weights length {len(weights)} != trajectory length {len(distances)}")
        distances = distances * weights
        return float(np.sum(distances) / np.sum(weights))
    
    return float(np.mean(distances))


def calculate_collision_rate(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    vehicle_size: Tuple[float, float] = (4.5, 2.0),
    obstacle_sizes: Optional[List[Tuple[float, float]]] = None,
    safety_margin: float = 0.0
) -> Dict[str, Union[float, int, List[int]]]:
    """
    Calculate collision rate for a planned trajectory.
    
    Checks if the ego vehicle's planned trajectory collides with static or
    dynamic obstacles. Critical safety metric for end-to-end driving.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y] positions
        obstacles: List of obstacle trajectories, each shape (T, 2) or (1, 2) for static
        vehicle_size: Ego vehicle (length, width) in meters
        obstacle_sizes: List of obstacle sizes, defaults to same as vehicle_size
        safety_margin: Additional safety buffer in meters
    
    Returns:
        Dictionary containing:
            - 'collision_rate': Fraction of timesteps with collision
            - 'num_collisions': Total number of collision timesteps
            - 'collision_timesteps': List of timestep indices with collisions
            - 'first_collision': Timestep of first collision (None if no collision)
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> obs = [np.array([[1.5, 0], [1.5, 0], [1.5, 0]])]
        >>> result = collision_rate(traj, obs)
    """
    T = len(trajectory)
    collision_timesteps = []
    
    # Default obstacle sizes
    if obstacle_sizes is None:
        obstacle_sizes = [vehicle_size] * len(obstacles)
    
    ego_length, ego_width = vehicle_size
    ego_radius = np.sqrt((ego_length/2)**2 + (ego_width/2)**2) + safety_margin
    
    for t in range(T):
        ego_pos = trajectory[t]
        
        for obs_traj, (obs_length, obs_width) in zip(obstacles, obstacle_sizes):
            # Handle static obstacles (single position)
            if len(obs_traj.shape) == 1 or len(obs_traj) == 1:
                obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
            else:
                # Dynamic obstacle - get position at time t
                if t < len(obs_traj):
                    obs_pos = obs_traj[t]
                else:
                    continue  # Obstacle trajectory ended
            
            obs_radius = np.sqrt((obs_length/2)**2 + (obs_width/2)**2) + safety_margin
            
            # Check circular collision
            distance = np.linalg.norm(ego_pos - obs_pos)
            if distance < (ego_radius + obs_radius):
                collision_timesteps.append(t)
                break  # Count each timestep only once
    
    num_collisions = len(collision_timesteps)
    collision_rate_val = num_collisions / T if T > 0 else 0.0
    first_collision = collision_timesteps[0] if collision_timesteps else None
    
    return {
        'collision_rate': float(collision_rate_val),
        'num_collisions': num_collisions,
        'collision_timesteps': collision_timesteps,
        'first_collision': first_collision
    }


def calculate_collision_with_fault_classification(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    ego_headings: np.ndarray,
    obstacle_headings: Optional[List[np.ndarray]] = None,
    obstacle_types: Optional[List[str]] = None,
    vehicle_size: Tuple[float, float] = (4.5, 2.0),
    obstacle_sizes: Optional[List[Tuple[float, float]]] = None,
    safety_margin: float = 0.0
) -> Dict[str, Union[float, int, List]]:
    """
    Calculate collision rate with fault classification (at-fault vs not-at-fault).
    
    Implements nuPlan-style collision classification to distinguish between
    collisions where ego is at fault vs being hit by others.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y] positions
        obstacles: List of obstacle trajectories, each shape (T, 2) or (1, 2) for static
        ego_headings: Ego heading angles in radians, shape (T,)
        obstacle_headings: List of obstacle headings for each obstacle, shape (T,) each
        obstacle_types: List of obstacle types: 'agent' (vehicle/pedestrian) or 'static'
        vehicle_size: Ego vehicle (length, width) in meters
        obstacle_sizes: List of obstacle sizes, defaults to same as vehicle_size
        safety_margin: Additional safety buffer in meters
    
    Returns:
        Dictionary containing:
            - 'collision_rate': Overall collision rate
            - 'at_fault_collision_rate': Rate of at-fault collisions
            - 'not_at_fault_collision_rate': Rate of not-at-fault collisions
            - 'at_fault_collisions': Number of at-fault collisions
            - 'not_at_fault_collisions': Number of not-at-fault collisions
            - 'collision_types': List of collision type strings per collision
            - 'first_at_fault_collision': Timestep of first at-fault collision
    
    Collision Types:
        - 'active_front': Ego rear-ends obstacle (AT FAULT)
        - 'stopped_track': Ego hits stationary obstacle (AT FAULT)
        - 'active_lateral': Side collision (AT FAULT if ego in wrong area)
        - 'active_rear': Ego is rear-ended (NOT AT FAULT)
        - 'passive_lateral': Ego is side-swiped in lane (NOT AT FAULT)
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> headings = np.array([0.0, 0.0, 0.0])
        >>> obs = [np.array([[1.5, 0], [1.5, 0], [1.5, 0]])]
        >>> result = calculate_collision_with_fault_classification(traj, obs, headings)
    """
    T = len(trajectory)
    collision_timesteps = []
    collision_types = []
    at_fault_timesteps = []
    not_at_fault_timesteps = []
    
    # Default obstacle sizes and types
    if obstacle_sizes is None:
        obstacle_sizes = [vehicle_size] * len(obstacles)
    if obstacle_types is None:
        obstacle_types = ['agent'] * len(obstacles)
    if obstacle_headings is None:
        obstacle_headings = [np.zeros(T if len(obs.shape) > 1 and len(obs) == T else 1) 
                            for obs in obstacles]
    
    ego_length, ego_width = vehicle_size
    ego_radius = np.sqrt((ego_length/2)**2 + (ego_width/2)**2) + safety_margin
    
    for t in range(T):
        ego_pos = trajectory[t]
        ego_heading = ego_headings[t]
        
        for obs_idx, (obs_traj, (obs_length, obs_width)) in enumerate(zip(obstacles, obstacle_sizes)):
            # Handle static obstacles (single position)
            is_static = len(obs_traj.shape) == 1 or len(obs_traj) == 1
            if is_static:
                obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
                obs_heading = obstacle_headings[obs_idx][0] if len(obstacle_headings[obs_idx]) > 0 else 0.0
                obs_velocity = 0.0
            else:
                # Dynamic obstacle - get position at time t
                if t < len(obs_traj):
                    obs_pos = obs_traj[t]
                    obs_heading = obstacle_headings[obs_idx][t] if t < len(obstacle_headings[obs_idx]) else 0.0
                    # Estimate velocity
                    if t > 0 and t < len(obs_traj):
                        obs_velocity = np.linalg.norm(obs_traj[t] - obs_traj[t-1])
                    else:
                        obs_velocity = 0.0
                else:
                    continue  # Obstacle trajectory ended
            
            obs_radius = np.sqrt((obs_length/2)**2 + (obs_width/2)**2) + safety_margin
            
            # Check circular collision
            distance = np.linalg.norm(ego_pos - obs_pos)
            if distance < (ego_radius + obs_radius):
                # Classify collision type
                # Compute relative position in ego's frame
                to_obs = obs_pos - ego_pos
                angle_to_obs = np.arctan2(to_obs[1], to_obs[0])
                relative_angle = angle_to_obs - ego_heading
                # Normalize to [-pi, pi]
                relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
                
                # Compute relative heading
                heading_diff = obs_heading - ego_heading
                heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                
                # Classify based on relative angle and velocity
                STOPPED_THRESHOLD = 0.1  # m/s
                FRONT_ANGLE = np.pi / 3  # 60 degrees
                REAR_ANGLE = 2 * np.pi / 3  # 120 degrees
                
                collision_type = 'unknown'
                is_at_fault = False
                
                if is_static or obs_velocity < STOPPED_THRESHOLD:
                    # Hitting stationary object is at-fault
                    collision_type = 'stopped_track'
                    is_at_fault = True
                elif abs(relative_angle) < FRONT_ANGLE:
                    # Obstacle in front - likely rear-ending (at-fault)
                    collision_type = 'active_front'
                    is_at_fault = True
                elif abs(relative_angle) > REAR_ANGLE:
                    # Obstacle behind - being rear-ended (not at-fault)
                    collision_type = 'active_rear'
                    is_at_fault = False
                else:
                    # Lateral collision - default to at-fault for safety
                    # In full implementation, would check lane position
                    collision_type = 'active_lateral'
                    is_at_fault = True
                
                collision_timesteps.append(t)
                collision_types.append(collision_type)
                
                if is_at_fault:
                    at_fault_timesteps.append(t)
                else:
                    not_at_fault_timesteps.append(t)
                
                break  # Count each timestep only once
    
    num_collisions = len(collision_timesteps)
    num_at_fault = len(at_fault_timesteps)
    num_not_at_fault = len(not_at_fault_timesteps)
    
    collision_rate_val = num_collisions / T if T > 0 else 0.0
    at_fault_rate = num_at_fault / T if T > 0 else 0.0
    not_at_fault_rate = num_not_at_fault / T if T > 0 else 0.0
    
    first_at_fault = at_fault_timesteps[0] if at_fault_timesteps else None
    
    return {
        'collision_rate': float(collision_rate_val),
        'at_fault_collision_rate': float(at_fault_rate),
        'not_at_fault_collision_rate': float(not_at_fault_rate),
        'num_collisions': num_collisions,
        'at_fault_collisions': num_at_fault,
        'not_at_fault_collisions': num_not_at_fault,
        'collision_timesteps': collision_timesteps,
        'collision_types': collision_types,
        'first_at_fault_collision': first_at_fault,
        'at_fault_timesteps': at_fault_timesteps,
        'not_at_fault_timesteps': not_at_fault_timesteps
    }


def calculate_progress_score(
    trajectory: np.ndarray,
    reference_path: np.ndarray,
    goal_position: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate progress along the reference path.
    
    Measures how far the ego vehicle advances along the intended route,
    important for evaluating task completion in end-to-end driving.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        reference_path: Reference centerline/path, shape (N, 2)
        goal_position: Optional goal position, shape (2,). If None, uses last point of reference
    
    Returns:
        Dictionary containing:
            - 'progress': Distance traveled along reference path
            - 'progress_ratio': Fraction of total path completed [0, 1]
            - 'goal_reached': Whether goal was reached (within 2m)
            - 'final_distance_to_goal': Distance from final position to goal
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> ref = np.array([[0, 0], [5, 0], [10, 0]])
        >>> score = progress_score(traj, ref)
    """
    if goal_position is None:
        goal_position = reference_path[-1]
    
    # Calculate total reference path length
    path_segments = np.diff(reference_path, axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=1)
    total_path_length = np.sum(path_lengths)
    
    # Find closest point on reference path to final trajectory position
    final_pos = trajectory[-1]
    distances_to_path = np.linalg.norm(reference_path - final_pos, axis=1)
    closest_idx = np.argmin(distances_to_path)
    
    # Calculate progress as distance along path to closest point
    if closest_idx == 0:
        progress = 0.0
    else:
        progress = np.sum(path_lengths[:closest_idx])
        # Add partial distance to closest point
        if closest_idx < len(reference_path) - 1:
            prev_point = reference_path[closest_idx - 1]
            next_point = reference_path[closest_idx]
            segment_vec = next_point - prev_point
            to_final = final_pos - prev_point
            # Project onto segment
            projection = np.dot(to_final, segment_vec) / (np.linalg.norm(segment_vec) + 1e-8)
            projection = np.clip(projection, 0, np.linalg.norm(segment_vec))
            progress += projection
    
    progress_ratio = progress / total_path_length if total_path_length > 0 else 0.0
    
    # Check if goal reached (within 2 meters)
    final_distance_to_goal = np.linalg.norm(final_pos - goal_position)
    goal_reached = final_distance_to_goal < 2.0
    
    return {
        'progress': float(progress),
        'progress_ratio': float(np.clip(progress_ratio, 0.0, 1.0)),
        'goal_reached': bool(goal_reached),
        'final_distance_to_goal': float(final_distance_to_goal)
    }




def calculate_route_completion(
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    completion_radius: float = 2.0
) -> Dict[str, Union[float, int, List[bool]]]:
    """
    Calculate route completion based on waypoint passing.
    
    Evaluates whether the vehicle successfully passes through required waypoints,
    used in navigation tasks and long-range planning evaluation.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        waypoints: Required waypoints to pass, shape (N, 2)
        completion_radius: Distance threshold to consider waypoint reached (meters)
    
    Returns:
        Dictionary containing:
            - 'completion_rate': Fraction of waypoints reached [0, 1]
            - 'num_waypoints_reached': Number of waypoints successfully reached
            - 'waypoint_status': Boolean list indicating which waypoints were reached
            - 'total_waypoints': Total number of waypoints
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        >>> waypoints = np.array([[1, 0], [2, 0], [5, 0]])
        >>> result = route_completion(traj, waypoints, completion_radius=0.5)
    """
    # Enforce ordered waypoint passing: iterate through trajectory and mark waypoints
    num_waypoints = len(waypoints)
    waypoint_status: List[bool] = [False] * num_waypoints
    if num_waypoints == 0:
        return {
            'completion_rate': 0.0,
            'num_waypoints_reached': 0,
            'waypoint_status': waypoint_status,
            'total_waypoints': 0
        }

    current_idx = 0
    for pos in trajectory:
        if current_idx >= num_waypoints:
            break
        dist = np.linalg.norm(pos - waypoints[current_idx])
        if dist <= completion_radius:
            waypoint_status[current_idx] = True
            current_idx += 1

    num_reached = sum(waypoint_status)
    completion_rate_val = num_reached / num_waypoints
    
    return {
        'completion_rate': float(completion_rate_val),
        'num_waypoints_reached': num_reached,
        'waypoint_status': waypoint_status,
        'total_waypoints': num_waypoints
    }


def average_displacement_error_planning(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    timestep_weights: Optional[Union[str, np.ndarray]] = 'linear',
    alpha: float = 1.0,
    horizons: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate Average Displacement Error (ADE) for planning.
    
    Similar to trajectory prediction ADE but applied to planning/control output.
    Often used with temporal weighting to emphasize long-term planning accuracy.
    
    Args:
        predicted_trajectory: Planned trajectory, shape (T, 2) or (T, 3)
        expert_trajectory: Expert trajectory, same shape
        timestep_weights: Weighting scheme - 'uniform', 'linear', 'exponential', or None
    
    Returns:
        Dictionary containing:
            - 'ADE': Average displacement error
            - 'FDE': Final displacement error
            - 'weighted_ADE': ADE with temporal weighting (if weights specified)
    


    Example:
        >>> pred = np.random.rand(10, 2)
        >>> expert = np.random.rand(10, 2)
    """
    if predicted_trajectory.shape != expert_trajectory.shape:
        raise ValueError("Trajectory shapes must match")
    
    T = len(predicted_trajectory)
    
    # Calculate per-timestep errors
    displacements = np.linalg.norm(predicted_trajectory - expert_trajectory, axis=1)
    
    ade = float(np.mean(displacements))
    fde = float(displacements[-1])
    
    # Calculate weighted ADE
    if isinstance(timestep_weights, np.ndarray):
        if len(timestep_weights) != T:
            raise ValueError("weights array length must equal trajectory length")
        weights = timestep_weights
    elif timestep_weights == 'linear':
        weights = np.linspace(1.0, 2.0, T)
    elif timestep_weights == 'exponential':
        # alpha controls the exponential rate; higher alpha weights later timesteps more
        weights = np.exp(alpha * np.linspace(0, 1, T))
    elif timestep_weights == 'uniform' or timestep_weights is None:
        weights = np.ones(T)
    else:
        weights = np.ones(T)

    weighted_ade = float(np.sum(displacements * weights) / np.sum(weights))

    # Multi-horizon ADE/FDE if requested (horizons in timesteps)
    multi_horizon: Dict[str, float] = {}
    if horizons is not None:
        for h in horizons:
            if h <= 0 or h > T:
                continue
            dh = float(np.mean(displacements[:h]))
            fh = float(displacements[h - 1])
            multi_horizon[f'ADE_{h}'] = dh
            multi_horizon[f'FDE_{h}'] = fh
    
    out = {
        'ADE': ade,
        'FDE': fde,
        'weighted_ADE': weighted_ade
    }
    out.update(multi_horizon)
    return out


def calculate_lateral_deviation(
    trajectory: np.ndarray,
    reference_path: np.ndarray
) -> Dict[str, float]:
    """
    Calculate lateral deviation from reference path.
    
    Measures cross-track error, important for lane-keeping and path-following
    performance in end-to-end driving systems.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        reference_path: Reference centerline, shape (N, 2)
    
    Returns:
        Dictionary containing:
            - 'mean_lateral_error': Average lateral deviation (meters)
            - 'max_lateral_error': Maximum lateral deviation (meters)
            - 'std_lateral_error': Standard deviation of lateral deviation
            - 'lane_keeping_rate': Fraction of time within lane (assuming 1.75m from center)
    
    Example:
        >>> traj = np.array([[0, 0.5], [1, 0.3], [2, -0.2]])
        >>> ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        >>> deviation = lateral_deviation(traj, ref)
    """
    # Compute perpendicular distance from each trajectory point to nearest path segment
    def point_to_segment_distance(p, a, b):
        # projection of p onto segment ab
        ap = p - a
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 == 0:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab_len2
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    lateral_errors = []
    # iterate over each point and all segments
    for p in trajectory:
        min_dist = float('inf')
        for i in range(len(reference_path) - 1):
            a = reference_path[i]
            b = reference_path[i + 1]
            d = point_to_segment_distance(p, a, b)
            if d < min_dist:
                min_dist = d
        # if only single reference point, fallback
        if np.isinf(min_dist):
            min_dist = np.linalg.norm(p - reference_path[0])
        lateral_errors.append(min_dist)
    
    lateral_errors = np.array(lateral_errors)
    
    # Lane keeping rate (assuming 3.5m lane width, so 1.75m from center)
    lane_width_half = 1.75
    within_lane = np.sum(lateral_errors <= lane_width_half)
    lane_keeping_rate = within_lane / len(lateral_errors) if len(lateral_errors) > 0 else 0.0
    
    return {
        'mean_lateral_error': float(np.mean(lateral_errors)),
        'max_lateral_error': float(np.max(lateral_errors)),
        'std_lateral_error': float(np.std(lateral_errors)),
        'lane_keeping_rate': float(lane_keeping_rate)
    }


def calculate_heading_error(
    predicted_headings: np.ndarray,
    expert_headings: np.ndarray
) -> Dict[str, float]:
    """
    Calculate heading/yaw angle error.
    
    Measures orientation accuracy, important for vehicle control and
    trajectory smoothness in end-to-end systems.
    
    Args:
        predicted_headings: Predicted heading angles in radians, shape (T,)
        expert_headings: Expert heading angles in radians, shape (T,)
    
    Returns:
        Dictionary containing:
            - 'mean_heading_error': Average heading error (radians)
            - 'mean_heading_error_deg': Average heading error (degrees)
            - 'max_heading_error': Maximum heading error (radians)
            - 'std_heading_error': Standard deviation (radians)
    
    Example:
        >>> pred_headings = np.array([0.0, 0.1, 0.2])
        >>> expert_headings = np.array([0.0, 0.15, 0.25])
        >>> error = heading_error(pred_headings, expert_headings)
    """
    if len(predicted_headings) != len(expert_headings):
        raise ValueError("Heading arrays must have same length")
    
    # Calculate angular difference (handling wrap-around)
    diff = predicted_headings - expert_headings
    diff = np.arctan2(np.sin(diff), np.cos(diff))  # Wrap to [-π, π]
    
    errors = np.abs(diff)
    
    return {
        'mean_heading_error': float(np.mean(errors)),
        'mean_heading_error_deg': float(np.degrees(np.mean(errors))),
        'max_heading_error': float(np.max(errors)),
        'std_heading_error': float(np.std(errors))
    }


def calculate_velocity_error(
    predicted_velocities: np.ndarray,
    expert_velocities: np.ndarray
) -> Dict[str, float]:
    """
    Calculate velocity/speed error.
    
    Measures how well the planner matches desired speeds,
    important for comfort and traffic flow.
    
    Args:
        predicted_velocities: Predicted velocities in m/s, shape (T,)
        expert_velocities: Expert velocities in m/s, shape (T,)
    
    Returns:
        Dictionary containing:
            - 'mean_velocity_error': Average velocity error (m/s)
            - 'rmse_velocity': Root mean squared velocity error
            - 'max_velocity_error': Maximum velocity error
    
    Example:
        >>> pred_vel = np.array([10.0, 12.0, 15.0])
        >>> expert_vel = np.array([10.0, 13.0, 14.0])
        >>> error = velocity_error(pred_vel, expert_vel)
    """
    if len(predicted_velocities) != len(expert_velocities):
        raise ValueError("Velocity arrays must have same length")
    
    errors = np.abs(predicted_velocities - expert_velocities)
    rmse = np.sqrt(np.mean((predicted_velocities - expert_velocities)**2))
    
    return {
        'mean_velocity_error': float(np.mean(errors)),
        'rmse_velocity': float(rmse),
        'max_velocity_error': float(np.max(errors))
    }


def calculate_comfort_metrics(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    max_longitudinal_accel: float = 4.0,
    max_lateral_accel: float = 4.0,
    max_jerk: float = 4.0,
    max_yaw_rate: float = 1.0,
    max_yaw_accel: float = 1.0,
    include_lateral: bool = True,
    use_smoothing: bool = False,
    smoothing_window: int = 15,
    smoothing_order: int = 2
) -> Dict[str, Union[float, int]]:
    """
    Calculate comfort metrics for driving quality (nuPlan-compatible).
    
    Evaluates smoothness and passenger comfort through longitudinal acceleration,
    lateral acceleration, jerk, yaw rate, and other dynamic measures. 
    Updated to match nuPlan standards with both longitudinal and lateral metrics.
    
    Args:
        trajectory: Trajectory positions, shape (T, 2) for [x, y]
        timestamps: Time at each position, shape (T,)
        max_longitudinal_accel: Comfort threshold for longitudinal acceleration (m/s²), default 4.0
        max_lateral_accel: Comfort threshold for lateral acceleration (m/s²), default 4.0
        max_jerk: Comfort threshold for jerk (m/s³), default 4.0
        max_yaw_rate: Comfort threshold for yaw rate (rad/s), default 1.0
        max_yaw_accel: Comfort threshold for yaw acceleration (rad/s²), default 1.0
        include_lateral: Whether to compute lateral acceleration and yaw metrics
        use_smoothing: Whether to use Savitzky-Golay filter for smoother derivatives (tuplan_garage style)
        smoothing_window: Window length for Savitzky-Golay filter (must be odd), default 15
        smoothing_order: Polynomial order for Savitzky-Golay filter, default 2
    
    Returns:
        Dictionary containing:
            - 'mean_longitudinal_accel': Average longitudinal acceleration magnitude
            - 'max_longitudinal_accel': Maximum longitudinal acceleration
            - 'mean_lateral_accel': Average lateral acceleration magnitude (if include_lateral)
            - 'max_lateral_accel': Maximum lateral acceleration (if include_lateral)
            - 'mean_jerk': Average jerk magnitude
            - 'max_jerk': Maximum jerk
            - 'mean_yaw_rate': Average yaw rate (if include_lateral)
            - 'max_yaw_rate': Maximum yaw rate (if include_lateral)
            - 'mean_yaw_accel': Average yaw acceleration (if include_lateral)
            - 'max_yaw_accel': Maximum yaw acceleration (if include_lateral)
            - 'comfort_violations': Number of timesteps exceeding thresholds
            - 'comfort_rate': Fraction of time within comfort limits
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2.5, 0], [4.5, 0]])
        >>> t = np.array([0.0, 1.0, 2.0, 3.0])
        >>> metrics = calculate_comfort_metrics(traj, t)
        
        >>> # With smoothing for noisy trajectories
        >>> metrics_smooth = calculate_comfort_metrics(traj, t, use_smoothing=True)
    """
    if len(trajectory) != len(timestamps):
        raise ValueError("Trajectory and timestamps must have same length")

    if len(trajectory) < 3:
        result = {
            'mean_longitudinal_accel': 0.0,
            'max_longitudinal_accel': 0.0,
            'mean_jerk': 0.0,
            'max_jerk': 0.0,
            'comfort_violations': 0,
            'comfort_rate': 1.0,
        }
        if include_lateral:
            result.update({
                'mean_lateral_accel': 0.0,
                'max_lateral_accel': 0.0,
                'mean_yaw_rate': 0.0,
                'max_yaw_rate': 0.0,
                'mean_yaw_accel': 0.0,
                'max_yaw_accel': 0.0,
            })
        return result

    # Calculate velocities (per-segment)
    dt = np.diff(timestamps)
    positions_diff = np.diff(trajectory, axis=0)
    # avoid division by zero
    dt_safe = np.where(dt == 0, 1e-8, dt)
    velocities = np.linalg.norm(positions_diff, axis=1) / dt_safe

    # Apply Savitzky-Golay smoothing if requested (tuplan_garage style)
    if use_smoothing and len(velocities) >= smoothing_window:
        try:
            from scipy.signal import savgol_filter
            # Ensure window length is odd
            if smoothing_window % 2 == 0:
                smoothing_window += 1
            # Ensure window is smaller than data length
            window = min(smoothing_window, len(velocities))
            if window % 2 == 0:
                window -= 1
            if window >= smoothing_order + 2:
                velocities = savgol_filter(velocities, window, smoothing_order)
        except ImportError:
            # Scipy not available, fall back to unsmoothed
            pass

    # Longitudinal accelerations
    dv = np.diff(velocities)
    dt_acc = dt_safe[:-1]
    longitudinal_accels = dv / dt_acc

    # Jerk (longitudinal)
    if len(longitudinal_accels) > 1:
        dj = np.diff(longitudinal_accels)
        dt_jerk = dt_acc[1:]
        dt_jerk_safe = np.where(dt_jerk == 0, 1e-8, dt_jerk)
        jerk = dj / dt_jerk_safe
    else:
        jerk = np.array([])

    # Lateral metrics (if requested)
    if include_lateral and len(trajectory) >= 3:
        # Compute headings from velocity vectors
        velocity_vectors = positions_diff / dt_safe[:, None]
        headings = np.arctan2(velocity_vectors[:, 1], velocity_vectors[:, 0])
        
        # Apply smoothing to headings if requested
        if use_smoothing and len(headings) >= smoothing_window:
            try:
                from scipy.signal import savgol_filter
                window = min(smoothing_window, len(headings))
                if window % 2 == 0:
                    window -= 1
                if window >= smoothing_order + 2:
                    # Unwrap before smoothing to handle angle discontinuities
                    headings_unwrapped = np.unwrap(headings)
                    headings_smooth = savgol_filter(headings_unwrapped, window, smoothing_order)
                    # Wrap back to [-pi, pi]
                    headings = np.arctan2(np.sin(headings_smooth), np.cos(headings_smooth))
            except ImportError:
                pass
        
        # Yaw rate (angular velocity)
        heading_diff = np.diff(headings)
        # Wrap to [-pi, pi]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        yaw_rates = heading_diff / dt_safe[1:]
        
        # Yaw acceleration
        if len(yaw_rates) > 1:
            dyaw = np.diff(yaw_rates)
            dt_yaw = dt_safe[2:]
            dt_yaw_safe = np.where(dt_yaw == 0, 1e-8, dt_yaw)
            yaw_accels = dyaw / dt_yaw_safe
        else:
            yaw_accels = np.array([])
        
        # Lateral acceleration: a_lat = v * yaw_rate
        # Align dimensions: yaw_rates corresponds to timestamps[2:]
        speeds_for_lateral = velocities[1:]  # velocities[1:] aligns with timestamps[1:-1]
        if len(speeds_for_lateral) == len(yaw_rates):
            lateral_accels = speeds_for_lateral * np.abs(yaw_rates)
        else:
            # Handle dimension mismatch
            min_len = min(len(speeds_for_lateral), len(yaw_rates))
            lateral_accels = speeds_for_lateral[:min_len] * np.abs(yaw_rates[:min_len])
    else:
        lateral_accels = np.array([])
        yaw_rates = np.array([])
        yaw_accels = np.array([])

    # Count comfort violations
    violations = 0
    if longitudinal_accels.size > 0:
        violations += int(np.sum(np.abs(longitudinal_accels) > max_longitudinal_accel))
    if jerk.size > 0:
        violations += int(np.sum(np.abs(jerk) > max_jerk))
    if include_lateral:
        if lateral_accels.size > 0:
            violations += int(np.sum(np.abs(lateral_accels) > max_lateral_accel))
        if yaw_rates.size > 0:
            violations += int(np.sum(np.abs(yaw_rates) > max_yaw_rate))
        if yaw_accels.size > 0:
            violations += int(np.sum(np.abs(yaw_accels) > max_yaw_accel))

    total_samples = (
        longitudinal_accels.size + jerk.size + 
        (lateral_accels.size + yaw_rates.size + yaw_accels.size if include_lateral else 0)
    )
    total_samples = max(total_samples, 1)
    comfort_rate = 1.0 - (violations / total_samples)

    result = {
        'mean_longitudinal_accel': float(np.mean(np.abs(longitudinal_accels))) if longitudinal_accels.size > 0 else 0.0,
        'max_longitudinal_accel': float(np.max(np.abs(longitudinal_accels))) if longitudinal_accels.size > 0 else 0.0,
        'mean_jerk': float(np.mean(np.abs(jerk))) if jerk.size > 0 else 0.0,
        'max_jerk': float(np.max(np.abs(jerk))) if jerk.size > 0 else 0.0,
        'comfort_violations': int(violations),
        'comfort_rate': float(np.clip(comfort_rate, 0.0, 1.0)),
    }
    
    if include_lateral:
        result.update({
            'mean_lateral_accel': float(np.mean(np.abs(lateral_accels))) if lateral_accels.size > 0 else 0.0,
            'max_lateral_accel': float(np.max(np.abs(lateral_accels))) if lateral_accels.size > 0 else 0.0,
            'mean_yaw_rate': float(np.mean(np.abs(yaw_rates))) if yaw_rates.size > 0 else 0.0,
            'max_yaw_rate': float(np.max(np.abs(yaw_rates))) if yaw_rates.size > 0 else 0.0,
            'mean_yaw_accel': float(np.mean(np.abs(yaw_accels))) if yaw_accels.size > 0 else 0.0,
            'max_yaw_accel': float(np.max(np.abs(yaw_accels))) if yaw_accels.size > 0 else 0.0,
        })
    
    return result


def calculate_driving_score(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    reference_path: np.ndarray,
    timestamps: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    mode: str = 'default',
    headings: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    lane_centerline: Optional[np.ndarray] = None,
    drivable_area: Optional[Any] = None,
    other_vehicles: Optional[List[Dict[str, np.ndarray]]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive driving score combining multiple metrics.
    
    Composite metric similar to nuPlan's driving score or CARLA's benchmark score,
    combining planning accuracy, safety, progress, and comfort.
    
    Args:
        predicted_trajectory: Planned trajectory, shape (T, 2)
        expert_trajectory: Expert trajectory, shape (T, 2)
        obstacles: List of obstacle trajectories
        reference_path: Reference path, shape (N, 2)
        timestamps: Time array, shape (T,)
        weights: Optional weights for each component (default/nuplan mode uses different defaults)
        mode: Scoring mode - 'default' or 'nuplan'
        headings: Optional heading angles (radians), shape (T,). Required for nuplan mode.
        velocities: Optional velocities (m/s), shape (T,). Required for nuplan mode.
        lane_centerline: Optional lane centerline, shape (N, 2). Used for nuplan mode.
        drivable_area: Optional drivable area polygon (Shapely). Used for nuplan mode.
        other_vehicles: Optional list of other vehicle dicts with 'trajectory', 'velocities', 'headings'.
                       Used for enhanced collision classification in nuplan mode.
    
    Returns:
        Dictionary containing:
            - 'driving_score': Overall score [0, 100]
            - 'planning_accuracy': Planning accuracy score [0, 100]
            - 'safety_score': Safety score [0, 100]
            - 'progress_score': Progress score [0, 100]
            - 'comfort_score': Comfort score [0, 100]
            
        Additional keys in 'nuplan' mode:
            - 'no_at_fault_collision': Boolean, pass/fail criterion
            - 'drivable_area_compliance': Score [0, 100]
            - 'driving_direction_compliance': Score [0, 100]
            - 'lateral_comfort_score': Score [0, 100]
    
    Example (default mode):
        >>> pred = np.random.rand(10, 2)
        >>> expert = np.random.rand(10, 2)
        >>> obs = [np.random.rand(10, 2)]
        >>> ref = np.random.rand(20, 2)
        >>> t = np.linspace(0, 5, 10)
        >>> score = calculate_driving_score(pred, expert, obs, ref, t)
    
    Example (nuplan mode):
        >>> score = calculate_driving_score(
        ...     pred, expert, obs, ref, t,
        ...     mode='nuplan',
        ...     headings=headings,
        ...     velocities=velocities,
        ...     lane_centerline=centerline,
        ...     other_vehicles=other_vehicles
        ... )
    """
    # Set default weights based on mode
    if mode == 'nuplan':
        default_weights = {
            'planning': 0.25,
            'safety': 0.40,
            'progress': 0.20,
            'comfort': 0.15
        }
    else:
        default_weights = {
            'planning': 0.3,
            'safety': 0.4,
            'progress': 0.2,
            'comfort': 0.1
        }
    
    if weights is None:
        weights = default_weights
    
    # Planning accuracy (lower L2 distance is better)
    l2_dist = calculate_l2_distance(predicted_trajectory, expert_trajectory)
    # Convert to score (0-100), assuming 5m is very poor, 0m is perfect
    planning_accuracy = max(0, 100 * (1 - min(l2_dist / 5.0, 1.0)))
    
    # Safety (no collisions is best)
    if mode == 'nuplan' and other_vehicles is not None and headings is not None:
        # Use fault classification for nuPlan mode
        # Extract obstacle headings from other_vehicles
        obstacle_headings = [v['headings'] for v in other_vehicles]
        obstacle_trajectories = [v['trajectory'] for v in other_vehicles]
        
        collision_result = calculate_collision_with_fault_classification(
            trajectory=predicted_trajectory,
            obstacles=obstacle_trajectories,
            ego_headings=headings,
            obstacle_headings=obstacle_headings
        )
        # nuPlan uses hard constraint: no at-fault collisions
        no_at_fault = collision_result['at_fault_collisions'] == 0
        safety_score_val = 100.0 if no_at_fault else 0.0
    else:
        # Default mode: use collision rate
        collision_result = calculate_collision_rate(predicted_trajectory, obstacles)
        safety_score_val = 100 * (1 - collision_result['collision_rate'])
        no_at_fault = None
    
    # Progress
    progress_result = calculate_progress_score(predicted_trajectory, reference_path)
    progress_score_val = 100 * progress_result['progress_ratio']
    
    # Comfort
    if mode == 'nuplan':
        # Enhanced comfort metrics with lateral acceleration
        comfort_result = calculate_comfort_metrics(
            predicted_trajectory, 
            timestamps,
            max_longitudinal_accel=4.0,  # nuPlan threshold
            max_lateral_accel=4.0,
            max_jerk=4.0,
            max_yaw_rate=0.5,
            max_yaw_accel=1.0,
            include_lateral=True
        )
        comfort_score_val = 100 * comfort_result['comfort_rate']
        lateral_comfort_score = 100 * (1 - min(comfort_result.get('max_lateral_accel', 0.0) / 4.0, 1.0))
    else:
        # Default mode: standard comfort with lateral metrics
        comfort_result = calculate_comfort_metrics(
            predicted_trajectory, 
            timestamps,
            include_lateral=True
        )
        comfort_score_val = 100 * comfort_result['comfort_rate']
        lateral_comfort_score = None
    
    # Weighted combination
    overall_score = (
        weights['planning'] * planning_accuracy +
        weights['safety'] * safety_score_val +
        weights['progress'] * progress_score_val +
        weights['comfort'] * comfort_score_val
    )
    
    result = {
        'driving_score': float(overall_score),
        'planning_accuracy': float(planning_accuracy),
        'safety_score': float(safety_score_val),
        'progress_score': float(progress_score_val),
        'comfort_score': float(comfort_score_val)
    }
    
    # Add nuPlan-specific metrics
    if mode == 'nuplan':
        result['no_at_fault_collision'] = no_at_fault
        
        if lateral_comfort_score is not None:
            result['lateral_comfort_score'] = float(lateral_comfort_score)
        
        # Drivable area compliance
        if drivable_area is not None or lane_centerline is not None:
            drivable_area_list = [drivable_area] if drivable_area is not None else None
            edge_result = calculate_distance_to_road_edge(
                trajectory=predicted_trajectory,
                drivable_area_polygons=drivable_area_list,
                lane_boundaries=lane_centerline,
                lane_width=3.5
            )
            # Score based on violation rate (0 violations = 100, all violations = 0)
            drivable_area_score = 100 * (1 - edge_result.get('violation_rate', 0.0))
            result['drivable_area_compliance'] = float(drivable_area_score)
        
        # Driving direction compliance
        if lane_centerline is not None:
            direction_result = calculate_driving_direction_compliance(
                trajectory=predicted_trajectory,
                reference_path=lane_centerline
            )
            # nuPlan score: 1.0 -> 100, 0.5 -> 50, 0.0 -> 0
            direction_score = direction_result['compliance_score'] * 100
            result['driving_direction_compliance'] = float(direction_score)
    
    return result


def calculate_time_to_collision(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    vehicle_size: Tuple[float, float] = (4.5, 2.0),
    obstacle_sizes: Optional[List[Tuple[float, float]]] = None,
    safety_margin: float = 0.0
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute minimum time-to-collision (TTC) along trajectory wrt obstacles.

    Returns min_ttc (seconds) and per-timestep min TTC values (inf if no imminent collision).
    """
    # This function assumes constant velocity per segment; timestamps are not given,
    # so TTC is approximated in timesteps (i.e., index-based). For precise TTC, use timestamps.
    T = len(trajectory)
    if obstacle_sizes is None:
        obstacle_sizes = [vehicle_size] * len(obstacles)

    ego_length, ego_width = vehicle_size
    ego_radius = np.sqrt((ego_length/2)**2 + (ego_width/2)**2) + safety_margin

    per_ttc = [float('inf')] * T
    # naive constant-speed TTC per pair: estimate relative closing speed along line-of-centers
    for t in range(T - 1):
        ego_pos = trajectory[t]
        ego_next = trajectory[t + 1]
        ego_vel = ego_next - ego_pos
        ego_speed = np.linalg.norm(ego_vel)

        best_ttc = float('inf')
        for obs_traj, (ol, ow) in zip(obstacles, obstacle_sizes):
            # get obstacle position at t (or static)
            if len(obs_traj.shape) == 1 or len(obs_traj) == 1:
                obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
                obs_vel = np.array([0.0, 0.0])
            else:
                if t < len(obs_traj) - 1:
                    obs_pos = obs_traj[t]
                    obs_next = obs_traj[t + 1]
                    obs_vel = obs_next - obs_pos
                else:
                    obs_pos = obs_traj[-1]
                    obs_vel = np.array([0.0, 0.0])

            rel_pos = obs_pos - ego_pos
            rel_vel = obs_vel - ego_vel
            rel_speed_along = np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-8)
            # if closing (rel_speed_along < 0), estimate time to distance = sum radii
            obs_radius = np.sqrt((ol/2)**2 + (ow/2)**2) + safety_margin
            dist = np.linalg.norm(rel_pos) - (ego_radius + obs_radius)
            if rel_speed_along < -1e-6 and dist > 0:
                ttc = dist / (-rel_speed_along)
                if ttc < best_ttc:
                    best_ttc = ttc

        per_ttc[t] = best_ttc

    min_ttc = min(per_ttc) if len(per_ttc) > 0 else float('inf')
    return {'min_ttc': float(min_ttc), 'per_ttc': per_ttc}


def calculate_lane_invasion_rate(
    trajectory: np.ndarray,
    lane_centerlines: List[np.ndarray],
    lane_width: float = 3.5
) -> Dict[str, Union[float, int]]:
    """
    Simple lane invasion metric: fraction of timesteps where vehicle is outside any lane centerline +- lane_width/2.
    """
    half_w = lane_width / 2.0
    T = len(trajectory)
    if T == 0:
        return {'invasion_rate': 0.0, 'invasion_count': 0}

    outside_count = 0
    for p in trajectory:
        inside_any = False
        for center in lane_centerlines:
            # compute nearest distance to center polyline (reuse lateral projection)
            min_dist = float('inf')
            for i in range(len(center) - 1):
                a = center[i]
                b = center[i + 1]
                ap = p - a
                ab = b - a
                ab_len2 = np.dot(ab, ab)
                if ab_len2 == 0:
                    d = np.linalg.norm(ap)
                else:
                    t = np.clip(np.dot(ap, ab) / ab_len2, 0.0, 1.0)
                    proj = a + t * ab
                    d = np.linalg.norm(p - proj)
                if d < min_dist:
                    min_dist = d
            if min_dist <= half_w:
                inside_any = True
                break
        if not inside_any:
            outside_count += 1

    invasion_rate = outside_count / T
    return {'invasion_rate': float(invasion_rate), 'invasion_count': outside_count}


def calculate_collision_severity(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    timestamps: Optional[np.ndarray] = None
) -> Dict[str, Union[float, List[float]]]:
    """
    Estimate collision severity as relative speed at collision timesteps (simple proxy).
    Returns list of severities and max severity.
    """
    T = len(trajectory)
    severities: List[float] = []
    for t in range(T - 1):
        ego_pos = trajectory[t]
        # find collisions at t using existing circular check
        for obs_traj in obstacles:
            if len(obs_traj.shape) == 1 or len(obs_traj) == 1:
                obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
                obs_next = obs_pos
            else:
                if t < len(obs_traj) - 1:
                    obs_pos = obs_traj[t]
                    obs_next = obs_traj[t + 1]
                else:
                    obs_pos = obs_traj[-1]
                    obs_next = obs_pos

            dist = np.linalg.norm(ego_pos - obs_pos)
            # rough radius used here: assume small vehicle footprint (1.5m)
            if dist < 1.5:
                # relative speed magnitude
                ego_vel = trajectory[t + 1] - ego_pos
                obs_vel = obs_next - obs_pos
                rel_speed = np.linalg.norm(ego_vel - obs_vel)
                severities.append(float(rel_speed))

    max_sev = max(severities) if severities else 0.0
    return {'severities': severities, 'max_severity': float(max_sev)}


def check_kinematic_feasibility(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    max_lateral_accel: float = 3.0,
    max_yaw_rate: float = 1.0
) -> Dict[str, Union[bool, float]]:
    """
    Basic kinematic feasibility checks: lateral accel and yaw rate thresholds.
    Returns boolean 'feasible' and max observed values.
    """
    if len(trajectory) < 3:
        return {'feasible': True, 'max_lateral_accel': 0.0, 'max_yaw_rate': 0.0}

    dt = np.diff(timestamps)
    dt_safe = np.where(dt == 0, 1e-8, dt)
    positions_diff = np.diff(trajectory, axis=0)
    velocities = positions_diff / dt_safe[:, None]
    speeds = np.linalg.norm(velocities, axis=1)

    # lateral accel approx: a_lat = v^2 * curvature; estimate curvature via finite differences of heading
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_diff = np.diff(headings)
    heading_dt = dt_safe[1:]
    yaw_rates = heading_diff / heading_dt
    # curvature approx: |yaw_rate|/speed (avoid div by zero)
    curvature = np.zeros_like(yaw_rates)
    for i in range(len(yaw_rates)):
        if speeds[i + 1] > 1e-6:
            curvature[i] = abs(yaw_rates[i]) / speeds[i + 1]
    lateral_accels = (speeds[1:]**2) * curvature

    max_lat = float(np.max(np.abs(lateral_accels))) if lateral_accels.size > 0 else 0.0
    max_yaw = float(np.max(np.abs(yaw_rates))) if yaw_rates.size > 0 else 0.0
    feasible = (max_lat <= max_lateral_accel) and (max_yaw <= max_yaw_rate)
    return {'feasible': bool(feasible), 'max_lateral_accel': max_lat, 'max_yaw_rate': max_yaw}


def calculate_planning_kl_divergence(
    predicted_distribution: np.ndarray,
    expert_distribution: np.ndarray,
    bins: int = 50
) -> float:
    """
    Calculate KL divergence between predicted and expert action distributions.
    
    For imitation learning, measures how well the learned policy matches
    the expert policy in terms of action distributions.
    
    Args:
        predicted_distribution: Predicted action probabilities, shape (N,)
        expert_distribution: Expert action probabilities, shape (N,)
        bins: Number of bins for discretization (if inputs are continuous)
    
    Returns:
        KL divergence value (lower is better)
    
    Example:
        >>> pred_dist = np.array([0.2, 0.5, 0.3])
        >>> expert_dist = np.array([0.1, 0.6, 0.3])
        >>> kl = planning_kl_divergence(pred_dist, expert_dist)
    """
    # Note: this function returns KL(expert || predicted)
    # Ensure distributions are normalized
    pred_dist = predicted_distribution / (np.sum(predicted_distribution) + 1e-10)
    expert_dist = expert_distribution / (np.sum(expert_distribution) + 1e-10)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    pred_dist = pred_dist + epsilon
    expert_dist = expert_dist + epsilon
    
    # Renormalize after adding epsilon
    pred_dist = pred_dist / np.sum(pred_dist)
    expert_dist = expert_dist / np.sum(expert_dist)
    
    # Calculate KL divergence
    kl_div = entropy(expert_dist, pred_dist)
    
    return float(kl_div)


def calculate_time_to_collision_enhanced(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    timestamps: np.ndarray,
    vehicle_size: Tuple[float, float] = (4.5, 2.0),
    obstacle_sizes: Optional[List[Tuple[float, float]]] = None,
    safety_margin: float = 0.0,
    projection_horizon: float = 1.0,
    stopped_speed_threshold: float = 0.005
) -> Dict[str, Union[float, List[float]]]:
    """
    Enhanced TTC with forward projection (nuPlan-style).
    
    Projects ego vehicle forward up to projection_horizon seconds and checks
    for collisions with obstacles. More accurate than basic TTC.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        obstacles: List of obstacle trajectories
        timestamps: Time at each position, shape (T,)
        vehicle_size: Ego vehicle (length, width) in meters
        obstacle_sizes: List of obstacle sizes
        safety_margin: Additional safety buffer in meters
        projection_horizon: How far into future to project (seconds), default 1.0
        stopped_speed_threshold: Speed below which vehicle is considered stopped (m/s)
    
    Returns:
        Dictionary containing:
            - 'min_ttc': Minimum TTC across all timesteps (seconds)
            - 'per_ttc': TTC at each timestep
            - 'ttc_violations': Number of timesteps with TTC < 3s
            - 'ttc_score': Score based on violations
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> t = np.array([0.0, 1.0, 2.0])
        >>> obs = [np.array([[5, 0]])]
        >>> result = calculate_time_to_collision_enhanced(traj, obs, t)
    """
    T = len(trajectory)
    if obstacle_sizes is None:
        obstacle_sizes = [vehicle_size] * len(obstacles)
    
    ego_length, ego_width = vehicle_size
    ego_radius = np.sqrt((ego_length/2)**2 + (ego_width/2)**2) + safety_margin
    
    # Compute velocities
    dt = np.diff(timestamps)
    dt_safe = np.where(dt == 0, 1e-8, dt)
    velocity_vectors = np.diff(trajectory, axis=0) / dt_safe[:, None]
    speeds = np.linalg.norm(velocity_vectors, axis=1)
    
    per_ttc = [float('inf')] * T
    
    # Project at intervals: 0s, 0.3s, 0.6s, 0.9s (nuPlan style)
    projection_times = np.arange(0, projection_horizon + 0.01, 0.3)
    
    for t in range(T - 1):
        ego_pos = trajectory[t]
        ego_speed = speeds[t] if t < len(speeds) else 0.0
        
        # Skip if ego is stopped
        if ego_speed < stopped_speed_threshold:
            continue
        
        ego_vel = velocity_vectors[t] if t < len(velocity_vectors) else np.zeros(2)
        
        best_ttc = float('inf')
        
        for proj_time in projection_times:
            # Project ego position forward
            projected_ego_pos = ego_pos + ego_vel * proj_time
            
            for obs_traj, (ol, ow) in zip(obstacles, obstacle_sizes):
                # Get obstacle position at projected time
                if len(obs_traj.shape) == 1 or len(obs_traj) == 1:
                    obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
                else:
                    # Find closest timestep to projected time
                    future_time = timestamps[t] + proj_time
                    # Find index in obstacle trajectory
                    obs_t_idx = t
                    if t < len(obs_traj):
                        obs_pos = obs_traj[obs_t_idx]
                    else:
                        continue
                
                obs_radius = np.sqrt((ol/2)**2 + (ow/2)**2) + safety_margin
                
                # Check if collision would occur at this projection
                dist = np.linalg.norm(projected_ego_pos - obs_pos)
                if dist < (ego_radius + obs_radius):
                    # Collision imminent at this projection time
                    ttc = proj_time if proj_time > 0 else 0.01
                    best_ttc = min(best_ttc, ttc)
        
        per_ttc[t] = best_ttc
    
    min_ttc = min(per_ttc) if len(per_ttc) > 0 else float('inf')
    
    # Count violations (TTC < 3 seconds)
    violations = sum(1 for ttc in per_ttc if ttc < 3.0 and ttc != float('inf'))
    ttc_score = 1.0 if violations == 0 else 0.0
    
    return {
        'min_ttc': float(min_ttc),
        'per_ttc': per_ttc,
        'ttc_violations': violations,
        'ttc_score': float(ttc_score)
    }


def calculate_distance_to_road_edge(
    trajectory: np.ndarray,
    drivable_area_polygons: Optional[List] = None,
    lane_boundaries: Optional[np.ndarray] = None,
    lane_width: float = 3.5
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate signed distance to road edge (Waymo Sim Agents style).
    
    Computes continuous distance metric where:
    - Negative values = inside drivable area
    - Positive values = outside drivable area
    - Zero = exactly at boundary
    
    More granular than binary in/out detection for safety assessment.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        drivable_area_polygons: List of Shapely Polygon objects (if available)
        lane_boundaries: Fallback to lane centerline, shape (N, 2)
        lane_width: Lane width in meters (used if no polygons available)
    
    Returns:
        Dictionary containing:
            - 'distances': Signed distances at each timestep (negative=inside)
            - 'mean_distance': Average signed distance
            - 'min_distance': Minimum distance (closest to edge)
            - 'offroad_rate': Fraction of timesteps outside drivable area
            - 'close_to_edge_rate': Fraction within 0.5m of edge
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0.5], [2, 2.0]])
        >>> lane = np.array([[0, 0], [2, 0], [4, 0]])
        >>> result = calculate_distance_to_road_edge(traj, lane_boundaries=lane)
    """
    T = len(trajectory)
    distances = np.zeros(T)
    
    if drivable_area_polygons is not None:
        # Use actual polygon-based calculation
        try:
            from shapely.geometry import Point
            for i, pos in enumerate(trajectory):
                point = Point(pos)
                # Find minimum distance to any polygon boundary
                min_dist = float('inf')
                is_inside = False
                
                for polygon in drivable_area_polygons:
                    if polygon.contains(point):
                        is_inside = True
                        # Distance to boundary (negative inside)
                        dist_to_boundary = point.distance(polygon.boundary)
                        min_dist = min(min_dist, dist_to_boundary)
                    else:
                        # Distance from outside (positive)
                        dist_to_boundary = point.distance(polygon.boundary)
                        min_dist = min(min_dist, dist_to_boundary)
                
                distances[i] = -min_dist if is_inside else min_dist
        except ImportError:
            # Shapely not available, fallback to lane-based
            drivable_area_polygons = None
    
    if drivable_area_polygons is None and lane_boundaries is not None:
        # Fallback: use lane centerline with width
        half_width = lane_width / 2.0
        
        for i, pos in enumerate(trajectory):
            # Find distance to nearest lane segment
            min_dist_to_centerline = float('inf')
            for j in range(len(lane_boundaries) - 1):
                a = lane_boundaries[j]
                b = lane_boundaries[j + 1]
                # Point to segment distance
                ap = pos - a
                ab = b - a
                ab_len2 = np.dot(ab, ab)
                if ab_len2 == 0:
                    d = np.linalg.norm(ap)
                else:
                    t = np.clip(np.dot(ap, ab) / ab_len2, 0.0, 1.0)
                    proj = a + t * ab
                    d = np.linalg.norm(pos - proj)
                min_dist_to_centerline = min(min_dist_to_centerline, d)
            
            # Signed distance: negative if within lane, positive if outside
            if min_dist_to_centerline <= half_width:
                distances[i] = -(half_width - min_dist_to_centerline)
            else:
                distances[i] = min_dist_to_centerline - half_width
    
    # Compute metrics
    mean_distance = float(np.mean(distances))
    min_distance = float(np.min(np.abs(distances)))
    offroad_count = np.sum(distances > 0)
    offroad_rate = float(offroad_count / T) if T > 0 else 0.0
    
    # Close to edge: within 0.5m
    close_to_edge_count = np.sum(np.abs(distances) < 0.5)
    close_to_edge_rate = float(close_to_edge_count / T) if T > 0 else 0.0
    
    return {
        'distances': distances,
        'mean_distance': mean_distance,
        'min_distance': min_distance,
        'offroad_rate': offroad_rate,
        'close_to_edge_rate': close_to_edge_rate
    }


def calculate_driving_direction_compliance(
    trajectory: np.ndarray,
    reference_path: np.ndarray,
    route_direction_vectors: Optional[np.ndarray] = None
) -> Dict[str, Union[float, bool, np.ndarray]]:
    """
    Detect wrong-way driving using lane direction (nuPlan-style).
    
    Measures progress made while driving against traffic direction.
    Uses thresholds from nuPlan:
    - < 2m wrong-way: compliant (score 1.0)
    - 2-6m wrong-way: warning (score 0.5)
    - > 6m wrong-way: violation (score 0.0)
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        reference_path: Reference centerline, shape (N, 2)
        route_direction_vectors: Direction vectors for route (optional), shape (N, 2)
    
    Returns:
        Dictionary containing:
            - 'max_wrong_way_distance': Maximum progress against traffic (meters)
            - 'compliance_score': 1.0 (< 2m), 0.5 (2-6m), 0.0 (> 6m)
            - 'is_compliant': Boolean indicating compliance
            - 'wrong_way_timesteps': Number of timesteps driving wrong way
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> ref = np.array([[0, 0], [5, 0], [10, 0]])
        >>> result = calculate_driving_direction_compliance(traj, ref)
    """
    T = len(trajectory)
    
    # Compute trajectory direction vectors
    if T < 2:
        return {
            'max_wrong_way_distance': 0.0,
            'compliance_score': 1.0,
            'is_compliant': True,
            'wrong_way_timesteps': 0
        }
    
    traj_directions = np.diff(trajectory, axis=0)
    
    # If route direction not provided, estimate from reference path
    if route_direction_vectors is None:
        route_direction_vectors = np.diff(reference_path, axis=0)
        # Normalize
        route_norms = np.linalg.norm(route_direction_vectors, axis=1, keepdims=True)
        route_direction_vectors = route_direction_vectors / (route_norms + 1e-8)
    
    # For each trajectory segment, find nearest reference segment
    wrong_way_distances = []
    wrong_way_count = 0
    
    for i in range(len(traj_directions)):
        traj_dir = traj_directions[i]
        traj_norm = np.linalg.norm(traj_dir)
        
        if traj_norm < 1e-6:
            continue  # Stationary
        
        traj_dir_normalized = traj_dir / traj_norm
        
        # Find closest reference segment
        traj_pos = trajectory[i]
        min_dist_idx = 0
        min_dist = float('inf')
        for j in range(len(reference_path) - 1):
            dist = np.linalg.norm(reference_path[j] - traj_pos)
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = j
        
        # Get reference direction at this point
        if min_dist_idx < len(route_direction_vectors):
            ref_dir = route_direction_vectors[min_dist_idx]
        else:
            ref_dir = route_direction_vectors[-1]
        
        # Check if driving against direction (dot product < 0)
        dot_product = np.dot(traj_dir_normalized, ref_dir)
        
        if dot_product < 0:
            # Driving wrong way - accumulate distance
            wrong_way_distances.append(traj_norm)
            wrong_way_count += 1
        else:
            # Correct direction - reset accumulation
            if wrong_way_distances:
                wrong_way_distances = []
    
    # Maximum continuous wrong-way distance
    max_wrong_way_distance = sum(wrong_way_distances) if wrong_way_distances else 0.0
    
    # nuPlan thresholds
    COMPLIANCE_THRESHOLD = 2.0  # meters
    VIOLATION_THRESHOLD = 6.0  # meters
    
    if max_wrong_way_distance < COMPLIANCE_THRESHOLD:
        compliance_score = 1.0
        is_compliant = True
    elif max_wrong_way_distance < VIOLATION_THRESHOLD:
        compliance_score = 0.5
        is_compliant = False
    else:
        compliance_score = 0.0
        is_compliant = False
    
    return {
        'max_wrong_way_distance': float(max_wrong_way_distance),
        'compliance_score': float(compliance_score),
        'is_compliant': bool(is_compliant),
        'wrong_way_timesteps': wrong_way_count
    }


def calculate_interaction_metrics(
    ego_trajectory: np.ndarray,
    other_trajectories: List[np.ndarray],
    vehicle_size: Tuple[float, float] = (4.5, 2.0)
) -> Dict[str, Union[float, np.ndarray, int]]:
    """
    Calculate interaction metrics between ego and other agents (Waymo Sim Agents style).
    
    Measures proximity and interaction intensity for evaluating realistic multi-agent behavior.
    Used in Waymo Sim Agents Challenge for interaction realism assessment.
    
    Args:
        ego_trajectory: Ego vehicle trajectory, shape (T, 2) for [x, y]
        other_trajectories: List of other agent trajectories, each shape (T, 2)
        vehicle_size: Vehicle size (length, width) for collision radius calculation
    
    Returns:
        Dictionary containing:
            - 'min_distance': Minimum distance to any object across entire trajectory (meters)
            - 'mean_distance_to_nearest': Average distance to nearest object per timestep
            - 'distance_to_nearest_per_timestep': Array of distances, shape (T,)
            - 'closest_object_id': Index of object that came closest
            - 'closest_approach_timestep': When closest approach occurred
            - 'num_close_interactions': Count of timesteps with distance < 5m
    
    Example:
        >>> ego_traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> other_trajs = [
        ...     np.array([[3, 1], [3, 0.5], [3, 0]]),  # Approaching vehicle
        ...     np.array([[0, 5], [1, 5], [2, 5]])     # Distant vehicle
        ... ]
        >>> metrics = calculate_interaction_metrics(ego_traj, other_trajs)
        >>> print(f"Closest approach: {metrics['min_distance']:.2f}m")
    """
    if len(other_trajectories) == 0:
        T = len(ego_trajectory)
        return {
            'min_distance': float('inf'),
            'mean_distance_to_nearest': float('inf'),
            'distance_to_nearest_per_timestep': np.full(T, float('inf')),
            'closest_object_id': -1,
            'closest_approach_timestep': -1,
            'num_close_interactions': 0
        }
    
    T = len(ego_trajectory)
    distance_to_nearest_per_timestep = np.full(T, float('inf'))
    
    # Calculate safe distance threshold (2x vehicle diagonal)
    ego_length, ego_width = vehicle_size
    safe_distance_threshold = 2 * np.sqrt((ego_length/2)**2 + (ego_width/2)**2)
    close_interaction_threshold = 5.0  # meters, Waymo standard
    
    # For each timestep, find distance to nearest object
    for t in range(T):
        ego_pos = ego_trajectory[t]
        min_dist_at_t = float('inf')
        
        for other_traj in other_trajectories:
            # Handle static obstacles (single position)
            if len(other_traj.shape) == 1 or len(other_traj) == 1:
                other_pos = other_traj if len(other_traj.shape) == 1 else other_traj[0]
            else:
                # Dynamic obstacle - get position at time t
                if t < len(other_traj):
                    other_pos = other_traj[t]
                else:
                    continue  # Trajectory ended
            
            distance = np.linalg.norm(ego_pos - other_pos)
            if distance < min_dist_at_t:
                min_dist_at_t = distance
        
        distance_to_nearest_per_timestep[t] = min_dist_at_t
    
    # Find global minimum distance
    min_distance = float(np.min(distance_to_nearest_per_timestep))
    mean_distance = float(np.mean(distance_to_nearest_per_timestep))
    closest_approach_timestep = int(np.argmin(distance_to_nearest_per_timestep))
    
    # Find which object was closest
    closest_object_id = -1
    ego_pos_at_closest = ego_trajectory[closest_approach_timestep]
    min_dist_to_check = float('inf')
    
    for obj_id, other_traj in enumerate(other_trajectories):
        if len(other_traj.shape) == 1 or len(other_traj) == 1:
            other_pos = other_traj if len(other_traj.shape) == 1 else other_traj[0]
        else:
            if closest_approach_timestep < len(other_traj):
                other_pos = other_traj[closest_approach_timestep]
            else:
                continue
        
        distance = np.linalg.norm(ego_pos_at_closest - other_pos)
        if distance < min_dist_to_check:
            min_dist_to_check = distance
            closest_object_id = obj_id
    
    # Count close interactions (< 5m)
    num_close_interactions = int(np.sum(distance_to_nearest_per_timestep < close_interaction_threshold))
    
    return {
        'min_distance': min_distance,
        'mean_distance_to_nearest': mean_distance,
        'distance_to_nearest_per_timestep': distance_to_nearest_per_timestep,
        'closest_object_id': closest_object_id,
        'closest_approach_timestep': closest_approach_timestep,
        'num_close_interactions': num_close_interactions
    }


