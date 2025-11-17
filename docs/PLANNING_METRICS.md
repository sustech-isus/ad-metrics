# Planning Metrics

This document describes metrics for evaluating end-to-end autonomous driving models that directly map sensor inputs to driving actions (steering, acceleration, braking) or planned trajectories.

## Table of Contents

1. [Overview](#overview)
2. [Planning Accuracy Metrics](#planning-accuracy-metrics)
   - [L2 Distance](#l2-distance)
   - [Average Displacement Error (ADE) & Final Displacement Error (FDE)](#average-displacement-error-ade--final-displacement-error-fde)
3. [Safety Metrics](#safety-metrics)
   - [Collision Rate](#collision-rate)
   - [Collision with Fault Classification](#collision-with-fault-classification)
   - [Time-To-Collision (TTC)](#time-to-collision-ttc)
   - [Enhanced Time-To-Collision (Forward Projection)](#enhanced-time-to-collision-forward-projection)
   - [Lane Invasion / Off-road Rate](#lane-invasion--off-road-rate)
   - [Collision Severity](#collision-severity)
   - [Kinematic Feasibility](#kinematic-feasibility)
   - [Distance to Road Edge](#distance-to-road-edge)
   - [Driving Direction Compliance](#driving-direction-compliance)
4. [Progress and Navigation Metrics](#progress-and-navigation-metrics)
   - [Progress Score](#progress-score)
   - [Route Completion](#route-completion)
5. [Control Accuracy Metrics](#control-accuracy-metrics)
   - [Lateral Deviation](#lateral-deviation)
   - [Heading Error](#heading-error)
   - [Velocity Error](#velocity-error)
6. [Comfort Metrics](#comfort-metrics)
   - [Comfort Metrics (Acceleration & Jerk)](#comfort-metrics-acceleration--jerk)
7. [Composite Metrics](#composite-metrics)
   - [Driving Score](#driving-score)
8. [Imitation Learning Metrics](#imitation-learning-metrics)
   - [Planning KL Divergence](#planning-kl-divergence)
9. [Benchmark References](#benchmark-references)
   - [nuPlan](#nuplan)
   - [CARLA](#carla)
   - [Waymo Open Dataset](#waymo-open-dataset)
   - [Argoverse 2](#argoverse-2)
10. [Best Practices](#best-practices)
11. [Example: Complete Evaluation Pipeline](#example-complete-evaluation-pipeline)
12. [References](#references)
13. [See Also](#see-also)

## Overview

End-to-end autonomous driving represents a paradigm shift from modular perception-planning-control pipelines to unified models that directly output driving actions. Examples include:

- **Tesla FSD**: Vision-based neural planner
- **Wayve**: End-to-end learned driving
- **nuPlan**: Closed-loop planning benchmark
- **CARLA**: Simulation-based evaluation

These metrics evaluate the quality, safety, and efficiency of end-to-end driving policies.

### Key Evaluation Dimensions

| Dimension | What It Measures | Key Metrics |
|-----------|------------------|-------------|
| **Planning Accuracy** | How well trajectories match expert/optimal paths | L2 distance, ADE, FDE |
| **Safety** | Collision avoidance and safe operation | Collision rate, collision fault classification, time-to-collision (basic & enhanced), distance to road edge, driving direction compliance |
| **Progress** | Task completion and navigation efficiency | Progress score, route completion |
| **Control** | Accuracy of path following and speed control | Lateral deviation, heading error, velocity error |
| **Comfort** | Smooth driving and passenger experience | Acceleration (longitudinal & lateral), jerk, yaw rate, yaw acceleration, comfort rate |
| **Composite** | Overall driving performance | Driving score (nuPlan/CARLA style) |

## Planning Accuracy Metrics

### L2 Distance

**Purpose**: Primary metric for trajectory matching quality.

**Formula**:
```
L2 = (1/N) * Σ w_i * ||p_i - g_i||_2
```
where:
- `p_i`: predicted waypoint at timestep i
- `g_i`: ground truth/expert waypoint at timestep i
- `w_i`: optional temporal weight (default: uniform)
- `N`: number of waypoints

**Usage**:
```python
from admetrics.planning import calculate_l2_distance

# Basic usage
dist = calculate_l2_distance(predicted_traj, expert_traj)

# Emphasize long-term planning (later timesteps weighted more)
weights = np.linspace(1.0, 2.0, len(predicted_traj))
dist_weighted = calculate_l2_distance(predicted_traj, expert_traj, weights=weights)
```

**Interpretation**:
- Lower is better
- Typical good performance: < 0.5m for urban driving
- Used in nuPlan, CARLA benchmarks

**Temporal Weighting Strategies**:
- **Uniform** (default): All timesteps equally important
- **Linear increasing**: Emphasize long-term planning accuracy
- **Exponential**: Strong emphasis on distant future
- **Custom**: Task-specific (e.g., emphasize near-goal states)

### Average Displacement Error (ADE) & Final Displacement Error (FDE)

**Purpose**: Standard metrics for trajectory prediction adapted to planning evaluation.

**Formulas**:
```
ADE = (1/T) * Σ w_t * ||p_t - g_t||_2
FDE = ||p_T - g_T||_2
```

**Usage**:
```python
from admetrics.planning import average_displacement_error_planning

# Uniform weighting
result = average_displacement_error_planning(pred_traj, expert_traj)

# Linear weighting (emphasize later timesteps)
result = average_displacement_error_planning(
    pred_traj, expert_traj,
    timestep_weights='linear'
)

# Exponential weighting (alpha controls rate)
result = average_displacement_error_planning(
    pred_traj, expert_traj,
    timestep_weights='exponential',
    alpha=0.5
)

# Multi-horizon ADE/FDE (Waymo/nuPlan style)
# Evaluate at 1s, 3s, 5s for 10Hz data (10, 30, 50 timesteps)
result = average_displacement_error_planning(
    pred_traj, expert_traj,
    horizons=[10, 30, 50]
)

print(f"Overall ADE: {result['ADE']:.3f}m")
print(f"Overall FDE: {result['FDE']:.3f}m")
print(f"Short-term (1s): ADE={result['ADE_10']:.3f}m, FDE={result['FDE_10']:.3f}m")
print(f"Mid-term (3s):   ADE={result['ADE_30']:.3f}m, FDE={result['FDE_30']:.3f}m")
print(f"Long-term (5s):  ADE={result['ADE_50']:.3f}m, FDE={result['FDE_50']:.3f}m")
```

**Multi-Horizon Best Practices**:

Industry-standard horizons (Waymo Open Dataset, nuPlan):
- **1 second**: Immediate control accuracy, critical for safety
- **3 seconds**: Medium-term planning, lane changes and turns
- **5 seconds**: Long-term planning, route following
- **8 seconds**: Strategic planning (motion forecasting)

Example for 10Hz trajectory data (10 samples per second):
```python
# Define horizons in timesteps based on your sampling rate
sampling_rate = 10  # Hz
horizons_seconds = [1, 3, 5, 8]
horizons_timesteps = [int(h * sampling_rate) for h in horizons_seconds]

result = average_displacement_error_planning(
    pred_traj, expert_traj,
    horizons=horizons_timesteps  # [10, 30, 50, 80]
)

# Access results
for h_sec, h_step in zip(horizons_seconds, horizons_timesteps):
    print(f"{h_sec}s horizon: ADE={result[f'ADE_{h_step}']:.2f}m")
```

**Parameters**:
- `timestep_weights`: 'uniform', 'linear', 'exponential', or array
- `alpha`: Exponential decay rate (for exponential weighting)
- `horizons`: List of timestep indices for multi-horizon evaluation

**Interpretation**:
- **ADE**: Overall trajectory accuracy
- **FDE**: Long-term planning capability
- Lower values indicate better performance
- FDE often more critical for goal-reaching tasks

## Safety Metrics

### Collision Rate

**Purpose**: Evaluate collision avoidance with static and dynamic obstacles.

**Formula**:
```
collision_rate = (# timesteps with collision) / (total timesteps)
```

**Usage**:
```python
from admetrics.planning import calculate_collision_rate

# Define obstacles
static_obstacles = [
    np.array([[5, 2]]),  # Parked car
    np.array([[10, -1]])  # Road barrier
]

dynamic_obstacles = [
    np.array([[8, 5], [8, 3], [8, 1]])  # Crossing pedestrian
]

obstacles = static_obstacles + dynamic_obstacles

result = calculate_collision_rate(
    ego_trajectory,
    obstacles,
    vehicle_size=(4.5, 2.0),  # length x width in meters
    safety_margin=0.5,  # additional safety buffer
    obstacle_sizes=[(2.0, 2.0)]  # default size for obstacles
)

print(f"Collision rate: {result['collision_rate']:.2%}")
print(f"Number of collisions: {result['num_collisions']}")
print(f"Collision timesteps: {result['collision_timesteps']}")
print(f"First collision: {result['first_collision']}")
```

**Parameters**:
- `vehicle_size`: (length, width) of ego vehicle
- `safety_margin`: Additional buffer around vehicle
- `obstacle_size`: Default size for obstacles

**Collision Detection**:
- Uses circular bounding approximation for efficiency
- Collision radius = `sqrt((L/2)^2 + (W/2)^2) + margin`
- Supports both static and dynamic obstacles

Implementation notes:
- The library exposes `calculate_collision_rate` which uses circular approximations for ego and obstacle radii.
- Static obstacles may be passed either as a single `[x, y]` point or as a length-1 trajectory (shape `(1,2)`).
- The function accepts an `obstacle_sizes` list of (length, width) tuples to specify per-obstacle sizes.

**Interpretation**:
- **Critical**: Should be 0.0% for safe operation
- Used as hard constraint in nuPlan (no collisions allowed)
- CARLA uses weighted collision penalty

### Collision with Fault Classification

**Purpose**: Classify collisions by fault type to distinguish aggressive driving (at-fault) from unavoidable defensive scenarios (not-at-fault). Based on nuPlan benchmark standards.

**Formula**:
Collision types:
- **At-fault collisions** (ego is aggressive):
  - `active_front`: Ego rear-ends another vehicle
  - `stopped_track`: Ego hits stopped or slow-moving object
  - `active_lateral`: Ego sideswipes another vehicle
- **Not-at-fault collisions** (ego is defensive):
  - `active_rear`: Ego is rear-ended by another vehicle

**Usage**:
```python
from admetrics.planning import calculate_collision_with_fault_classification

# Trajectory with velocities
ego_trajectory = np.array([[0, 0], [1, 0], [2, 0]])  # shape (T, 2)
ego_velocities = np.array([5.0, 5.0, 5.0])  # m/s, shape (T,)
ego_headings = np.array([0.0, 0.0, 0.0])  # radians, shape (T,)

# Other vehicles (list of dicts)
other_vehicles = [
    {
        'trajectory': np.array([[3, 0], [3, 0], [3, 0]]),  # stopped car ahead
        'velocities': np.array([0.0, 0.0, 0.0]),
        'headings': np.array([0.0, 0.0, 0.0])
    },
    {
        'trajectory': np.array([[-2, 1], [0, 1], [2, 1]]),  # crossing vehicle
        'velocities': np.array([2.0, 2.0, 2.0]),
        'headings': np.array([0.0, 0.0, 0.0])
    }
]

result = calculate_collision_with_fault_classification(
    ego_trajectory=ego_trajectory,
    ego_velocities=ego_velocities,
    ego_headings=ego_headings,
    other_vehicles=other_vehicles,
    vehicle_size=(4.5, 2.0),  # ego size (length, width)
    safety_margin=0.5,
    slow_speed_threshold=0.5,  # m/s, stopped vehicle threshold
    front_rear_angle_threshold=np.pi/4  # 45 degrees
)

print(f"Total collisions: {result['total_collisions']}")
print(f"At-fault: {result['at_fault_collisions']}, Not-at-fault: {result['not_at_fault_collisions']}")
print(f"Collision types: {result['collision_types']}")
print(f"Fault classification: {result['fault_classification']}")
```

**Classification Logic**:
1. **Stopped/slow target** (v < 0.5 m/s) → `stopped_track` (at-fault)
2. **Front collision** (relative angle < 45°):
   - If ego faster → `active_front` (at-fault, rear-ending)
   - If target faster → `active_rear` (not-at-fault, being rear-ended)
3. **Rear collision** (relative angle > 135°):
   - If ego faster → `active_rear` (not-at-fault, being hit from behind)
   - If target faster → `active_front` (at-fault, backing into)
4. **Lateral collision** (45° < angle < 135°) → `active_lateral` (at-fault, sideswipe)

**Returns**:
- `total_collisions`: Total number of collisions
- `at_fault_collisions`: Count of aggressive/at-fault collisions
- `not_at_fault_collisions`: Count of defensive/unavoidable collisions
- `collision_types`: List of collision type labels
- `fault_classification`: List of 'at_fault' or 'not_at_fault' labels
- `collision_timesteps`: When collisions occurred
- `num_collisions`: Same as total_collisions

**Interpretation**:
- **At-fault collisions** indicate planning failures (should be 0)
- **Not-at-fault collisions** may be unavoidable (defensive behavior)
- Used in nuPlan and tuplan_garage PDMScorer for realistic fault assignment
- Critical for distinguishing aggressive vs. defensive driving

## Progress and Navigation Metrics

### Progress Score

**Purpose**: Measure task completion along reference path.

**Usage**:
```python
from admetrics.planning import calculate_progress_score

result = calculate_progress_score(
    executed_trajectory,
    reference_path,
    goal_threshold=2.0  # meters
)

print(f"Progress: {result['progress']:.2f}m")
print(f"Progress ratio: {result['progress_ratio']:.2%}")
print(f"Goal reached: {result['goal_reached']}")
print(f"Distance to goal: {result['final_distance_to_goal']:.2f}m")
```

**Returns**:
- `progress`: Distance traveled along reference path (meters)
- `progress_ratio`: Fraction of path completed [0, 1]
- `goal_reached`: Boolean indicating goal achievement
- `final_distance_to_goal`: Euclidean distance to final goal

**Interpretation**:
- Higher progress ratio is better
- Critical for task completion evaluation
- Used in nuPlan progress score calculation

### Route Completion

**Purpose**: Evaluate waypoint-based navigation.

**Usage**:
```python
from admetrics.planning import calculate_route_completion

# Define waypoints (e.g., intersections, landmarks)
waypoints = np.array([
    [100, 50],   # Waypoint 1
    [200, 150],  # Waypoint 2
    [300, 200]   # Waypoint 3
])

result = calculate_route_completion(
    executed_trajectory,
    waypoints,
    completion_radius=3.0  # meters
)

print(f"Waypoints reached: {result['num_waypoints_reached']}/{result['total_waypoints']}")
print(f"Completion rate: {result['completion_rate']:.2%}")
print(f"Waypoint status: {result['waypoint_status']}")
```

**Waypoint Completion Logic**:
- Waypoint is "reached" if vehicle passes within `completion_radius`
- Must visit waypoints in order (cannot skip)
- Returns boolean status for each waypoint

Implementation notes:
- `calculate_route_completion` enforces ordered waypoint passing: it scans the trajectory and marks waypoints sequentially. This matches common navigation benchmarks that require visiting waypoints in order.

**Interpretation**:
- 100% completion rate indicates successful navigation
- Used in CARLA route completion metric
- Critical for goal-directed driving scenarios

## Control Accuracy Metrics

### Lateral Deviation

**Purpose**: Measure cross-track error from reference path (lane keeping).

**Usage**:
```python
from admetrics.planning import calculate_lateral_deviation

result = calculate_lateral_deviation(
    actual_trajectory,
    reference_centerline,
    lane_width=3.5  # meters (default)
)

print(f"Mean lateral error: {result['mean_lateral_error']:.3f}m")
print(f"Max lateral error: {result['max_lateral_error']:.3f}m")
print(f"Std lateral error: {result['std_lateral_error']:.3f}m")
print(f"Lane keeping rate: {result['lane_keeping_rate']:.2%}")
```

**Calculation**:
- Projects each trajectory point onto nearest reference segment
- Computes perpendicular distance (cross-track error)
- Lane keeping: fraction of points within ±lane_width/2

Implementation notes:
- `calculate_lateral_deviation` projects points onto the nearest path segment (not just vertices) to compute cross-track error, improving accuracy on long straight segments.

**Interpretation**:
- Lower error indicates better path following
- Lane keeping rate should be close to 100%
- Critical for highway driving evaluation

### Heading Error

**Purpose**: Measure orientation/yaw accuracy.

**Usage**:
```python
from admetrics.planning import calculate_heading_error

# Headings in radians
predicted_headings = np.array([0.0, 0.1, 0.2, 0.3])  # rad
expert_headings = np.array([0.0, 0.12, 0.19, 0.31])  # rad

result = calculate_heading_error(predicted_headings, expert_headings)

print(f"Mean error: {result['mean_heading_error']:.4f} rad")
print(f"Mean error: {result['mean_heading_error_deg']:.2f}°")
print(f"Max error: {result['max_heading_error']:.4f} rad")
```

**Implementation Notes**:
- Properly handles angle wraparound ([-π, π])
- Uses `atan2` for correct quadrant calculation
- Returns both radians and degrees

**Interpretation**:
- Lower error indicates better orientation control
- Important for parallel parking, tight turns
- Typical good performance: < 5° mean error

### Velocity Error

**Purpose**: Measure speed control accuracy.

**Usage**:
```python
from admetrics.planning import calculate_velocity_error

result = calculate_velocity_error(predicted_velocities, expert_velocities)

print(f"Mean error: {result['mean_velocity_error']:.3f} m/s")
print(f"RMSE: {result['rmse_velocity']:.3f} m/s")
print(f"Max error: {result['max_velocity_error']:.3f} m/s")
```

**Interpretation**:
- Lower error indicates better speed tracking
- Important for traffic flow matching
- Typical good performance: < 1.0 m/s mean error

## Comfort Metrics

### Comfort Metrics (Acceleration & Jerk)

**Purpose**: Evaluate smoothness and passenger comfort across longitudinal and lateral dimensions.

**Enhanced Features** (nuPlan-aligned):
- **Longitudinal acceleration/jerk**: Forward/backward motion smoothness
- **Lateral acceleration**: Cornering comfort (v × yaw_rate)
- **Yaw rate**: Rotational smoothness
- **Yaw acceleration**: Angular jerk

**Usage**:
```python
from admetrics.planning import calculate_comfort_metrics

# Basic usage (longitudinal only - backward compatible)
result = calculate_comfort_metrics(
    trajectory,
    timestamps,
    max_acceleration=3.0,  # m/s² threshold
    max_jerk=2.0  # m/s³ threshold
)

# Enhanced usage with lateral metrics (nuPlan-style)
result = calculate_comfort_metrics(
    trajectory,
    timestamps,
    headings=headings,  # Required for lateral metrics (shape: T,)
    max_acceleration=4.0,  # nuPlan threshold (was 3.0)
    max_jerk=4.0,  # nuPlan threshold
    max_lateral_accel=4.0,  # Lateral comfort threshold
    max_yaw_rate=0.5,  # rad/s
    max_yaw_accel=1.0  # rad/s²
)

print(f"Longitudinal acceleration: {result['mean_acceleration']:.3f} m/s²")
print(f"Lateral acceleration: {result['mean_lateral_acceleration']:.3f} m/s²")
print(f"Yaw rate: {result['mean_yaw_rate']:.3f} rad/s")
print(f"Jerk: {result['mean_jerk']:.3f} m/s³")
print(f"Comfort rate: {result['comfort_rate']:.2%}")
print(f"Violations: {result['comfort_violations']}")
```

**Calculation**:
- **Longitudinal Acceleration**: `a_lon = dv/dt` (from trajectory speed)
- **Lateral Acceleration**: `a_lat = v × yaw_rate` (centripetal)
- **Yaw Rate**: `ω = dθ/dt` (from heading angle)
- **Yaw Acceleration**: `α = dω/dt`
- **Jerk**: `j = da/dt` (rate of change of acceleration)
- **Comfort violations**: Count of timesteps exceeding any threshold

**Returns**:
- `mean_acceleration`: Average longitudinal acceleration magnitude
- `max_acceleration`: Peak longitudinal acceleration
- `mean_lateral_acceleration`: Average lateral acceleration (if headings provided)
- `max_lateral_acceleration`: Peak lateral acceleration
- `mean_yaw_rate`: Average yaw rate (if headings provided)
- `max_yaw_rate`: Peak yaw rate
- `mean_yaw_acceleration`: Average yaw acceleration (if headings provided)
- `max_yaw_acceleration`: Peak yaw acceleration
- `mean_jerk`: Average jerk magnitude
- `max_jerk`: Peak jerk
- `comfort_rate`: Fraction of timesteps within all thresholds
- `comfort_violations`: Number of threshold violations

**Typical Thresholds**:

| Metric | Comfortable | Acceptable | Aggressive | nuPlan Standard |
|--------|-------------|------------|------------|-----------------|
| Longitudinal Accel | < 2.0 m/s² | < 3.0 m/s² | > 4.0 m/s² | **4.0 m/s²** |
| Lateral Accel | < 2.0 m/s² | < 3.0 m/s² | > 4.0 m/s² | **4.0 m/s²** |
| Jerk | < 2.0 m/s³ | < 3.0 m/s³ | > 4.0 m/s³ | **4.0 m/s³** |
| Yaw Rate | < 0.3 rad/s | < 0.5 rad/s | > 0.8 rad/s | **0.5 rad/s** |
| Yaw Accel | < 0.5 rad/s² | < 1.0 rad/s² | > 2.0 rad/s² | **1.0 rad/s²** |

**nuPlan Alignment**:
- Updated thresholds to match nuPlan benchmark (4.0 m/s² vs. previous 3.0 m/s²)
- Added lateral acceleration for cornering comfort evaluation
- Yaw metrics track rotational smoothness (critical for passenger comfort)
- All metrics normalized to 0-100 scale in driving score

**Interpretation**:
- Higher comfort rate is better (target: > 95%)
- Lateral acceleration critical for high-speed cornering
- Yaw rate violations indicate abrupt steering
- Used in nuPlan comfort score and Waymo Sim Agents benchmark
- Critical for passenger acceptance and ride quality

**Implementation Notes**:
- Lateral metrics require `headings` parameter (shape: T,)
- Heading unwrapping prevents 2π discontinuities in yaw rate calculation
- Backward compatible: omitting `headings` computes longitudinal-only metrics
- Uses central differences for smoother derivative estimation

## Composite Metrics

### Driving Score

**Purpose**: Comprehensive evaluation combining planning, safety, progress, and comfort.

**Modes**:
- **Default mode**: Basic composite scoring with collision rate
- **nuPlan mode**: Industry-standard scoring with fault classification, lateral comfort, drivable area compliance

**Formula** (default weights):
```
Driving Score = 0.3 * Planning + 0.4 * Safety + 0.2 * Progress + 0.1 * Comfort
```

**Formula** (nuPlan weights):
```
Driving Score = 0.25 * Planning + 0.40 * Safety + 0.20 * Progress + 0.15 * Comfort
```

**Usage (Default Mode)**:
```python
from admetrics.planning import calculate_driving_score

result = calculate_driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    reference_path,
    timestamps,
    weights={
        'planning': 0.3,
        'safety': 0.4,
        'progress': 0.2,
        'comfort': 0.1
    }
)

print(f"Driving Score: {result['driving_score']:.1f}/100")
print(f"\nComponent Scores:")
print(f"  Planning: {result['planning_accuracy']:.1f}/100")
print(f"  Safety: {result['safety_score']:.1f}/100")
print(f"  Progress: {result['progress_score']:.1f}/100")
print(f"  Comfort: {result['comfort_score']:.1f}/100")
```

**Usage (nuPlan Mode)**:
```python
# nuPlan-aligned scoring with enhanced metrics
result = calculate_driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    reference_path,
    timestamps,
    mode='nuplan',  # Enable nuPlan mode
    headings=ego_headings,  # Required for lateral comfort
    velocities=ego_velocities,  # Required for collision classification
    lane_centerline=lane_centerline,  # For direction compliance
    drivable_area=drivable_polygon,  # Optional Shapely polygon
    other_vehicles=other_vehicles  # For fault classification
)

print(f"Driving Score: {result['driving_score']:.1f}/100")
print(f"\nCore Metrics:")
print(f"  Planning: {result['planning_accuracy']:.1f}/100")
print(f"  Safety: {result['safety_score']:.1f}/100")
print(f"  Progress: {result['progress_score']:.1f}/100")
print(f"  Comfort: {result['comfort_score']:.1f}/100")

print(f"\nuPlan-Specific Metrics:")
print(f"  No at-fault collision: {result['no_at_fault_collision']}")
print(f"  Lateral comfort: {result['lateral_comfort_score']:.1f}/100")
print(f"  Drivable area compliance: {result['drivable_area_compliance']:.1f}/100")
print(f"  Driving direction compliance: {result['driving_direction_compliance']:.1f}/100")
```

**Component Calculations**:

| Mode | Planning | Safety | Progress | Comfort |
|------|----------|--------|----------|---------|
| **Default** | L2 distance (0-5m) | Collision rate | Progress ratio | Comfort rate (accel/jerk) |
| **nuPlan** | L2 distance (0-5m) | At-fault collisions (pass/fail) | Progress ratio | Comfort rate (accel/jerk/lateral) |

**nuPlan Mode Enhancements**:
1. **Safety**: Uses fault classification (at-fault = 0 score, no at-fault = 100 score)
2. **Comfort**: Includes lateral acceleration, yaw rate, yaw acceleration (4.0 m/s² thresholds)
3. **Additional Metrics**:
   - `no_at_fault_collision`: Boolean pass/fail criterion
   - `lateral_comfort_score`: Lateral acceleration quality [0-100]
   - `drivable_area_compliance`: Road boundary compliance [0-100]
   - `driving_direction_compliance`: Wrong-way detection [0-100]

**Parameters**:

| Parameter | Default Mode | nuPlan Mode | Description |
|-----------|--------------|-------------|-------------|
| `mode` | `'default'` | `'nuplan'` | Scoring mode |
| `headings` | Optional | **Required** | Heading angles (radians), shape (T,) |
| `velocities` | Optional | **Required** | Velocities (m/s), shape (T,) |
| `lane_centerline` | Optional | Recommended | Lane direction, shape (N, 2) |
| `drivable_area` | Optional | Recommended | Shapely polygon for road boundaries |
| `other_vehicles` | Optional | Recommended | List of vehicle dicts for fault classification |

**Custom Weighting Examples**:

```python
# Safety-critical applications (autonomous shuttles)
safety_focused = {
    'planning': 0.2,
    'safety': 0.6,
    'progress': 0.1,
    'comfort': 0.1
}

# Efficiency-focused (delivery robots)
efficiency_focused = {
    'planning': 0.2,
    'safety': 0.3,
    'progress': 0.4,
    'comfort': 0.1
}

# Comfort-focused (passenger vehicles)
comfort_focused = {
    'planning': 0.25,
    'safety': 0.35,
    'progress': 0.15,
    'comfort': 0.25
}

# nuPlan benchmark (recommended for research comparison)
nuplan_weights = {
    'planning': 0.25,
    'safety': 0.40,
    'progress': 0.20,
    'comfort': 0.15
}
```

**Interpretation**:
- Score range: 0-100 (higher is better)
- **Default mode**: Good for quick evaluation, basic collision detection
- **nuPlan mode**: Industry-standard, comprehensive evaluation with fault classification
- Adjust weights based on application requirements
- nuPlan mode requires additional data (headings, velocities) but provides more accurate safety assessment

**When to Use Each Mode**:
- **Default**: Quick prototyping, limited data availability, simple scenarios
- **nuPlan**: Benchmark comparison, research publication, comprehensive evaluation, fault analysis

## Imitation Learning Metrics

### Planning KL Divergence

**Purpose**: Measure how well learned policy matches expert behavior.

**Formula**:
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```
where P is expert distribution, Q is learned policy distribution.

**Usage**:
```python
from admetrics.planning import calculate_planning_kl_divergence

# Expert action distribution (e.g., steering histogram)
expert_actions = np.array([0.1, 0.3, 0.4, 0.15, 0.05])

# Learned policy distribution
learned_actions = np.array([0.12, 0.28, 0.38, 0.16, 0.06])

kl_div = calculate_planning_kl_divergence(learned_actions, expert_actions)

print(f"KL divergence: {kl_div:.4f}")
```

**Application Contexts**:
- Behavioral cloning evaluation
- Policy distillation quality
- Expert demonstration matching
- Action distribution analysis

**Interpretation**:
- Lower KL divergence = better policy match
- KL = 0 means perfect match
- Sensitive to distribution mismatches
- Not symmetric: KL(P||Q) ≠ KL(Q||P)

Implementation notes:
- `calculate_planning_kl_divergence(predicted, expert)` currently computes KL(expert || predicted). The function expects arrays in the order `(predicted_distribution, expert_distribution)` and returns KL(expert || predicted) — call accordingly or swap arguments if you prefer the reverse ordering.

## Safety and Additional Metrics

### Time-To-Collision (TTC)

**Purpose**: Estimate minimum time until collision under constant-velocity assumptions (safety early-warning).

**Usage**:
```python
from admetrics.planning import calculate_time_to_collision

result = calculate_time_to_collision(ego_trajectory, obstacles)
print(result['min_ttc'])
```

**Notes**:
- The implementation approximates TTC on a per-timestep basis assuming constant velocity between sampled points. For accurate TTC, provide high-frequency timestamps and adapt the method.

### Enhanced Time-To-Collision (Forward Projection)

**Purpose**: More accurate TTC estimation using forward trajectory projection over a time horizon (nuPlan-style). Accounts for both vehicles' motion over time.

**Formula**:
```
TTC = min{t : d(t) < collision_threshold, t ∈ [0, projection_horizon]}
```
where:
- `d(t)`: Distance between projected ego and obstacle positions at time t
- Projection: Linear extrapolation using current velocities
- Excludes stopped vehicles (v < 0.005 m/s)

**Usage**:
```python
from admetrics.planning import calculate_time_to_collision_enhanced

# Ego vehicle state
ego_trajectory = np.array([[0, 0], [1, 0], [2, 0]])  # shape (T, 2)
ego_velocities = np.array([10.0, 10.0, 10.0])  # m/s, shape (T,)
ego_headings = np.array([0.0, 0.0, 0.0])  # radians, shape (T,)
timestamps = np.array([0.0, 0.1, 0.2])  # seconds, shape (T,)

# Other vehicles
other_vehicles = [
    {
        'trajectory': np.array([[20, 0], [19, 0], [18, 0]]),
        'velocities': np.array([5.0, 5.0, 5.0]),
        'headings': np.array([np.pi, np.pi, np.pi])  # moving towards ego
    }
]

result = calculate_time_to_collision_enhanced(
    ego_trajectory=ego_trajectory,
    ego_velocities=ego_velocities,
    ego_headings=ego_headings,
    timestamps=timestamps,
    other_vehicles=other_vehicles,
    projection_horizon=1.0,  # Project 1 second forward
    projection_dt=0.3,  # Check every 0.3 seconds
    collision_threshold=5.0,  # Distance threshold for collision
    stopped_threshold=0.005  # Velocity threshold for stopped vehicles
)

print(f"Minimum TTC: {result['min_ttc']:.2f}s")
print(f"Critical timesteps: {result['critical_timesteps']}")
print(f"TTC values: {result['ttc_values']}")
```

**Parameters**:
- `projection_horizon`: How far ahead to project (default: 1.0s, nuPlan standard)
- `projection_dt`: Time resolution for projection (default: 0.3s)
- `collision_threshold`: Distance threshold for imminent collision (default: 5.0m)
- `stopped_threshold`: Speed below which vehicles are ignored (default: 0.005 m/s)

**Projection Logic**:
1. At each timestep, project both ego and other vehicles forward using:
   - `position(t+Δt) = position(t) + velocity × cos/sin(heading) × Δt`
2. Check distances at intervals: 0s, 0.3s, 0.6s, 0.9s (up to horizon)
3. TTC = smallest projection time where distance < threshold
4. Ignores stopped vehicles (parked cars don't create time pressure)

**Advantages over Basic TTC**:
- Accounts for both vehicles' motion (not just relative velocity)
- Uses actual heading information (not just velocity direction)
- Temporal resolution (0.3s intervals) matches nuPlan standards
- Excludes irrelevant stopped vehicles
- Returns full TTC profile, not just minimum

**Returns**:
- `min_ttc`: Minimum TTC across all timesteps (seconds, np.inf if none)
- `ttc_values`: List of TTC values at each trajectory timestep
- `critical_timesteps`: List of timesteps where TTC is finite

**Interpretation**:
- **TTC < 2.0s**: Critical warning (immediate action needed)
- **TTC < 3.0s**: Warning (monitor closely)
- **TTC > 5.0s**: Safe
- Used in nuPlan's time_to_collision_within_bound metric
- More accurate than constant-velocity TTC for maneuvering scenarios

### Lane Invasion / Off-road Rate

**Purpose**: Fraction of timesteps where the vehicle is outside lane centerline bounds.

**Usage**:
```python
from admetrics.planning import calculate_lane_invasion_rate

result = calculate_lane_invasion_rate(ego_trajectory, lane_centerlines)
print(result['invasion_rate'])
```

**Notes**:
- Checks whether trajectory samples lie within half the lane width of any provided centerline polyline.

### Collision Severity

**Purpose**: Estimate collision severity (proxy by relative speed at collision timesteps).

**Usage**:
```python
from admetrics.planning import calculate_collision_severity

result = calculate_collision_severity(ego_trajectory, obstacles)
print(result['max_severity'])
```

**Notes**:
- Uses relative speed at close distances as a simple proxy. For full severity analysis incorporate mass, impact geometry, and energy calculations.

### Kinematic Feasibility

**Purpose**: Quick checks for lateral acceleration and yaw-rate feasibility.

**Usage**:
```python
from admetrics.planning import check_kinematic_feasibility

result = check_kinematic_feasibility(ego_trajectory, timestamps)
print(result['feasible'], result['max_lateral_accel'])
```

**Notes**:
- Approximates curvature from finite differences of heading and computes lateral acceleration as v^2 * curvature. Use as a fast sanity check, not a full dynamics simulation.

### Distance to Road Edge

**Purpose**: Measure proximity to drivable area boundaries for safety margin evaluation (Waymo Sim Agents metric).

**Formula**:
```
distance_to_edge = signed_distance(trajectory_point, drivable_area_boundary)
  - Negative: Inside drivable area (safe)
  - Positive: Outside drivable area (violation)
  - Zero: On boundary
```

**Usage**:
```python
from admetrics.planning import calculate_distance_to_road_edge
from shapely.geometry import Polygon

# Option 1: Using Shapely polygon (preferred)
drivable_area = Polygon([
    [0, -5], [100, -5], [100, 5], [0, 5]  # Road boundary
])

result = calculate_distance_to_road_edge(
    trajectory=ego_trajectory,
    drivable_area=drivable_area,
    vehicle_width=2.0  # Account for vehicle dimensions
)

# Option 2: Using lane centerline (fallback)
lane_centerline = np.array([
    [0, 0], [50, 0], [100, 0]
])

result = calculate_distance_to_road_edge(
    trajectory=ego_trajectory,
    lane_centerline=lane_centerline,
    lane_width=3.5,
    vehicle_width=2.0
)

print(f"Mean distance to edge: {result['mean_distance_to_edge']:.2f}m")
print(f"Min distance to edge: {result['min_distance_to_edge']:.2f}m")
print(f"Drivable area violations: {result['num_violations']}")
print(f"Violation rate: {result['violation_rate']:.2%}")
```

**Parameters**:
- `drivable_area`: Shapely Polygon representing drivable region (preferred method)
- `lane_centerline`: Fallback if no polygon available (np.array, shape (N, 2))
- `lane_width`: Lane width in meters (used with centerline method)
- `vehicle_width`: Vehicle width to account for body extent (default: 2.0m)

**Calculation Methods**:
1. **Polygon-based** (requires Shapely):
   - Uses `polygon.exterior.distance()` for signed distance
   - Negative = inside (safe), positive = outside (violation)
   - Most accurate for complex road geometries

2. **Centerline-based** (fallback):
   - Computes perpendicular distance from trajectory to centerline
   - Assumes straight lane boundaries at ±lane_width/2
   - Less accurate but doesn't require HD map polygons

**Returns**:
- `mean_distance_to_edge`: Average signed distance (meters)
- `min_distance_to_edge`: Closest approach to boundary (meters)
- `max_distance_to_edge`: Farthest distance (for off-road violations)
- `distances`: Array of distances at each trajectory point
- `num_violations`: Count of timesteps outside drivable area
- `violation_rate`: Fraction of trajectory outside drivable area
- `violation_timesteps`: Indices where violations occur

**Interpretation**:
- **Negative (inside)**: Safe, larger magnitude = more margin
- **Near zero**: Close to boundary (warning)
- **Positive (outside)**: Drivable area violation (critical)
- Target: Keep min_distance < -1.0m (1m safety margin inside boundary)
- Used in Waymo Sim Agents Challenge for realism evaluation
- Critical for road edge safety and HD map compliance

### Driving Direction Compliance

**Purpose**: Detect wrong-way driving by comparing trajectory heading to expected lane direction (nuPlan metric).

**Formula**:
```
heading_error = |heading_ego - heading_lane|
compliance_score = {
    1.0  if total_wrong_way_distance < 2m
    0.5  if 2m ≤ total_wrong_way_distance < 6m
    0.0  if total_wrong_way_distance ≥ 6m
}
```

**Usage**:
```python
from admetrics.planning import calculate_driving_direction_compliance

# Lane direction defined by centerline segments
lane_centerline = np.array([
    [0, 0], [50, 0], [100, 0]  # Lane heading: 0 radians (east)
])

result = calculate_driving_direction_compliance(
    trajectory=ego_trajectory,
    headings=ego_headings,  # shape (T,), radians
    lane_centerline=lane_centerline,
    angle_threshold=np.pi/2,  # 90 degrees = wrong-way
    short_distance_threshold=2.0,  # nuPlan: < 2m ok
    medium_distance_threshold=6.0   # nuPlan: 2-6m warning
)

print(f"Compliance score: {result['compliance_score']:.1f}")
print(f"Wrong-way distance: {result['total_wrong_way_distance']:.2f}m")
print(f"Violation rate: {result['violation_rate']:.2%}")
print(f"Max heading error: {result['max_heading_error']:.2f} rad")
```

**Parameters**:
- `lane_centerline`: Expected lane direction (shape: N, 2)
- `angle_threshold`: Heading difference threshold for wrong-way (default: π/2 = 90°)
- `short_distance_threshold`: Distance threshold for score=1.0 (default: 2.0m, nuPlan)
- `medium_distance_threshold`: Distance threshold for score=0.5 (default: 6.0m, nuPlan)

**Detection Logic**:
1. Compute expected lane heading from centerline segments
2. Find nearest centerline segment for each trajectory point
3. Calculate heading error: `|heading_ego - heading_lane|`
4. If error > 90°, mark as wrong-way driving
5. Accumulate wrong-way distance: `Σ segment_length` (where wrong-way)
6. Apply nuPlan thresholds for compliance score

**Returns**:
- `compliance_score`: 1.0 (compliant), 0.5 (minor violation), 0.0 (major violation)
- `total_wrong_way_distance`: Total distance driven in wrong direction (meters)
- `num_violations`: Count of wrong-way timesteps
- `violation_rate`: Fraction of trajectory in wrong direction
- `violation_timesteps`: Indices where wrong-way detected
- `heading_errors`: Array of heading errors at each point (radians)
- `max_heading_error`: Maximum heading deviation (radians)
- `mean_heading_error`: Average heading error (radians)

**nuPlan Scoring**:
- **Score 1.0**: Excellent (< 2m wrong-way)
- **Score 0.5**: Acceptable (2-6m wrong-way, e.g., lane change)
- **Score 0.0**: Failure (≥ 6m wrong-way, e.g., driving opposite direction)

**Interpretation**:
- Critical for highway on-ramps and divided roads
- Detects serious planning failures (opposite lane, wrong-way on highway)
- Used in nuPlan's drivable_area_compliance metric
- Should be 1.0 for safe operation (< 2m total wrong-way distance)
- Violations indicate severe planning errors or map misalignment

## Benchmark References

### nuPlan

**Website**: https://www.nuscenes.org/nuplan

**Key Metrics**:
- **Closed-loop score**: Composite metric (0-100)
  - No collisions (pass/fail)
  - Progress (% of route completed)
  - Driving within lane (% of time)
  - Comfort (smooth acceleration/jerk)
- **Open-loop score**: L2 distance to expert

**Evaluation Protocol**:
- 14 scenario types (lane following, turning, parking, etc.)
- 40 scenarios per type
- Pass threshold: > 60% overall score

**Our Implementation**:
```python
# nuPlan-style evaluation
score = calculate_driving_score(
    pred_traj, expert_traj, obstacles, ref_path, timestamps,
    weights={'planning': 0.25, 'safety': 0.40, 'progress': 0.20, 'comfort': 0.15}
)
```

### CARLA

**Website**: https://leaderboard.carla.org/

**Key Metrics**:
- **Route completion**: % of route successfully completed
- **Infraction score**: Penalty-based system
  - Collision with pedestrian: -0.50
  - Collision with vehicle: -0.40
  - Collision with static: -0.25
  - Red light violation: -0.30
  - Stop sign violation: -0.20

**Driving Score Formula**:
```
DS = Route Completion × Π (1 - infraction_penalty_i)
```

**Our Implementation**:
```python
# CARLA-style evaluation
route_result = calculate_route_completion(traj, waypoints)
collision_result = calculate_collision_rate(traj, obstacles)

# Compute CARLA-like score
carla_score = route_result['completion_rate'] * (
    1.0 if collision_result['collision_rate'] == 0 else 0.0
)
```

### Waymo Open Dataset

**Website**: https://waymo.com/open/

**Planning Metrics**:
- **Displacement error**: L2 distance at multiple horizons (1s, 3s, 5s, 8s)
- **Heading error**: Angle difference in radians
- **Kinematic feasibility**: Check acceleration/jerk limits

**Sim Agents Challenge**:
- **Realism score**: How realistic is the behavior
- **Kinematic metrics**: Velocity, acceleration validity
- **Interaction quality**: Multi-agent scenario handling

### Argoverse 2

**Website**: https://www.argoverse.org/

**Scenario-based Evaluation**:
- **Unprotected turns**: Yielding behavior
- **Lane changes**: Gap acceptance, merge quality
- **Pedestrian crossing**: Stopping distance, yielding

**Metrics**:
- Scenario-specific success rates
- Interaction-aware metrics
- Multi-agent coordination

## Best Practices

### 1. Metric Selection

**Choose metrics based on application**:

| Application | Priority Metrics |
|-------------|-----------------|
| Urban robotaxi | Safety (collision), Comfort, Progress |
| Highway autopilot | Lateral deviation, Velocity error, Comfort |
| Parking assistant | L2 distance, Heading error, Collision |
| Delivery robot | Progress, Route completion, Safety |

### 2. Evaluation Protocol

**Comprehensive evaluation checklist**:

```python
# 1. Planning accuracy
l2 = calculate_l2_distance(pred, expert)
ade_fde = average_displacement_error_planning(pred, expert)

# 2. Safety (critical)
collision = calculate_collision_rate(pred, obstacles)
assert collision['collision_rate'] == 0.0, "Safety violation!"

ttc = calculate_time_to_collision(pred, obstacles)
print(f"Minimum TTC: {ttc['min_ttc']:.2f}s")

severity = calculate_collision_severity(pred, obstacles)
if severity['max_severity'] > 0:
    print(f"Warning: Collision severity {severity['max_severity']:.2f}")

# 3. Task completion
progress = calculate_progress_score(pred, ref_path)
route = calculate_route_completion(pred, waypoints)

# 4. Control quality
lateral = calculate_lateral_deviation(pred, ref_centerline)
heading = calculate_heading_error(pred_headings, expert_headings)

# Lane adherence
lane_invasion = calculate_lane_invasion_rate(pred, lane_centerlines)
print(f"Lane invasion rate: {lane_invasion['invasion_rate']:.2%}")

# 5. Comfort
comfort = calculate_comfort_metrics(pred, timestamps)

# 6. Kinematic feasibility
feasibility = check_kinematic_feasibility(pred, timestamps)
if not feasibility['feasible']:
    print(f"Warning: Kinematically infeasible trajectory!")
    print(f"  Max lateral accel: {feasibility['max_lateral_accel']:.2f} m/s²")

# 7. Overall assessment
overall = calculate_driving_score(pred, expert, obstacles, ref_path, timestamps)
```

### 3. Temporal Considerations

**Short-term vs. Long-term**:
- **Short-term** (< 1s): Reactive control, obstacle avoidance
- **Medium-term** (1-3s): Tactical planning, lane changes
- **Long-term** (3-8s): Strategic planning, route following

**Weighting strategies**:
```python
# Emphasize long-term planning
weights_longterm = np.linspace(1.0, 3.0, T)

# Emphasize near-term control
weights_nearterm = np.linspace(3.0, 1.0, T)

# Exponential decay (recent timesteps critical)
weights_exponential = np.exp(-0.1 * np.arange(T))
```

### 4. Scenario Coverage

**Ensure diverse scenarios**:
- Lane following (straight, curved)
- Intersections (protected/unprotected turns)
- Lane changes (left/right, single/multiple)
- Parking (parallel, perpendicular, angled)
- Traffic interactions (yielding, merging)
- Adverse conditions (rain, night, occlusion)

### 5. Statistical Significance

**Report with uncertainty**:
```python
# Multiple runs
scores = []
for seed in range(100):
    np.random.seed(seed)
    # ... run evaluation
    result = calculate_driving_score(pred, expert, obstacles, ref_path, t)
    scores.append(result['driving_score'])

print(f"Driving Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
print(f"95% CI: [{np.percentile(scores, 2.5):.1f}, {np.percentile(scores, 97.5):.1f}]")
```

### 6. Ablation Studies

**Component analysis**:
```python
# Baseline model
baseline_score = calculate_driving_score(baseline_pred, expert, obstacles, ref, t)

# Improved planning module
improved_planning_score = calculate_driving_score(improved_pred, expert, obstacles, ref, t)

# Compute improvement
improvement = improved_planning_score['driving_score'] - baseline_score['driving_score']
print(f"Planning improvement: +{improvement:.1f} points")

# Per-component improvement
for component in ['planning_accuracy', 'safety_score', 'progress_score', 'comfort_score']:
    delta = improved_planning_score[component] - baseline_score[component]
    print(f"  {component}: {delta:+.1f}")
```

### 7. Real-world vs. Simulation

**Sim-to-real considerations**:
- Simulation metrics are optimistic (perfect sensing, deterministic)
- Real-world: Add sensor noise, localization error, actuation delay
- Domain adaptation: Evaluate on both sim and real data
- Reality gap: Expect 10-20% performance drop

### 8. Computational Efficiency

**Optimization tips**:
```python
# Vectorized computation (fast)
dists = np.linalg.norm(pred_traj - expert_traj, axis=1)
l2 = np.mean(dists)

# Avoid loops (slow)
# for i in range(len(pred_traj)):
#     dist = np.linalg.norm(pred_traj[i] - expert_traj[i])

# Batch processing
results = []
for traj in trajectories:
    result = calculate_driving_score(traj, expert, obstacles, ref, t)
    results.append(result)
```

### 9. Failure Analysis

**Debugging poor performance**:
```python
# Identify failure modes
result = calculate_driving_score(pred, expert, obstacles, ref, t)

if result['safety_score'] < 100:
    print("SAFETY FAILURE: Collision detected")
    coll = calculate_collision_rate(pred, obstacles)
    print(f"  First collision at t={coll['first_collision']}")

if result['progress_score'] < 50:
    print("PROGRESS FAILURE: Route incomplete")
    prog = calculate_progress_score(pred, ref)
    print(f"  Only {prog['progress_ratio']:.1%} completed")

if result['comfort_score'] < 70:
    print("COMFORT FAILURE: Aggressive driving")
    comfort = calculate_comfort_metrics(pred, t)
    print(f"  {comfort['comfort_violations']} violations")
```

### 10. Reporting Standards

**Complete evaluation report should include**:
1. **Dataset**: Name, version, scenario types, number of samples
2. **Model**: Architecture, parameters, training details
3. **Metrics**: All relevant metrics with mean ± std
4. **Scenarios**: Performance breakdown by scenario type
5. **Failures**: Collision rate, failure cases, edge cases
6. **Comparison**: Baseline models, state-of-the-art benchmarks
7. **Visualizations**: Trajectory plots, error distributions
8. **Reproducibility**: Code, hyperparameters, random seeds

## Example: Complete Evaluation Pipeline

```python
import numpy as np
from admetrics.planning import (
    calculate_l2_distance,
    calculate_collision_rate,
    calculate_progress_score,
    calculate_lateral_deviation,
    calculate_comfort_metrics,
    calculate_driving_score,
    calculate_time_to_collision,
    calculate_lane_invasion_rate,
    calculate_collision_severity,
    check_kinematic_feasibility,
)

def evaluate_end_to_end_model(model, test_scenarios):
    """Complete evaluation pipeline."""
    
    results = {
        'l2_distances': [],
        'collision_rates': [],
        'progress_scores': [],
        'lateral_errors': [],
        'comfort_scores': [],
        'driving_scores': [],
        'ttc_values': [],
        'lane_invasions': [],
        'kinematic_feasible': [],
    }
    
    for scenario in test_scenarios:
        # Get model prediction
        pred_traj = model.predict(scenario['sensor_data'])
        
        # Extract ground truth
        expert_traj = scenario['expert_trajectory']
        obstacles = scenario['obstacles']
        ref_path = scenario['reference_path']
        timestamps = scenario['timestamps']
        lane_centerlines = scenario.get('lane_centerlines', [])
        
        # Compute metrics
        results['l2_distances'].append(
            calculate_l2_distance(pred_traj, expert_traj)
        )
        
        coll = calculate_collision_rate(pred_traj, obstacles)
        results['collision_rates'].append(coll['collision_rate'])
        
        prog = calculate_progress_score(pred_traj, ref_path)
        results['progress_scores'].append(prog['progress_ratio'])
        
        lat = calculate_lateral_deviation(pred_traj, ref_path)
        results['lateral_errors'].append(lat['mean_lateral_error'])
        
        comfort = calculate_comfort_metrics(pred_traj, timestamps)
        results['comfort_scores'].append(comfort['comfort_rate'])
        
        overall = calculate_driving_score(pred_traj, expert_traj, obstacles, ref_path, timestamps)
        results['driving_scores'].append(overall['driving_score'])
        
        # Additional safety metrics
        ttc = calculate_time_to_collision(pred_traj, obstacles)
        results['ttc_values'].append(ttc['min_ttc'])
        
        if lane_centerlines:
            invasion = calculate_lane_invasion_rate(pred_traj, lane_centerlines)
            results['lane_invasions'].append(invasion['invasion_rate'])
        
        feasibility = check_kinematic_feasibility(pred_traj, timestamps)
        results['kinematic_feasible'].append(1.0 if feasibility['feasible'] else 0.0)
    
    # Aggregate results
    report = {
        'L2 Distance': f"{np.mean(results['l2_distances']):.3f} ± {np.std(results['l2_distances']):.3f}m",
        'Collision Rate': f"{np.mean(results['collision_rates']):.2%}",
        'Progress': f"{np.mean(results['progress_scores']):.1%}",
        'Lateral Error': f"{np.mean(results['lateral_errors']):.3f}m",
        'Comfort': f"{np.mean(results['comfort_scores']):.1%}",
        'Driving Score': f"{np.mean(results['driving_scores']):.1f}/100",
        'Min TTC': f"{np.mean(results['ttc_values']):.2f}s",
        'Kinematic Feasibility': f"{np.mean(results['kinematic_feasible']):.1%}",
    }
    
    if results['lane_invasions']:
        report['Lane Invasion Rate'] = f"{np.mean(results['lane_invasions']):.2%}"
    
    return report

# Usage
# report = evaluate_end_to_end_model(my_model, test_data)
# for metric, value in report.items():
#     print(f"{metric}: {value}")
```

## References

1. **nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles**
   - Caesar, H., Kabzan, J., Tan, K.S., Fong, W.K., Wolff, E., Lang, A., Fletcher, L., Beijbom, O., & Omari, S. (2021)
   - https://arxiv.org/abs/2106.11810
   - https://www.nuscenes.org/nuplan
   - 1200h driving data from 4 cities (Boston, Pittsburgh, Las Vegas, Singapore), closed-loop simulation framework

2. **CARLA: An open urban driving simulator**
   - Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017)
   - CoRL 2017
   - https://arxiv.org/abs/1711.03938
   - https://carla.org
   - Open-source simulator for autonomous driving research

3. **Scalability in perception for autonomous driving: Waymo open dataset**
   - Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H., Timofeev, A., Ettinger, S., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1912.04838
   - https://waymo.com/open
   - Large-scale dataset with 1,150 scenes, diverse urban/suburban environments, high-quality LiDAR and camera data

4. **Argoverse 2: Next generation datasets for self-driving perception and forecasting**
   - Wilson, B., Qi, W., Agarwal, T., Lambert, J., Singh, J., Khandelwal, S., Pan, B., Kumar, R., Hartnett, A., Khandelwal, S., Pan, B., et al. (2023)
   - NeurIPS 2021 Datasets and Benchmarks Track
   - https://arxiv.org/abs/2301.00493
   - https://argoverse.github.io/user-guide/
   - Three datasets: Sensor Dataset (1,000 sequences), Lidar Dataset (20,000 sequences), Motion Forecasting Dataset (250,000 scenarios)

5. **Planning-oriented autonomous driving (UniAD)**
   - Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., Lu, L., Jia, X., Liu, Q., Dai, J., Qiao, Y., & Li, H. (2023)
   - CVPR 2023 (Award Candidate - Best Paper)
   - https://arxiv.org/abs/2212.10156
   - https://github.com/OpenDriveLab/UniAD
   - Unified autonomous driving framework with hierarchical task planning (perception, prediction, planning)

6. **End-to-end driving via conditional imitation learning**
   - Codevilla, F., Müller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018)
   - ICRA 2018
   - https://arxiv.org/abs/1710.02410
   - Imitation learning approach for autonomous driving


nuPlan Paper: https://arxiv.org/abs/2106.11810
Waymo Sim Agents: https://waymo.com/open/challenges/2024/sim-agents/
tuplan_garage (PDM-Closed): https://github.com/autonomousvision/tuplan_garage
nuPlan Devkit: https://github.com/motional/nuplan-devkit

## See Also

- [DETECTION_METRICS.md](DETECTION_METRICS.md) - Object detection metrics
- [TRACKING_METRICS.md](TRACKING_METRICS.md) - Multi-object tracking metrics
- [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) - Motion prediction metrics
- [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) - 3D occupancy metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
