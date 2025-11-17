"""
Example: nuPlan-style Composite Driving Score Evaluation

This example demonstrates how to use the nuPlan scoring mode for comprehensive
autonomous driving evaluation, including:
- At-fault collision classification
- Lateral comfort metrics
- Drivable area compliance
- Driving direction compliance
"""

import numpy as np
from admetrics.planning import calculate_driving_score

# Simulate a simple driving scenario
np.random.seed(42)

# Time series
timestamps = np.linspace(0, 5, 50)  # 5 seconds, 10 Hz
T = len(timestamps)

# Ego vehicle trajectory (following a curved path)
t = np.linspace(0, 2 * np.pi, T)
predicted_trajectory = np.column_stack([
    np.linspace(0, 50, T),  # x: moving forward
    5 * np.sin(t / 3)       # y: gentle curve
])

# Expert trajectory (similar but slightly smoother)
expert_trajectory = np.column_stack([
    np.linspace(0, 50, T),
    4.5 * np.sin(t / 3)
])

# Reference path (road centerline)
reference_path = np.column_stack([
    np.linspace(0, 60, 100),
    np.zeros(100)  # Straight road
])

# Lane centerline (for direction compliance)
lane_centerline = np.column_stack([
    np.linspace(0, 60, 20),
    np.zeros(20)
])

# Ego vehicle state
ego_velocities = np.full(T, 10.0)  # 10 m/s constant speed
ego_headings = np.arctan2(
    np.gradient(predicted_trajectory[:, 1]),
    np.gradient(predicted_trajectory[:, 0])
)

# Obstacles (static and dynamic)
static_obstacles = [
    np.array([[25, 10]]),  # Parked car to the side
    np.array([[40, -8]])   # Road sign
]

# Other vehicles for collision classification
other_vehicles = [
    {
        'trajectory': np.column_stack([
            np.linspace(30, 35, T),  # Slower vehicle ahead
            np.zeros(T)
        ]),
        'velocities': np.full(T, 5.0),  # 5 m/s
        'headings': np.zeros(T)
    },
    {
        'trajectory': np.column_stack([
            np.linspace(-10, 20, T),  # Crossing vehicle
            np.linspace(-5, 5, T)
        ]),
        'velocities': np.full(T, 8.0),
        'headings': np.full(T, np.pi / 4)  # 45 degree angle
    }
]

# Combine obstacles for basic collision detection
all_obstacles = static_obstacles + [v['trajectory'] for v in other_vehicles]

print("=" * 70)
print("nuPlan-Style Driving Score Evaluation")
print("=" * 70)
print()

# ============================================================================
# Example 1: Default Mode (Basic Scoring)
# ============================================================================
print("Example 1: Default Mode")
print("-" * 70)

default_result = calculate_driving_score(
    predicted_trajectory=predicted_trajectory,
    expert_trajectory=expert_trajectory,
    obstacles=all_obstacles,
    reference_path=reference_path,
    timestamps=timestamps,
    mode='default'
)

print(f"Overall Driving Score: {default_result['driving_score']:.1f}/100")
print(f"\nComponent Scores:")
print(f"  Planning Accuracy:  {default_result['planning_accuracy']:.1f}/100")
print(f"  Safety Score:       {default_result['safety_score']:.1f}/100")
print(f"  Progress Score:     {default_result['progress_score']:.1f}/100")
print(f"  Comfort Score:      {default_result['comfort_score']:.1f}/100")
print()

# ============================================================================
# Example 2: nuPlan Mode (Comprehensive Evaluation)
# ============================================================================
print("Example 2: nuPlan Mode (Full Metrics)")
print("-" * 70)

nuplan_result = calculate_driving_score(
    predicted_trajectory=predicted_trajectory,
    expert_trajectory=expert_trajectory,
    obstacles=all_obstacles,
    reference_path=reference_path,
    timestamps=timestamps,
    mode='nuplan',
    headings=ego_headings,
    velocities=ego_velocities,
    lane_centerline=lane_centerline,
    other_vehicles=other_vehicles
)

print(f"Overall Driving Score: {nuplan_result['driving_score']:.1f}/100")
print(f"\nCore Component Scores:")
print(f"  Planning Accuracy:  {nuplan_result['planning_accuracy']:.1f}/100")
print(f"  Safety Score:       {nuplan_result['safety_score']:.1f}/100")
print(f"  Progress Score:     {nuplan_result['progress_score']:.1f}/100")
print(f"  Comfort Score:      {nuplan_result['comfort_score']:.1f}/100")

print(f"\nnuPlan-Specific Metrics:")
print(f"  No At-Fault Collision:        {nuplan_result['no_at_fault_collision']}")
print(f"  Lateral Comfort Score:        {nuplan_result['lateral_comfort_score']:.1f}/100")
print(f"  Driving Direction Compliance: {nuplan_result['driving_direction_compliance']:.1f}/100")

# Note: drivable_area_compliance only available if drivable_area provided
print()

# ============================================================================
# Example 3: nuPlan Mode with Custom Weights
# ============================================================================
print("Example 3: nuPlan Mode with Safety-Focused Weighting")
print("-" * 70)

safety_focused_result = calculate_driving_score(
    predicted_trajectory=predicted_trajectory,
    expert_trajectory=expert_trajectory,
    obstacles=all_obstacles,
    reference_path=reference_path,
    timestamps=timestamps,
    mode='nuplan',
    weights={
        'planning': 0.15,
        'safety': 0.60,    # Emphasize safety
        'progress': 0.15,
        'comfort': 0.10
    },
    headings=ego_headings,
    velocities=ego_velocities,
    lane_centerline=lane_centerline,
    other_vehicles=other_vehicles
)

print(f"Overall Driving Score: {safety_focused_result['driving_score']:.1f}/100")
print(f"  (Safety weight: 60% vs. default 40%)")
print()

# ============================================================================
# Example 4: Comparing Default vs nuPlan Mode
# ============================================================================
print("Example 4: Mode Comparison")
print("-" * 70)
print(f"{'Metric':<30} {'Default':<15} {'nuPlan':<15}")
print("-" * 70)
print(f"{'Overall Score':<30} {default_result['driving_score']:<15.1f} {nuplan_result['driving_score']:<15.1f}")
print(f"{'Planning Accuracy':<30} {default_result['planning_accuracy']:<15.1f} {nuplan_result['planning_accuracy']:<15.1f}")
print(f"{'Safety Score':<30} {default_result['safety_score']:<15.1f} {nuplan_result['safety_score']:<15.1f}")
print(f"{'Progress Score':<30} {default_result['progress_score']:<15.1f} {nuplan_result['progress_score']:<15.1f}")
print(f"{'Comfort Score':<30} {default_result['comfort_score']:<15.1f} {nuplan_result['comfort_score']:<15.1f}")
print()

# ============================================================================
# Example 5: With Shapely Drivable Area
# ============================================================================
print("Example 5: nuPlan Mode with Drivable Area Polygon")
print("-" * 70)

try:
    from shapely.geometry import Polygon
    
    # Define drivable area (road boundaries)
    drivable_area = Polygon([
        [0, -7],    # Road extends from y=-7 to y=7
        [60, -7],
        [60, 7],
        [0, 7]
    ])
    
    full_nuplan_result = calculate_driving_score(
        predicted_trajectory=predicted_trajectory,
        expert_trajectory=expert_trajectory,
        obstacles=all_obstacles,
        reference_path=reference_path,
        timestamps=timestamps,
        mode='nuplan',
        headings=ego_headings,
        velocities=ego_velocities,
        lane_centerline=lane_centerline,
        drivable_area=drivable_area,
        other_vehicles=other_vehicles
    )
    
    print(f"Overall Driving Score: {full_nuplan_result['driving_score']:.1f}/100")
    print(f"\nAll nuPlan Metrics:")
    print(f"  No At-Fault Collision:        {full_nuplan_result['no_at_fault_collision']}")
    print(f"  Lateral Comfort Score:        {full_nuplan_result['lateral_comfort_score']:.1f}/100")
    print(f"  Drivable Area Compliance:     {full_nuplan_result['drivable_area_compliance']:.1f}/100")
    print(f"  Driving Direction Compliance: {full_nuplan_result['driving_direction_compliance']:.1f}/100")
    
except ImportError:
    print("Shapely not installed - skipping drivable area example")
    print("Install with: pip install shapely")

print()
print("=" * 70)
print("Evaluation Complete!")
print("=" * 70)
print()
print("Key Differences Between Modes:")
print("  - Default: Uses simple collision rate, basic comfort metrics")
print("  - nuPlan:  Uses fault classification, lateral comfort, compliance checks")
print("  - nuPlan mode provides more detailed safety assessment")
print("  - nuPlan mode aligns with industry benchmark standards")
