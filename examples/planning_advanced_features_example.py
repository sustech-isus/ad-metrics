"""
Example: Multi-Horizon Evaluation and Smoothing

This example demonstrates:
1. Multi-horizon ADE/FDE evaluation (Waymo/nuPlan style)
2. Savitzky-Golay smoothing for comfort metrics (tuplan_garage style)
3. Interaction metrics for multi-agent scenarios
"""

import numpy as np
from admetrics.planning import (
    average_displacement_error_planning,
    calculate_comfort_metrics,
    calculate_interaction_metrics
)

print("=" * 70)
print("Multi-Horizon Evaluation and Advanced Features")
print("=" * 70)
print()

# ============================================================================
# Example 1: Multi-Horizon ADE/FDE Evaluation (Waymo/nuPlan Standard)
# ============================================================================
print("Example 1: Multi-Horizon ADE/FDE Evaluation")
print("-" * 70)

# Generate sample trajectories (simulating 8 seconds at 10Hz = 80 timesteps)
np.random.seed(42)
sampling_rate = 10  # Hz
duration = 8  # seconds
T = sampling_rate * duration  # 80 timesteps

# Predicted trajectory with some error that increases over time
t = np.linspace(0, duration, T)
predicted_traj = np.column_stack([
    t * 5,  # x: 5 m/s forward
    0.5 * np.sin(t / 2) + 0.1 * np.random.randn(T)  # y: slight curve with noise
])

# Expert trajectory (smoother, slightly different path)
expert_traj = np.column_stack([
    t * 5,
    0.4 * np.sin(t / 2)
])

# Define standard evaluation horizons
horizons_seconds = [1, 3, 5, 8]  # Waymo/nuPlan standard
horizons_timesteps = [int(h * sampling_rate) for h in horizons_seconds]

# Evaluate at multiple horizons
result = average_displacement_error_planning(
    predicted_traj,
    expert_traj,
    horizons=horizons_timesteps
)

print(f"Overall Metrics:")
print(f"  ADE: {result['ADE']:.3f}m")
print(f"  FDE: {result['FDE']:.3f}m")
print()

print(f"Horizon-Specific Metrics:")
for h_sec, h_step in zip(horizons_seconds, horizons_timesteps):
    ade_key = f'ADE_{h_step}'
    fde_key = f'FDE_{h_step}'
    print(f"  {h_sec}s horizon (timestep {h_step:2d}):")
    print(f"    ADE = {result[ade_key]:.3f}m")
    print(f"    FDE = {result[fde_key]:.3f}m")

print()
print("Interpretation:")
print("  - Short-term (1s):  Critical for immediate safety")
print("  - Mid-term (3s):    Lane changes and tactical maneuvers")
print("  - Long-term (5s):   Route following and navigation")
print("  - Strategic (8s):   Motion forecasting and prediction")
print()

# ============================================================================
# Example 2: Comfort Metrics with Savitzky-Golay Smoothing
# ============================================================================
print("Example 2: Comfort Metrics with Smoothing (tuplan_garage style)")
print("-" * 70)

# Generate noisy trajectory
t_comfort = np.linspace(0, 5, 50)
noisy_traj = np.column_stack([
    t_comfort * 10 + 0.2 * np.random.randn(50),  # Noisy forward motion
    np.sin(t_comfort) + 0.15 * np.random.randn(50)  # Noisy lateral motion
])

timestamps = t_comfort

# Calculate comfort metrics WITHOUT smoothing
result_no_smooth = calculate_comfort_metrics(
    noisy_traj,
    timestamps,
    use_smoothing=False,
    include_lateral=True
)

# Calculate comfort metrics WITH Savitzky-Golay smoothing
result_with_smooth = calculate_comfort_metrics(
    noisy_traj,
    timestamps,
    use_smoothing=True,
    smoothing_window=15,  # Window of 15 samples (~1.5s at 10Hz)
    smoothing_order=2,    # Polynomial order 2
    include_lateral=True
)

print("Comparison: No Smoothing vs. Savitzky-Golay Smoothing")
print()
print(f"{'Metric':<30} {'No Smooth':<15} {'With Smooth':<15} {'Improvement':<15}")
print("-" * 75)

metrics_to_compare = [
    ('max_longitudinal_accel', 'Max Long. Accel (m/s²)'),
    ('max_lateral_accel', 'Max Lat. Accel (m/s²)'),
    ('max_jerk', 'Max Jerk (m/s³)'),
    ('max_yaw_rate', 'Max Yaw Rate (rad/s)'),
    ('comfort_rate', 'Comfort Rate (%)'),
]

for key, label in metrics_to_compare:
    no_smooth_val = result_no_smooth[key]
    smooth_val = result_with_smooth[key]
    
    if key == 'comfort_rate':
        no_smooth_str = f"{no_smooth_val * 100:.1f}"
        smooth_str = f"{smooth_val * 100:.1f}"
        improvement = f"+{(smooth_val - no_smooth_val) * 100:.1f}%"
    else:
        no_smooth_str = f"{no_smooth_val:.3f}"
        smooth_str = f"{smooth_val:.3f}"
        pct_change = ((no_smooth_val - smooth_val) / no_smooth_val * 100) if no_smooth_val > 0 else 0
        improvement = f"-{pct_change:.1f}%" if pct_change > 0 else f"+{abs(pct_change):.1f}%"
    
    print(f"{label:<30} {no_smooth_str:<15} {smooth_str:<15} {improvement:<15}")

print()
print("Recommendation:")
print("  - Use smoothing for: GPS trajectories, low-rate sensors, jerky data")
print("  - Skip smoothing for: High-quality motion planning, already smooth data")
print("  - Window size: ~1-2 seconds worth of samples")
print()

# ============================================================================
# Example 3: Interaction Metrics (Waymo Sim Agents Style)
# ============================================================================
print("Example 3: Interaction Metrics for Multi-Agent Scenarios")
print("-" * 70)

# Ego vehicle trajectory
ego_traj = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
    [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]
])

# Other vehicles
other_vehicles = [
    # Vehicle 1: Merging from side
    np.array([
        [2, 5], [2, 4.5], [2, 4], [2.5, 3.5], [3, 3],
        [3.5, 2.5], [4, 2], [4.5, 1.5], [5, 1], [5.5, 0.5]
    ]),
    
    # Vehicle 2: Slower vehicle ahead
    np.array([
        [8, 0], [8.5, 0], [9, 0], [9.5, 0], [10, 0],
        [10.5, 0], [11, 0], [11.5, 0], [12, 0], [12.5, 0]
    ]),
    
    # Vehicle 3: Distant vehicle (not interacting)
    np.array([
        [0, 10], [1, 10], [2, 10], [3, 10], [4, 10],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ])
]

# Calculate interaction metrics
interaction = calculate_interaction_metrics(
    ego_traj,
    other_vehicles,
    vehicle_size=(4.5, 2.0)
)

print(f"Interaction Analysis:")
print(f"  Minimum distance to any object: {interaction['min_distance']:.2f}m")
print(f"  Mean distance to nearest:       {interaction['mean_distance_to_nearest']:.2f}m")
print(f"  Closest object ID:               {interaction['closest_object_id']} (Vehicle {interaction['closest_object_id'] + 1})")
print(f"  Closest approach at timestep:    {interaction['closest_approach_timestep']}")
print(f"  Close interactions (< 5m):       {interaction['num_close_interactions']}")
print()

print(f"Distance to nearest object per timestep:")
for t, dist in enumerate(interaction['distance_to_nearest_per_timestep']):
    marker = " ⚠️" if dist < 5.0 else ""
    print(f"  t={t}: {dist:.2f}m{marker}")

print()
print("Interpretation:")
print("  - Min distance < 2m:  Critical interaction (collision risk)")
print("  - Min distance < 5m:  Close interaction (attention needed)")
print("  - Min distance > 10m: Safe interaction")
print(f"  - Current status: {'⚠️ CLOSE INTERACTION' if interaction['min_distance'] < 5.0 else '✅ SAFE'}")
print()

# ============================================================================
# Example 4: Combined Multi-Horizon + Smoothing Analysis
# ============================================================================
print("Example 4: Combined Analysis - Multi-Horizon with Smoothing")
print("-" * 70)

# Generate longer noisy trajectory
t_long = np.linspace(0, 8, 80)
noisy_long_traj = np.column_stack([
    t_long * 5 + 0.3 * np.random.randn(80),
    0.5 * np.sin(t_long / 2) + 0.2 * np.random.randn(80)
])

# Calculate comfort at different horizons with and without smoothing
horizons = [10, 30, 50, 80]  # 1s, 3s, 5s, 8s

print("Comfort metrics at different horizons:")
print()
print(f"{'Horizon':<12} {'No Smooth':<20} {'With Smooth':<20}")
print("-" * 52)

for h in horizons:
    traj_subset = noisy_long_traj[:h]
    time_subset = t_long[:h]
    
    comfort_no_smooth = calculate_comfort_metrics(
        traj_subset, time_subset, use_smoothing=False, include_lateral=False
    )
    
    comfort_smooth = calculate_comfort_metrics(
        traj_subset, time_subset, use_smoothing=True, include_lateral=False
    )
    
    h_sec = h / 10
    no_smooth_rate = comfort_no_smooth['comfort_rate'] * 100
    smooth_rate = comfort_smooth['comfort_rate'] * 100
    
    print(f"{h_sec:.0f}s ({h:2d}):   {no_smooth_rate:.1f}% comfort     {smooth_rate:.1f}% comfort")

print()
print("Key Findings:")
print("  - Smoothing improves comfort scores by reducing noise artifacts")
print("  - Longer horizons may accumulate more noise → smoothing more beneficial")
print("  - Use smoothing for real-world sensor data with measurement noise")
print()

print("=" * 70)
print("Examples Complete!")
print("=" * 70)
print()
print("Summary of New Features:")
print("  ✅ Multi-horizon evaluation: Assess planning at 1s, 3s, 5s, 8s")
print("  ✅ Savitzky-Golay smoothing: Reduce noise in comfort metrics")
print("  ✅ Interaction metrics: Analyze multi-agent proximity")
print("  ✅ Industry alignment: Waymo, nuPlan, tuplan_garage standards")
