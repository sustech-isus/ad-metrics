"""
Example usage of simulation quality metrics for autonomous driving simulators.

This script demonstrates how to evaluate the fidelity and realism of simulated
sensor data (camera, LiDAR, radar) compared to real-world data.
"""

import numpy as np
from admetrics.simulation import (
    calculate_camera_image_quality,
    calculate_lidar_point_cloud_quality,
    calculate_radar_quality,
    calculate_sensor_noise_characteristics,
    calculate_multimodal_sensor_alignment,
    calculate_temporal_consistency,
    calculate_perception_sim2real_gap,
    calculate_weather_simulation_quality,
    calculate_vehicle_dynamics_quality,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def example_1_camera_quality():
    """Example 1: Camera Image Quality Assessment."""
    print_section("Example 1: Camera Image Quality")
    
    # Generate synthetic camera images (simulated vs real)
    np.random.seed(42)
    
    # Real-world images (with realistic noise and artifacts)
    real_images = np.random.rand(5, 224, 224, 3) * 255
    
    # Simulated images (similar but with different characteristics)
    # Simulation might have perfect lighting, less noise, different color distribution
    sim_images = real_images.copy()
    sim_images = sim_images * 1.1  # Slightly brighter
    sim_images = np.clip(sim_images + np.random.randn(*sim_images.shape) * 5, 0, 255)
    
    quality = calculate_camera_image_quality(
        sim_images, 
        real_images,
        metrics=['psnr', 'color_distribution', 'brightness', 'contrast']
    )
    
    print("Camera Quality Metrics:")
    print(f"  PSNR: {quality['psnr']:.2f} dB")
    print(f"  Color KL Divergence: {quality['color_kl_divergence']:.4f}")
    print(f"  Brightness Ratio: {quality['brightness_ratio']:.3f}")
    print(f"  Contrast Ratio: {quality['contrast_ratio']:.3f}")
    print("\nInterpretation:")
    print("  - PSNR > 30 dB: Good quality")
    print("  - KL divergence < 0.1: Similar color distribution")
    print("  - Ratios close to 1.0: Well-matched brightness/contrast")


def example_2_lidar_quality():
    """Example 2: LiDAR Point Cloud Quality."""
    print_section("Example 2: LiDAR Point Cloud Quality")
    
    np.random.seed(42)
    
    # Real LiDAR scan (1000 points with intensity)
    real_angles = np.random.uniform(0, 2*np.pi, 1000)
    real_ranges = np.random.uniform(5, 80, 1000)
    real_z = np.random.uniform(-2, 3, 1000)
    
    real_points = np.column_stack([
        real_ranges * np.cos(real_angles),
        real_ranges * np.sin(real_angles),
        real_z,
        np.random.uniform(0, 255, 1000)  # intensity
    ])
    
    # Simulated LiDAR (different point density, slightly different distribution)
    sim_angles = np.random.uniform(0, 2*np.pi, 1200)  # More points in sim
    sim_ranges = np.random.uniform(5, 80, 1200)
    sim_z = np.random.uniform(-2, 3, 1200)
    
    sim_points = np.column_stack([
        sim_ranges * np.cos(sim_angles),
        sim_ranges * np.sin(sim_angles),
        sim_z,
        np.random.uniform(50, 200, 1200)  # Different intensity distribution
    ])
    
    quality = calculate_lidar_point_cloud_quality(sim_points, real_points, max_range=100.0)
    
    print("LiDAR Quality Metrics:")
    print(f"  Chamfer Distance: {quality['chamfer_distance']:.3f} m")
    print(f"  Point Density Ratio: {quality['point_density_ratio']:.3f}")
    print(f"    Sim points: {quality['point_count_sim']}")
    print(f"    Real points: {quality['point_count_real']}")
    print(f"  Range Distribution KL: {quality['range_distribution_kl']:.4f}")
    print(f"  Vertical Angle KL: {quality['vertical_angle_kl']:.4f}")
    if 'intensity_correlation' in quality:
        print(f"  Intensity Correlation: {quality['intensity_correlation']:.3f}")
    
    print("\nInterpretation:")
    print("  - Chamfer < 0.5m: Good geometric match")
    print("  - Density ratio ~1.0: Matching point cloud density")
    print("  - Low KL divergence: Similar spatial distributions")


def example_3_radar_quality():
    """Example 3: Radar Detection Quality."""
    print_section("Example 3: Radar Detection Quality")
    
    np.random.seed(42)
    
    # Real radar detections [x, y, z, radial_velocity, rcs]
    real_radar = np.column_stack([
        np.random.uniform(-50, 50, 30),  # x
        np.random.uniform(-30, 30, 30),  # y
        np.random.uniform(-2, 2, 30),     # z
        np.random.uniform(-20, 20, 30),   # velocity
        np.random.uniform(-5, 30, 30)     # RCS in dBsm
    ])
    
    # Simulated radar (different detection characteristics)
    sim_radar = np.column_stack([
        np.random.uniform(-50, 50, 35),   # More detections
        np.random.uniform(-30, 30, 35),
        np.random.uniform(-2, 2, 35),
        np.random.uniform(-20, 20, 35),
        np.random.uniform(0, 35, 35)      # Different RCS distribution
    ])
    
    quality = calculate_radar_quality(sim_radar, real_radar)
    
    print("Radar Quality Metrics:")
    print(f"  Detection Density Ratio: {quality['detection_density_ratio']:.3f}")
    print(f"    Sim detections: {quality['detection_count_sim']}")
    print(f"    Real detections: {quality['detection_count_real']}")
    print(f"  Spatial Chamfer: {quality['spatial_chamfer']:.3f} m")
    if 'velocity_distribution_kl' in quality:
        print(f"  Velocity Distribution KL: {quality['velocity_distribution_kl']:.4f}")
    if 'rcs_distribution_kl' in quality:
        print(f"  RCS Distribution KL: {quality['rcs_distribution_kl']:.4f}")
    
    print("\nInterpretation:")
    print("  - Density ratio ~1.0: Similar detection rates")
    print("  - Low spatial error: Accurate positioning")
    print("  - Velocity/RCS KL: Realistic detection characteristics")


def example_4_sensor_noise():
    """Example 4: Sensor Noise Characteristics."""
    print_section("Example 4: Sensor Noise Characteristics")
    
    np.random.seed(42)
    
    # Ground truth position
    ground_truth = np.array([10.0, 5.0, 0.0])
    
    # Real sensor measurements (100 repeated measurements with realistic noise)
    real_noise_std = 0.15  # meters
    real_measurements = ground_truth + np.random.randn(100, 3) * real_noise_std
    
    # Simulated sensor measurements (different noise characteristics)
    sim_noise_std = 0.10  # Simulation often has less noise
    sim_measurements = ground_truth + np.random.randn(100, 3) * sim_noise_std
    
    quality = calculate_sensor_noise_characteristics(
        sim_measurements,
        real_measurements,
        ground_truth=ground_truth
    )
    
    print("Noise Characteristics:")
    print(f"  Noise Std (Sim): {quality['noise_std_sim']:.4f} m")
    print(f"  Noise Std (Real): {quality['noise_std_real']:.4f} m")
    print(f"  Std Ratio: {quality['noise_std_ratio']:.3f}")
    print(f"\n  Bias (Sim): {quality['bias_sim']:.4f} m")
    print(f"  Bias (Real): {quality['bias_real']:.4f} m")
    print(f"\n  SNR (Sim): {quality['snr_sim_db']:.2f} dB")
    print(f"  SNR (Real): {quality['snr_real_db']:.2f} dB")
    print(f"\n  KS Test Statistic: {quality['noise_distribution_ks_statistic']:.4f}")
    print(f"  KS Test p-value: {quality['noise_distribution_ks_pvalue']:.4f}")
    
    print("\nInterpretation:")
    print("  - Std ratio ~1.0: Matching noise levels")
    print("  - KS p-value > 0.05: Similar noise distributions")
    print("  - Low bias: Accurate measurements")
    print(f"  - Warning: Sim has {(1 - quality['noise_std_ratio']) * 100:.1f}% less noise!")


def example_5_multimodal_alignment():
    """Example 5: Multimodal Sensor Alignment."""
    print_section("Example 5: Multimodal Sensor Alignment")
    
    np.random.seed(42)
    
    # Generate ground truth detections
    gt_detections = np.array([
        [10, 5, 0, 4.5, 2.0, 1.8, 0.1],
        [20, -3, 0, 4.5, 2.0, 1.8, -0.2],
        [15, 10, 0, 5.0, 2.2, 2.0, 0.5],
    ])
    
    # Camera detections (slight position errors, size errors)
    camera_detections = gt_detections.copy()
    camera_detections[:, :3] += np.random.randn(3, 3) * 0.3  # Position noise
    camera_detections[:, 3:6] += np.random.randn(3, 3) * 0.2  # Size errors
    
    # LiDAR detections (different errors, might miss some detections)
    lidar_detections = gt_detections.copy()
    lidar_detections[:, :3] += np.random.randn(3, 3) * 0.2
    lidar_detections[:, 3:6] += np.random.randn(3, 3) * 0.15
    
    quality = calculate_multimodal_sensor_alignment(camera_detections, lidar_detections)
    
    print("Multimodal Alignment Metrics:")
    print(f"  Detection Agreement Rate: {quality['detection_agreement_rate']:.2%}")
    print(f"  Matched Detections: {quality['matched_detections']}")
    print(f"  Camera-only Detections: {quality['camera_only_detections']}")
    print(f"  LiDAR-only Detections: {quality['lidar_only_detections']}")
    print(f"\n  Spatial Alignment Error: {quality['spatial_alignment_error']:.3f} m")
    print(f"  Size Consistency Error: {quality['size_consistency_error']:.3f} m")
    print(f"    Length: {quality['length_error']:.3f} m")
    print(f"    Width: {quality['width_error']:.3f} m")
    print(f"    Height: {quality['height_error']:.3f} m")
    print(f"  Orientation Error: {quality['orientation_error_deg']:.2f}°")
    
    print("\nInterpretation:")
    print("  - Agreement > 90%: Good sensor calibration")
    print("  - Spatial error < 0.3m: Well-aligned sensors")
    print("  - Low size/orientation errors: Consistent detections")


def example_6_temporal_consistency():
    """Example 6: Temporal Consistency Evaluation."""
    print_section("Example 6: Temporal Consistency")
    
    np.random.seed(42)
    
    # Generate detection sequence (20 frames at 10 Hz)
    fps = 10.0
    num_frames = 20
    
    # Simulate objects moving smoothly
    detections_sequence = []
    for t in range(num_frames):
        # Base objects
        num_objects = 8 + np.random.randint(-2, 3)  # Slight variation
        
        detections = []
        for obj_id in range(num_objects):
            # Object moves smoothly
            x = obj_id * 5 + t * 0.5 + np.random.randn() * 0.1
            y = np.random.randn() * 2
            z = 0
            
            detections.append([x, y, z, 4.5, 2.0, 1.8, 0.0])
        
        detections_sequence.append(np.array(detections))
    
    quality = calculate_temporal_consistency(detections_sequence, fps=fps)
    
    print("Temporal Consistency Metrics:")
    print(f"  Detection Count (Mean ± Std): {quality['detection_count_mean']:.1f} ± {quality['detection_count_std']:.1f}")
    print(f"  Detection Count Variance: {quality['detection_count_variance']:.2f}")
    print(f"\n  Frame-to-Frame Consistency: {quality['frame_to_frame_consistency']:.2%}")
    print(f"  Consistency Std: {quality['consistency_std']:.3f}")
    print(f"\n  Flicker Rate: {quality['flicker_rate']:.2%}")
    if 'avg_new_detections_per_frame' in quality:
        print(f"  Avg New Detections/Frame: {quality['avg_new_detections_per_frame']:.2f}")
    if 'avg_lost_detections_per_frame' in quality:
        print(f"  Avg Lost Detections/Frame: {quality['avg_lost_detections_per_frame']:.2f}")
    
    print("\nInterpretation:")
    print("  - High frame consistency: Stable detections")
    print("  - Low flicker rate: No spurious appearances/disappearances")
    print("  - Low variance: Consistent scene representation")


def example_7_sim2real_gap():
    """Example 7: Perception Sim-to-Real Gap."""
    print_section("Example 7: Perception Sim-to-Real Performance Gap")
    
    np.random.seed(42)
    
    # Simulate detection results on 10 scenes
    sim_results = []
    real_results = []
    
    for scene_id in range(10):
        # Ground truth objects
        num_gt = 5 + np.random.randint(-2, 3)
        ground_truth = np.random.randn(num_gt, 7) * 10
        ground_truth[:, 2] = 0  # z = 0
        
        # Simulation: Higher recall, more false positives
        num_sim_det = num_gt + np.random.randint(0, 3)
        sim_predictions = ground_truth[:num_sim_det].copy()
        sim_predictions += np.random.randn(*sim_predictions.shape) * 0.3
        
        # Real world: Lower recall, fewer false positives, more noise
        num_real_det = max(1, num_gt - np.random.randint(0, 2))
        real_predictions = ground_truth[:num_real_det].copy()
        real_predictions += np.random.randn(*real_predictions.shape) * 0.8
        
        sim_results.append({
            'predictions': sim_predictions,
            'ground_truth': ground_truth
        })
        
        real_results.append({
            'predictions': real_predictions,
            'ground_truth': ground_truth
        })
    
    gap = calculate_perception_sim2real_gap(sim_results, real_results)
    
    print("Sim-to-Real Performance Gap:")
    print(f"\nSimulation Performance:")
    print(f"  Precision: {gap['precision_sim']:.2%}")
    print(f"  Recall: {gap['recall_sim']:.2%}")
    print(f"  F1-Score: {gap['f1_sim']:.2%}")
    
    print(f"\nReal-World Performance:")
    print(f"  Precision: {gap['precision_real']:.2%}")
    print(f"  Recall: {gap['recall_real']:.2%}")
    print(f"  F1-Score: {gap['f1_real']:.2%}")
    
    print(f"\nPerformance Gaps:")
    print(f"  Precision Gap: {gap['precision_gap']:+.2%}")
    print(f"  Recall Gap: {gap['recall_gap']:+.2%}")
    print(f"  F1 Gap: {gap['f1_gap']:+.2%}")
    print(f"  Performance Drop: {gap['performance_drop_pct']:.1f}%")
    
    print("\nInterpretation:")
    print("  - Positive gap: Sim performs better (common)")
    print("  - Performance drop: Expected degradation in real world")
    print("  - Large gaps (>20%): Significant sim-to-real domain shift")


def example_8_weather_quality():
    """Example 8: Weather and Environmental Quality Assessment."""
    print_section("Example 8: Weather and Environmental Quality")
    
    np.random.seed(50)
    
    # Example 8a: Rain simulation validation
    print("8a. Rain Simulation Validation")
    print("-" * 80)
    
    # Moderate rain scenario (gamma distribution is typical for rain)
    sim_rain = {
        'intensity': np.random.gamma(3, 4, 1000),  # ~12 mm/h average rain rate
        'visibility': np.random.normal(300, 60, 1000),  # Reduced visibility (m)
        'images': np.random.rand(10, 64, 64, 3) * 180  # Darker images due to rain
    }
    
    real_rain = {
        'intensity': np.random.gamma(3.1, 3.9, 1000),  # Similar rain distribution
        'visibility': np.random.normal(290, 65, 1000),
        'images': np.random.rand(10, 64, 64, 3) * 175
    }
    
    rain_quality = calculate_weather_simulation_quality(
        sim_rain, real_rain, 
        weather_type='rain',
        metrics=['intensity_distribution', 'visibility_range', 
                 'temporal_consistency', 'spatial_distribution']
    )
    
    print(f"  Rain Intensity KL Divergence: {rain_quality['intensity_kl_divergence']:.3f}")
    print(f"    Sim mean: {rain_quality['intensity_mean_sim']:.1f} mm/h")
    print(f"    Real mean: {rain_quality['intensity_mean_real']:.1f} mm/h")
    print(f"    Std ratio: {rain_quality['intensity_std_ratio']:.2f}")
    print(f"  Visibility Mean Error: {rain_quality['visibility_mean_error']:.1f} m")
    print(f"    KS p-value: {rain_quality['visibility_ks_pvalue']:.3f}")
    print(f"  Temporal Stability: {rain_quality['temporal_stability']:.2f}")
    print(f"  Spatial Correlation: {rain_quality['spatial_correlation']:.2f}")
    print(f"  Overall Realism Score: {rain_quality['overall_realism_score']:.1f}/100")
    
    if rain_quality['overall_realism_score'] > 75:
        print("  ✓ Rain simulation is production-ready")
    else:
        print(f"  ⚠ Rain simulation needs improvement")
    
    # Example 8b: Fog simulation validation
    print("\n8b. Fog Simulation Validation")
    print("-" * 80)
    
    # Dense fog scenario (exponential distribution typical for fog density)
    sim_fog = {
        'intensity': np.random.exponential(2.5, 1000),  # Fog density
        'visibility': np.random.normal(180, 35, 1000),  # Low visibility
    }
    
    real_fog = {
        'intensity': np.random.exponential(2.6, 1000),
        'visibility': np.random.normal(175, 38, 1000),
    }
    
    fog_quality = calculate_weather_simulation_quality(
        sim_fog, real_fog,
        weather_type='fog',
        metrics=['intensity_distribution', 'visibility_range']
    )
    
    print(f"  Fog Density KL Divergence: {fog_quality['intensity_kl_divergence']:.3f}")
    print(f"    Sim mean: {fog_quality['intensity_mean_sim']:.2f}")
    print(f"    Real mean: {fog_quality['intensity_mean_real']:.2f}")
    print(f"  Visibility Mean Error: {fog_quality['visibility_mean_error']:.1f} m")
    print(f"    Sim median: {fog_quality['visibility_median_sim']:.1f} m")
    print(f"    Real median: {fog_quality['visibility_median_real']:.1f} m")
    
    # Fog visibility is safety-critical
    if fog_quality['visibility_mean_error'] < 50:
        print("  ✓ Fog visibility simulation is accurate")
    else:
        print("  ⚠ Fog visibility needs calibration")
    
    # Example 8c: Lighting condition validation
    print("\n8c. Lighting Condition Validation (Dusk/Dawn)")
    print("-" * 80)
    
    # Dusk lighting: gradual transition, beta distribution
    sim_dusk = {
        'lighting': np.random.beta(3, 3, 1000),  # Mid-range lighting (0-1)
    }
    
    real_dusk = {
        'lighting': np.random.beta(3.2, 2.9, 1000),
    }
    
    lighting_quality = calculate_weather_simulation_quality(
        sim_dusk, real_dusk,
        weather_type='lighting',
        metrics=['lighting_histogram']
    )
    
    print(f"  Lighting KL Divergence: {lighting_quality['lighting_kl_divergence']:.3f}")
    print(f"    Sim mean: {lighting_quality['lighting_mean_sim']:.2f}")
    print(f"    Real mean: {lighting_quality['lighting_mean_real']:.2f}")
    
    if lighting_quality['lighting_kl_divergence'] < 0.3:
        print("  ✓ Lighting distribution is realistic")
    else:
        print("  ⚠ Lighting distribution needs adjustment")
    
    # Example 8d: Shadow realism validation
    print("\n8d. Shadow Realism Validation")
    print("-" * 80)
    
    # Create images with shadow patterns
    sim_shadow_imgs = np.ones((10, 64, 64, 3)) * 150
    sim_shadow_imgs[:, :, :30, :] = 70  # Shadow on left side
    
    real_shadow_imgs = np.ones((10, 64, 64, 3)) * 145
    real_shadow_imgs[:, :, :30, :] = 75  # Similar shadow
    
    sim_shadows = {
        'shadow_coverage': np.random.beta(3, 7, 500),  # ~30% shadow coverage
        'images': sim_shadow_imgs
    }
    
    real_shadows = {
        'shadow_coverage': np.random.beta(3.1, 6.9, 500),
        'images': real_shadow_imgs
    }
    
    shadow_quality = calculate_weather_simulation_quality(
        sim_shadows, real_shadows,
        weather_type='shadows',
        metrics=['shadow_realism']
    )
    
    print(f"  Shadow Coverage Error: {shadow_quality['shadow_coverage_error']:.3f}")
    print(f"    Sim coverage: {shadow_quality['shadow_coverage_sim']:.2f}")
    print(f"    Real coverage: {shadow_quality['shadow_coverage_real']:.2f}")
    print(f"  Shadow Edge Sharpness Ratio: {shadow_quality['shadow_edge_sharpness']:.2f}")
    
    if shadow_quality['shadow_coverage_error'] < 0.1:
        print("  ✓ Shadow coverage is accurate")
    else:
        print("  ⚠ Shadow coverage needs calibration")
    
    # Example 8e: Comprehensive weather quality assessment
    print("\n8e. Comprehensive Weather Assessment")
    print("-" * 80)
    
    weather_conditions = {
        'light_rain': {'sim': sim_rain, 'real': real_rain, 'type': 'rain'},
        'dense_fog': {'sim': sim_fog, 'real': real_fog, 'type': 'fog'},
    }
    
    print("Weather Simulation Quality Summary:")
    for condition_name, data in weather_conditions.items():
        quality = calculate_weather_simulation_quality(
            data['sim'], data['real'], weather_type=data['type']
        )
        score = quality.get('overall_realism_score', 0)
        status = "✓" if score > 75 else "⚠"
        print(f"  {status} {condition_name:15s}: {score:5.1f}/100")
    
    print("\nWeather Validation Best Practices:")
    print("  1. Test multiple intensity levels (light/moderate/heavy)")
    print("  2. Validate temporal transitions (onset, steady-state, clearing)")
    print("  3. Check sensor-specific effects (camera droplets, LiDAR absorption)")
    print("  4. Verify geographic variations (tropical vs. temperate rain)")
    print("  5. Ensure overall realism score > 75 for production use")
    print("=" * 80 + "\n")


def example_9_vehicle_dynamics_quality():
    """Example 9: Vehicle Dynamics and Physics Validation."""
    print_section("Example 9: Vehicle Dynamics Quality")
    
    np.random.seed(42)
    
    # Scenario 1: Highway merging (acceleration + lateral movement)
    print("Scenario 1: Highway Merging")
    print("-" * 40)
    N, T = 20, 100  # 20 vehicles, 100 timesteps
    dt = 0.1  # 10 Hz
    t = np.linspace(0, (T-1)*dt, T)
    
    # Create realistic highway merge trajectories
    # Real: Smooth acceleration from 20 m/s to 30 m/s with lane change
    real_merge_traj = np.zeros((N, T, 6))
    for i in range(N):
        # Longitudinal: smooth acceleration
        v0 = 20.0 + np.random.randn() * 2.0
        a = 1.5 + np.random.randn() * 0.3  # ~1.5 m/s² acceleration
        vx = np.clip(v0 + a * t, v0, 30.0)
        x = np.cumsum(vx) * dt
        
        # Lateral: smooth lane change around t=3-5 seconds
        merge_start = 30 + np.random.randint(-5, 5)
        merge_duration = 30
        y = 3.5 * (1.0 / (1.0 + np.exp(-(np.arange(T) - merge_start) / 5)))
        vy = np.gradient(y, dt)
        
        real_merge_traj[i, :, 0] = x
        real_merge_traj[i, :, 1] = y
        real_merge_traj[i, :, 2] = vx
        real_merge_traj[i, :, 3] = vy
    
    # Sim: Similar but with slightly different dynamics
    sim_merge_traj = real_merge_traj.copy()
    # Simulation has slightly higher acceleration
    sim_merge_traj[:, :, 2] *= 1.15
    sim_merge_traj[:, :, 0] = np.cumsum(sim_merge_traj[:, :, 2], axis=1) * dt
    # Add some noise
    sim_merge_traj[:, :, :4] += np.random.randn(N, T, 4) * 0.1
    
    merge_quality = calculate_vehicle_dynamics_quality(
        sim_merge_traj, real_merge_traj,
        maneuver_type='acceleration',
        metrics=['acceleration_profile', 'lateral_dynamics', 'trajectory_smoothness']
    )
    
    print(f"Overall Dynamics Score: {merge_quality.get('overall_score', 0):.1f}/100")
    if 'acceleration_kl_divergence' in merge_quality:
        kl = merge_quality['acceleration_kl_divergence']
        status = "✓" if kl < 0.5 else "⚠"
        print(f"  {status} Acceleration Distribution Match: KL={kl:.3f}")
    
    if 'lateral_accel_max_sim' in merge_quality:
        lat_sim = merge_quality['lateral_accel_max_sim']
        lat_real = merge_quality.get('lateral_accel_max_real', 0)
        print(f"  • Lateral Accel: Sim={lat_sim:.2f}, Real={lat_real:.2f} m/s²")
    
    if 'longitudinal_jerk_mean_sim' in merge_quality:
        jerk_sim = merge_quality['longitudinal_jerk_mean_sim']
        jerk_real = merge_quality.get('longitudinal_jerk_mean_real', 0)
        status = "✓" if jerk_sim < 3.0 else "⚠"
        print(f"  {status} Longitudinal Jerk: Sim={jerk_sim:.2f}, Real={jerk_real:.2f} m/s³")
    
    # Scenario 2: Emergency braking
    print("\nScenario 2: Emergency Braking")
    print("-" * 40)
    N, T = 15, 60
    
    # Real: Vehicle braking from 25 m/s to 0 with realistic deceleration
    real_brake_traj = np.zeros((N, T, 6))
    for i in range(N):
        v0 = 25.0 + np.random.randn() * 2.0
        # Emergency braking ~7 m/s²
        decel = 7.0 + np.random.randn() * 0.5
        
        vx = np.zeros(T)
        x = np.zeros(T)
        for t_idx in range(T):
            if t_idx < 10:  # Cruising
                vx[t_idx] = v0
            else:  # Braking
                vx[t_idx] = max(0, v0 - decel * (t_idx - 10) * dt)
            
            if t_idx > 0:
                x[t_idx] = x[t_idx-1] + vx[t_idx] * dt
        
        real_brake_traj[i, :, 0] = x
        real_brake_traj[i, :, 2] = vx
    
    # Sim: Slightly different braking characteristics
    sim_brake_traj = real_brake_traj.copy()
    # Simulation has slightly longer braking distance (lower deceleration)
    for i in range(N):
        v0 = sim_brake_traj[i, 0, 2]
        decel = 6.5  # Slightly lower
        vx = np.zeros(T)
        x = np.zeros(T)
        for t_idx in range(T):
            if t_idx < 10:
                vx[t_idx] = v0
            else:
                vx[t_idx] = max(0, v0 - decel * (t_idx - 10) * dt)
            if t_idx > 0:
                x[t_idx] = x[t_idx-1] + vx[t_idx] * dt
        
        sim_brake_traj[i, :, 0] = x
        sim_brake_traj[i, :, 2] = vx
    
    brake_quality = calculate_vehicle_dynamics_quality(
        sim_brake_traj, real_brake_traj,
        maneuver_type='braking',
        metrics=['braking_distance']
    )
    
    print(f"Overall Dynamics Score: {brake_quality.get('overall_score', 0):.1f}/100")
    if 'braking_distance_error' in brake_quality:
        error = brake_quality['braking_distance_error']
        status = "✓" if abs(error) < 15 else "⚠"
        print(f"  {status} Braking Distance Error: {error:+.1f}%")
    
    if 'deceleration_mean' in brake_quality:
        decel = brake_quality['deceleration_mean']
        print(f"  • Mean Deceleration: {decel:.2f} m/s²")
    
    # Scenario 3: Urban lane changes
    print("\nScenario 3: Urban Lane Changes")
    print("-" * 40)
    N, T = 10, 80
    
    # Real: Lane change at moderate speed
    real_lane_traj = np.zeros((N, T, 6))
    for i in range(N):
        vx = 15.0 + np.random.randn() * 1.0  # ~15 m/s constant speed
        x = np.arange(T) * vx * dt
        
        # Lane change at t=2s for duration of 3s
        lane_start = 20 + np.random.randint(-3, 3)
        y = 3.5 * (1.0 / (1.0 + np.exp(-(np.arange(T) - lane_start) / 4)))
        vy = np.gradient(y, dt)
        
        real_lane_traj[i, :, 0] = x
        real_lane_traj[i, :, 1] = y
        real_lane_traj[i, :, 2] = vx
        real_lane_traj[i, :, 3] = vy
    
    # Sim: Similar but with sharper lane change
    sim_lane_traj = real_lane_traj.copy()
    for i in range(N):
        lane_start = 20
        # Sharper transition
        y = 3.5 * (1.0 / (1.0 + np.exp(-(np.arange(T) - lane_start) / 3)))
        vy = np.gradient(y, dt)
        sim_lane_traj[i, :, 1] = y
        sim_lane_traj[i, :, 3] = vy
    
    lane_quality = calculate_vehicle_dynamics_quality(
        sim_lane_traj, real_lane_traj,
        maneuver_type='lane_change',
        metrics=['lateral_dynamics', 'trajectory_smoothness']
    )
    
    print(f"Overall Dynamics Score: {lane_quality.get('overall_score', 0):.1f}/100")
    if 'lateral_jerk_ratio' in lane_quality:
        ratio = lane_quality['lateral_jerk_ratio']
        status = "✓" if 0.8 <= ratio <= 1.2 else "⚠"
        print(f"  {status} Lateral Jerk Ratio (sim/real): {ratio:.2f}")
    
    if 'lateral_accel_max_sim' in lane_quality:
        lat_sim = lane_quality['lateral_accel_max_sim']
        lat_real = lane_quality.get('lateral_accel_max_real', 0)
        print(f"  • Lateral Accel: Sim={lat_sim:.2f}, Real={lat_real:.2f} m/s²")
    
    # Comprehensive validation summary
    print("\nVehicle Dynamics Quality Summary:")
    print("-" * 40)
    scenarios = {
        'Highway Merge': merge_quality,
        'Emergency Brake': brake_quality,
        'Urban Lane Change': lane_quality,
    }
    
    for scenario_name, quality in scenarios.items():
        score = quality.get('overall_score', 0)
        status = "✓" if score > 70 else "⚠"
        print(f"  {status} {scenario_name:20s}: {score:5.1f}/100")
    
    print("\nVehicle Dynamics Best Practices:")
    print("  1. Validate acceleration: KL divergence < 0.5 for good match")
    print("  2. Check braking: Distance error < 15% for safety-critical")
    print("  3. Lateral dynamics: Typical lane change 2-4 m/s² lateral accel")
    print("  4. Smoothness: Jerk < 3 m/s³ for comfortable motion")
    print("  5. Match scenario types: Highway vs urban have different profiles")
    print("  6. Physics limits: Accel < 4 m/s² normal, < 10 m/s² emergency")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Simulation Quality Metrics - Usage Examples")
    print("=" * 80)
    
    example_1_camera_quality()
    example_2_lidar_quality()
    example_3_radar_quality()
    example_4_sensor_noise()
    example_5_multimodal_alignment()
    example_6_temporal_consistency()
    example_7_sim2real_gap()
    example_8_weather_quality()
    example_9_vehicle_dynamics_quality()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("\n")
    print("Key Takeaways for Simulation Quality:")
    print("  1. Camera: PSNR, color distribution, brightness/contrast matching")
    print("  2. LiDAR: Chamfer distance, point density, spatial distributions")
    print("  3. Radar: Detection density, velocity/RCS characteristics")
    print("  4. Noise: Std ratio, KS test, SNR comparison")
    print("  5. Calibration: Multimodal alignment, spatial consistency")
    print("  6. Temporal: Frame-to-frame consistency, flicker rate")
    print("  7. Sim2Real: Precision/recall gaps, performance degradation")
    print("  8. Weather: Rain/fog/lighting realism, environmental fidelity")
    print("  9. Dynamics: Acceleration, braking, lateral motion, smoothness")
    print("=" * 80 + "\n")
