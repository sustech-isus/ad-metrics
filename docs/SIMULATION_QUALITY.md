# Simulation Quality Metrics

This document describes metrics for evaluating the quality and realism of simulated sensor data in autonomous vehicle simulators (CARLA, LGSVL, AirSim, Waymo Sim, nuPlan, etc.).

## Table of Contents

1. [Overview](#overview)
2. [Camera Image Quality](#camera-image-quality)
3. [LiDAR Point Cloud Quality](#lidar-point-cloud-quality)
4. [Radar Quality](#radar-quality)
5. [Sensor Noise Characteristics](#sensor-noise-characteristics)
6. [Multimodal Sensor Alignment](#multimodal-sensor-alignment)
7. [Temporal Consistency](#temporal-consistency)
8. [Sim-to-Real Gap](#sim-to-real-gap)
9. [Weather and Environmental Quality](#weather-and-environmental-quality)
10. [Vehicle Dynamics Quality](#vehicle-dynamics-quality)
11. [Best Practices](#best-practices)
12. [References](#references)
13. [See Also](#see-also)

## Overview

Simulation quality is critical for successful sim-to-real transfer in autonomous driving. Poor simulation fidelity leads to:
- **Performance degradation** when deploying to real vehicles
- **False confidence** in perception/planning algorithms
- **Wasted development time** on unrealistic scenarios

### Key Evaluation Dimensions

| Dimension | What It Measures | Why It Matters |
|-----------|------------------|----------------|
| **Visual Realism** | Camera image fidelity | Affects vision-based perception |
| **Geometric Accuracy** | LiDAR/radar spatial quality | Critical for localization, mapping |
| **Noise Characteristics** | Sensor noise patterns | Impacts robustness, filtering |
| **Calibration** | Multi-sensor alignment | Essential for sensor fusion |
| **Temporal Coherence** | Frame-to-frame consistency | Affects tracking, prediction |
| **Performance Gap** | Sim vs real detection accuracy | Quantifies domain shift |

## Camera Image Quality

### Purpose
Evaluate visual realism of simulated camera images compared to real-world data.

### Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
**Formula:**
```
PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
```

**Interpretation:**
- **> 40 dB**: Excellent quality (near-identical)
- **30-40 dB**: Good quality
- **20-30 dB**: Acceptable
- **< 20 dB**: Poor quality

#### Color Distribution (KL Divergence)
**Measures:** Similarity of RGB histograms between sim and real

**Interpretation:**
- **< 0.1**: Very similar color distribution
- **0.1-0.5**: Moderate similarity
- **> 0.5**: Significant color shift (lighting/material differences)

#### Brightness/Contrast Ratios
**Measures:** Ratio of mean brightness and standard deviation

**Interpretation:**
- **~1.0**: Well-matched
- **> 1.2**: Sim is brighter/higher contrast
- **< 0.8**: Sim is darker/lower contrast

### Usage

```python
from admetrics.simulation import camera_image_quality

quality = calculate_camera_image_quality(
    sim_images,      # (N, H, W, 3)
    real_images,     # (N, H, W, 3)
    metrics=['psnr', 'ssim', 'color_distribution', 'brightness', 'contrast']
)

print(f"PSNR: {quality['psnr']:.2f} dB")
print(f"Color KL: {quality['color_kl_divergence']:.4f}")
```

### Common Issues
- **Too bright/clean**: Simulations often lack weather effects, dirt, glare
- **Perfect colors**: Real cameras have color response curves
- **No motion blur**: Cameras have exposure time
- **Incorrect dynamic range**: HDR vs LDR differences

## LiDAR Point Cloud Quality

### Purpose
Evaluate geometric fidelity of simulated LiDAR scans.

### Metrics

#### Chamfer Distance
**Measures:** Average nearest-neighbor distance between point clouds (bidirectional)

**Formula:**
```
Chamfer = (1/|Sim|) Σ min_dist(p_sim, Real) + (1/|Real|) Σ min_dist(p_real, Sim)
```

**Interpretation:**
- **< 0.2m**: Excellent match
- **0.2-0.5m**: Good match
- **> 0.5m**: Significant geometric differences

#### Point Density Ratio
**Measures:** Ratio of point counts

**Interpretation:**
- **~1.0**: Matching density
- **> 1.5**: Sim has too many points (unrealistic)
- **< 0.7**: Sim is too sparse

#### Range Distribution
**Measures:** KL divergence of distance-from-sensor distributions

**Why:** Simulators might not model occlusion, max range correctly

#### Vertical Angle Distribution
**Measures:** Distribution of elevation angles

**Why:** Checks beam pattern matching (16/32/64/128 channel LiDARs)

### Usage

```python
from admetrics.simulation import lidar_point_cloud_quality

quality = calculate_lidar_point_cloud_quality(
    sim_points,    # (N, 3) or (N, 4) with intensity
    real_points,   # (M, 3) or (M, 4)
    max_range=100.0
)

print(f"Chamfer Distance: {quality['chamfer_distance']:.3f}m")
print(f"Point Density Ratio: {quality['point_density_ratio']:.3f}")
```

### Common Issues
- **Perfect reflections**: Real LiDAR has material-dependent returns
- **No dropouts**: Glass, water absorb/scatter beams
- **Ideal beam patterns**: Real LiDARs have beam divergence
- **Missing multi-path**: Reflections, scattering

## Radar Quality

### Purpose
Evaluate radar simulation fidelity (less common but important for robust AV).

### Metrics

#### Detection Density
**Measures:** Ratio of detection counts

#### Spatial Accuracy (Chamfer)
**Measures:** Position accuracy of radar detections

#### Velocity Distribution
**Measures:** Radial velocity histogram matching

**Why:** Radar measures Doppler shift - critical for velocity estimation

#### RCS Distribution (Radar Cross-Section)
**Measures:** Signal strength distribution

**Why:** RCS varies by material, angle - affects detection range

### Usage

```python
from admetrics.simulation import radar_quality

quality = calculate_radar_quality(
    sim_detections,  # (N, 5) [x, y, z, velocity, rcs]
    real_detections  # (M, 5)
)

print(f"Detection Density: {quality['detection_density_ratio']:.3f}")
print(f"RCS Distribution KL: {quality['rcs_distribution_kl']:.4f}")
```

### Common Issues
- **Too many detections**: Radar has clutter, false alarms
- **Unrealistic RCS**: Material properties affect returns
- **Missing multi-path**: Ground reflections common
- **Ideal velocity**: Real radar has velocity resolution limits

## Sensor Noise Characteristics

### Purpose
Validate that simulated sensor noise matches real-world noise patterns.

### Metrics

#### Noise Std Ratio
**Measures:** Ratio of noise standard deviations

**Critical:** Most simulators underestimate noise

#### KS Test (Kolmogorov-Smirnov)
**Measures:** Distribution similarity test

**Interpretation:**
- **p-value > 0.05**: Distributions match (cannot reject null hypothesis)
- **p-value < 0.05**: Distributions differ significantly

#### Signal-to-Noise Ratio (SNR)
**Measures:** SNR in dB

**Formula:**
```
SNR = 10 * log10(signal_power / noise_power)
```

### Usage

```python
from admetrics.simulation import sensor_noise_characteristics

# Repeated measurements of same target
quality = calculate_sensor_noise_characteristics(
    sim_measurements,   # (N, D)
    real_measurements,  # (M, D)
    ground_truth=gt     # (D,)
)

print(f"Noise Std Ratio: {quality['noise_std_ratio']:.3f}")
print(f"KS p-value: {quality['noise_distribution_ks_pvalue']:.4f}")
```

### Common Issues
- **Too little noise**: Simulations are often optimistic
- **Wrong distribution**: Real sensors have non-Gaussian tails
- **No systematic errors**: Bias, drift, temperature effects
- **Missing correlations**: Spatial/temporal noise correlations

## Multimodal Sensor Alignment

### Purpose
Evaluate calibration quality between different sensor modalities.

### Metrics

#### Detection Agreement Rate
**Measures:** Fraction of detections visible in both sensors

**Why:** Mis-calibration leads to disagreement

#### Spatial Alignment Error
**Measures:** Mean position difference for matched detections

**Interpretation:**
- **< 0.2m**: Excellent calibration
- **0.2-0.5m**: Acceptable
- **> 0.5m**: Poor calibration (fusion will fail)

#### Size/Orientation Consistency
**Measures:** Bounding box parameter consistency

### Usage

```python
from admetrics.simulation import multimodal_sensor_alignment

quality = calculate_multimodal_sensor_alignment(
    camera_detections,  # (N, 7) [x, y, z, l, w, h, yaw]
    lidar_detections    # (M, 7)
)

print(f"Agreement Rate: {quality['detection_agreement_rate']:.2%}")
print(f"Spatial Alignment: {quality['spatial_alignment_error']:.3f}m")
```

### Common Issues
- **Perfect alignment**: Real systems have calibration errors
- **No temporal offset**: Camera/LiDAR not synchronized
- **Missing FOV differences**: Different sensor coverage

## Temporal Consistency

### Purpose
Evaluate frame-to-frame coherence of sensor data.

### Metrics

#### Detection Count Variance
**Measures:** Stability of detection counts over time

**Why:** Flickering detections indicate poor tracking

#### Frame-to-Frame Consistency
**Measures:** Fraction of detections that persist between frames

**Interpretation:**
- **> 80%**: Good temporal coherence
- **50-80%**: Moderate flickering
- **< 50%**: Severe instability

#### Flicker Rate
**Measures:** Rate of appearance/disappearance events

### Usage

```python
from admetrics.simulation import temporal_consistency

quality = calculate_temporal_consistency(
    detections_sequence,  # List of (N_t, D) arrays
    fps=10.0
)

print(f"Frame Consistency: {quality['frame_to_frame_consistency']:.2%}")
print(f"Flicker Rate: {quality['flicker_rate']:.2%}")
```

### Common Issues
- **Unstable detections**: Real trackers smooth detections
- **No prediction**: Real systems use Kalman filters
- **Missing occlusion handling**: Objects appear/disappear realistically

## Sim-to-Real Gap

### Purpose
Quantify performance degradation from simulation to real-world deployment.

### Metrics

#### Precision/Recall Gap
**Measures:** Difference in detection metrics

**Formula:**
```
Gap = Performance_sim - Performance_real
```

#### Performance Drop
**Measures:** Percentage degradation

**Interpretation:**
- **< 10%**: Excellent sim quality
- **10-20%**: Acceptable gap
- **20-30%**: Significant domain shift
- **> 30%**: Poor simulation fidelity

### Usage

```python
from admetrics.simulation import perception_sim2real_gap

gap = calculate_perception_sim2real_gap(
    sim_results,   # List of dicts with predictions/ground_truth
    real_results   # Same format
)

print(f"Performance Drop: {gap['performance_drop_pct']:.1f}%")
print(f"Precision Gap: {gap['precision_gap']:+.2%}")
```

### Mitigation Strategies

**Domain Randomization:**
- Randomize lighting, textures, weather
- Vary sensor parameters (noise, blur, distortion)
- Add artifacts (lens flare, motion blur, compression)

**Domain Adaptation:**
- CycleGAN for image translation
- Feature alignment techniques
- Self-supervised learning on real data

**Hybrid Approaches:**
- Train on sim, fine-tune on real
- Mix sim and real data
- Use sim for rare scenarios, real for common

## Weather and Environmental Quality

### Purpose
Validate the realism of weather effects (rain, fog, snow) and environmental conditions (lighting, shadows) in simulation. Critical for testing perception robustness across diverse operating conditions.

### Key Importance
- **Safety-Critical**: Weather affects 30-40% of real-world driving scenarios
- **Perception Robustness**: Ensures algorithms work in adverse conditions
- **Scenario Coverage**: Validates rare but important edge cases
- **Regulatory Requirements**: Many regions require all-weather testing

### Usage

```python
from admetrics.simulation import calculate_weather_simulation_quality
import numpy as np

# Rain simulation validation
sim_data = {
    'intensity': np.random.gamma(2, 5, 1000),  # Rain rate in mm/h
    'visibility': np.random.normal(200, 50, 1000),  # Visibility in meters
    'images': sim_camera_images  # (N, H, W, C) numpy array
}

real_data = {
    'intensity': np.random.gamma(2.1, 4.9, 1000),
    'visibility': np.random.normal(195, 52, 1000),
    'images': real_camera_images
}

# Evaluate rain simulation quality
quality = calculate_weather_simulation_quality(
    sim_data, real_data, 
    weather_type='rain',
    metrics=['intensity_distribution', 'visibility_range', 
             'temporal_consistency', 'spatial_distribution']
)

print(f"Rain intensity realism (KL divergence): {quality['intensity_kl_divergence']:.3f}")
print(f"Visibility accuracy (m): {quality['visibility_mean_error']:.1f}")
print(f"Temporal stability: {quality['temporal_stability']:.2f}")
print(f"Overall realism score: {quality['overall_realism_score']:.1f}/100")
```

### Metrics

#### Intensity Distribution
**Purpose**: Compare weather intensity distributions (rain rate, fog density, snow accumulation)

**Metric**: Kullback-Leibler (KL) divergence between sim and real intensity histograms

**Interpretation:**
- `< 0.1`: Excellent match
- `0.1 - 0.5`: Good match
- `0.5 - 1.0`: Acceptable
- `> 1.0`: Poor match, needs calibration

**Weather Type Thresholds:**
| Weather | Intensity Range | Unit | Typical Distribution |
|---------|----------------|------|---------------------|
| Light rain | 0-2.5 | mm/h | Gamma(2, 3) |
| Moderate rain | 2.5-10 | mm/h | Gamma(3, 4) |
| Heavy rain | 10-50 | mm/h | Gamma(5, 6) |
| Light fog | 500-1000 | m visibility | Normal(750, 100) |
| Moderate fog | 200-500 | m visibility | Normal(350, 80) |
| Dense fog | < 200 | m visibility | Exponential(150) |
| Light snow | 0-1 | cm/h | Gamma(1.5, 2) |
| Heavy snow | > 5 | cm/h | Gamma(6, 3) |

#### Visibility Range
**Purpose**: Validate visibility distance under weather conditions

**Metrics:**
- `visibility_mean_error`: Mean absolute error in visibility (meters)
- `visibility_ks_pvalue`: KS test p-value (> 0.05 indicates similar distributions)

**Interpretation:**
- Mean error < 50m: Excellent
- Mean error 50-100m: Good
- Mean error 100-200m: Acceptable
- Mean error > 200m: Poor

**Safety Impact:**
- Fog < 200m visibility: Requires special sensors (thermal, radar)
- Rain reducing visibility > 30%: Affects camera-based perception

#### Temporal Consistency
**Purpose**: Ensure weather effects change smoothly frame-to-frame

**Metrics:**
- `temporal_stability`: Measure of frame-to-frame stability (0-1)
- `frame_to_frame_correlation`: Auto-correlation at lag 1

**Interpretation:**
- Stability > 0.9: Excellent (smooth transitions)
- Stability 0.7-0.9: Good  
- Stability 0.5-0.7: Acceptable
- Stability < 0.5: Poor (jittery/unrealistic)

**Why It Matters:**
- Sudden weather changes confuse tracking algorithms
- Unrealistic flickering affects vision models
- Real weather transitions are gradual (minutes to hours)

#### Spatial Distribution
**Purpose**: Validate spatial patterns of weather effects

**Metric**: `spatial_correlation` - Correlation coefficient of spatial gradients

**Interpretation:**
- Correlation > 0.8: Excellent spatial realism
- Correlation 0.6-0.8: Good
- Correlation 0.4-0.6: Acceptable
- Correlation < 0.4: Poor

**Spatial Patterns:**
- Rain: Uniform locally, gradual transitions
- Fog: Patches and layers, non-uniform density
- Snow: Accumulation patterns, wind effects

#### Particle Density (Rain/Snow)
**Purpose**: Estimate density of rain drops or snow flakes

**Metric**: `particle_density_ratio` - Ratio of sim to real particle variance

**Interpretation:**
- Ratio 0.8-1.2: Excellent match
- Ratio 0.6-1.4: Good
- Ratio 0.4-1.6: Acceptable
- Ratio outside range: Poor

**Typical Densities:**
- Light rain: ~100 drops/m³
- Heavy rain: ~500-1000 drops/m³
- Light snow: ~50 flakes/m³
- Heavy snow: ~200-300 flakes/m³

#### Lighting Conditions
**Purpose**: Validate day/night/dusk lighting distributions

**Metrics:**
- `lighting_kl_divergence`: KL divergence of lighting histograms
- `lighting_mean_sim/real`: Mean lighting intensity (0-1 normalized)

**Lighting Categories:**
| Condition | Mean Intensity | Time of Day | Challenges |
|-----------|---------------|-------------|-----------|
| Bright daylight | 0.8-1.0 | 10am-3pm | Glare, shadows |
| Overcast day | 0.6-0.8 | Any daytime | Low contrast |
| Dusk/Dawn | 0.3-0.6 | 6-8am, 5-7pm | Mixed lighting |
| Night | 0.0-0.3 | 9pm-6am | Low light, headlights |

**Interpretation:**
- KL divergence < 0.3: Excellent lighting realism
- KL divergence 0.3-0.6: Good
- KL divergence > 0.6: Poor

#### Shadow Realism
**Purpose**: Validate shadow coverage and edge sharpness

**Metrics:**
- `shadow_coverage_error`: Absolute error in shadow coverage (0-1)
- `shadow_edge_sharpness`: Ratio of edge gradient strengths

**Interpretation:**
- Coverage error < 0.1: Excellent (within 10%)
- Coverage error 0.1-0.2: Good
- Coverage error > 0.2: Poor
- Edge sharpness ratio 0.8-1.2: Realistic shadows

**Shadow Characteristics:**
- Sunny day: 20-40% coverage, sharp edges
- Overcast: 5-10% coverage, soft edges  
- Urban canyon: 40-60% coverage, complex patterns
- Open highway: 10-20% coverage, moving shadows

#### Overall Realism Score
**Purpose**: Combined quality score (0-100) across all weather metrics

**Computation**: Weighted average of normalized sub-metrics
- Intensity distribution: 30%
- Visibility range: 25%
- Temporal stability: 25%
- Spatial correlation: 20%

**Interpretation:**
- Score > 80: Production-ready weather simulation
- Score 70-80: Good quality, minor calibration needed
- Score 60-70: Acceptable for development/testing
- Score < 60: Significant improvements needed

### Weather Type Examples

#### Rain Validation
```python
# Moderate rain scenario
sim_rain = {
    'intensity': np.random.gamma(3, 4, 1000),  # ~12 mm/h average
    'visibility': np.random.normal(300, 60, 1000),  # Reduced visibility
    'images': rain_sim_images
}

real_rain = {
    'intensity': np.random.gamma(3.1, 3.9, 1000),
    'visibility': np.random.normal(290, 65, 1000),
    'images': rain_real_images
}

quality = calculate_weather_simulation_quality(
    sim_rain, real_rain, weather_type='rain'
)

# Check if rain simulation is realistic enough
if quality['overall_realism_score'] > 75:
    print("✓ Rain simulation is production-ready")
else:
    print(f"⚠ Rain simulation needs improvement (score: {quality['overall_realism_score']:.1f})")
```

#### Fog Validation
```python
# Dense fog scenario
sim_fog = {
    'intensity': np.random.exponential(2.5, 1000),  # Fog density
    'visibility': np.random.normal(150, 30, 1000),  # Low visibility
    'images': fog_sim_images
}

real_fog = {
    'intensity': np.random.exponential(2.6, 1000),
    'visibility': np.random.normal(145, 35, 1000),
    'images': fog_real_images
}

quality = calculate_weather_simulation_quality(
    sim_fog, real_fog, weather_type='fog'
)

# Fog quality is critical for safety
assert quality['visibility_mean_error'] < 50, "Fog visibility too inaccurate"
assert quality['spatial_correlation'] > 0.6, "Fog spatial patterns unrealistic"
```

#### Lighting Transition Validation
```python
# Dusk lighting transition
sim_dusk = {
    'lighting': np.random.beta(3, 3, 1000),  # Gradual transition
    'images': dusk_sim_images
}

real_dusk = {
    'lighting': np.random.beta(3.2, 2.9, 1000),
    'images': dusk_real_images
}

quality = calculate_weather_simulation_quality(
    sim_dusk, real_dusk, 
    weather_type='lighting',
    metrics=['lighting_histogram', 'temporal_consistency']
)

# Lighting transitions should be smooth
assert quality['temporal_stability'] > 0.85, "Lighting transition too abrupt"
```

### Integration with Perception Testing

```python
# Test perception pipeline under various weather conditions
weather_conditions = ['clear', 'rain', 'fog', 'snow']
realism_scores = {}

for condition in weather_conditions:
    # Run weather simulation
    sim_data = generate_weather_scenario(condition)
    real_data = load_real_weather_data(condition)
    
    # Validate weather realism
    weather_quality = calculate_weather_simulation_quality(
        sim_data, real_data, weather_type=condition
    )
    
    realism_scores[condition] = weather_quality['overall_realism_score']
    
    # Only test perception if weather is realistic enough
    if weather_quality['overall_realism_score'] > 70:
        perception_results = test_perception_pipeline(sim_data)
        print(f"{condition}: Perception mAP = {perception_results['mAP']:.2f}")
    else:
        print(f"⚠ Skipping {condition}: Weather realism too low ({weather_quality['overall_realism_score']:.1f})")

# Report weather simulation coverage
print(f"\nWeather Simulation Quality:")
for condition, score in realism_scores.items():
    status = "✓" if score > 75 else "⚠"
    print(f"  {status} {condition}: {score:.1f}/100")
```

### Best Practices for Weather Validation

**1. Collect Real-World Weather Data**
- Record sensor data in actual weather conditions
- Include diverse scenarios (light/moderate/heavy)
- Capture temporal transitions (onset, steady-state, clearing)
- Document environmental context (temperature, wind, location)

**2. Validate Across Weather Intensity Ranges**
```python
# Test multiple rain intensities
rain_intensities = [
    ('light', 0, 2.5),      # mm/h
    ('moderate', 2.5, 10),
    ('heavy', 10, 50)
]

for name, min_rate, max_rate in rain_intensities:
    sim_data = generate_rain_scenario(min_rate, max_rate)
    real_data = load_real_rain_data(name)
    quality = calculate_weather_simulation_quality(sim_data, real_data)
    assert quality['overall_realism_score'] > 70, f"{name} rain unrealistic"
```

**3. Check Sensor-Specific Effects**
- Camera: Droplets, reduced contrast, lens artifacts
- LiDAR: Absorption, scattering, reduced range
- Radar: Enhanced reflections (wet surfaces)

**4. Validate Temporal Dynamics**
- Weather onset (clear → rain transition)
- Intensity variations over time
- Weather clearing (fog dissipation)

**5. Test Geographic Variations**
- Tropical rain (high intensity, short duration)
- Fog patterns (coastal vs. valley)
- Snow characteristics (wet vs. dry, accumulation)

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Unrealistic rain | KL divergence > 1.0 | Adjust rain particle distribution |
| Flickering weather | Temporal stability < 0.5 | Add temporal smoothing filter |
| Wrong visibility | Mean error > 100m | Calibrate fog density model |
| Uniform fog | Spatial correlation < 0.4 | Add spatial variation/patches |
| Sudden transitions | Frame correlation < 0.7 | Implement gradual onset/offset |
| Wrong lighting | KL divergence > 0.6 | Match sun position and intensity |
| Shadow artifacts | Coverage error > 0.2 | Fix shadow casting algorithm |

## Vehicle Dynamics Quality

### Purpose
Validate the realism of vehicle motion and physics in simulated traffic, including acceleration profiles, braking behavior, lateral dynamics, and trajectory smoothness. Critical for planning/control validation and multi-agent simulation.

### Metrics

#### Acceleration Profile Validation
**What It Measures:** How well simulated vehicles match real-world acceleration patterns

**Metrics:**
- **KL Divergence**: Measures distribution similarity between sim/real acceleration histograms
  - < 0.3: Excellent match
  - 0.3-0.8: Good match
  - \> 1.0: Poor match
- **Mean Absolute Error**: Average difference in acceleration magnitude (m/s²)
  - < 0.5 m/s²: Excellent
  - 0.5-1.5 m/s²: Acceptable
  - \> 2.0 m/s²: Poor

**When to Use:** Validating acceleration behavior in highway merges, intersections, stop-and-go traffic

#### Braking Distance and Deceleration
**What It Measures:** Realism of braking physics and stopping distances

**Metrics:**
- **Braking Distance Error**: Difference between simulated and real stopping distances
  - < 10%: Excellent
  - 10-25%: Acceptable
  - \> 30%: Unsafe
- **Deceleration Rate Statistics**:
  - Mean: Normal braking ~3-5 m/s², emergency ~6-9 m/s²
  - Max: Should not exceed physical limits (~10 m/s² on dry road)
  - Distribution: Should match real-world braking patterns

**When to Use:** Safety-critical scenarios, emergency braking validation, AEB testing

#### Lateral Dynamics
**What It Measures:** Cornering, lane changes, and turning behavior

**Metrics:**
- **Lateral Acceleration**: Should match vehicle physics
  - Highway lane change: 2-4 m/s²
  - Sharp turn: 5-8 m/s²
  - Emergency maneuver: up to 10 m/s²
- **Lateral Jerk**: Rate of change of lateral acceleration
  - Ratio between sim/real should be 0.8-1.2 for realism
  - High jerk = uncomfortable/unrealistic maneuvers
- **Steering Smoothness**: Variance in steering angle changes

**When to Use:** Lane change validation, intersection turns, evasive maneuvers

#### Trajectory Smoothness
**What It Measures:** How natural and smooth vehicle paths are

**Metrics:**
- **Longitudinal Jerk**: Rate of acceleration change (m/s³)
  - < 2 m/s³: Smooth, comfortable
  - 2-5 m/s³: Acceptable
  - \> 5 m/s³: Jerky, unrealistic
- **Lateral Jerk**: Similar thresholds for lateral motion
- **Path Curvature Variance**: Measures smoothness of steering

**When to Use:** Comfort evaluation, path planning validation, multi-agent simulation

#### Speed Distribution Matching
**What It Measures:** Whether vehicles drive at realistic speeds for the scenario

**Metrics:**
- **KL Divergence**: Distribution similarity
  - < 0.2: Excellent match
  - 0.2-0.5: Good
  - \> 0.8: Poor
- **Mean Speed Error**: Absolute difference in average speeds (m/s)
- **KS Statistic**: Kolmogorov-Smirnov test for distribution matching
  - p-value > 0.05: Distributions statistically similar

**When to Use:** Highway scenarios, urban traffic, speed limit compliance

#### Reaction Time Estimation
**What It Measures:** Delay between stimulus and vehicle response

**Metrics:**
- **Mean Reaction Time**: Typical human range 0.7-2.5 seconds
  - Alert driver: 0.7-1.0 s
  - Average driver: 1.0-1.5 s
  - Distracted: 1.5-2.5 s
- **Reaction Time Variance**: Should show realistic variability

**When to Use:** Multi-agent behavior validation, safety margin calculation

### API Usage

```python
from admetrics.simulation import calculate_vehicle_dynamics_quality
import numpy as np

# Simulated vehicle trajectories: (N, T, D) where D >= 4 [x, y, vx, vy]
sim_trajectories = np.load('sim_vehicle_trajectories.npy')  # (50, 100, 6)
real_trajectories = np.load('real_vehicle_trajectories.npy')  # (50, 100, 6)

# Full dynamics validation
results = calculate_vehicle_dynamics_quality(
    sim_trajectories=sim_trajectories,
    real_trajectories=real_trajectories,
    dt=0.1,  # 10 Hz sampling
    maneuver_type='mixed',  # or 'acceleration', 'braking', 'lane_change'
    metrics='all'  # or list like ['acceleration_profile', 'braking_distance']
)

# Check overall quality
print(f"Overall Dynamics Score: {results['overall_score']:.1f}/100")

# Acceleration validation
if 'acceleration_kl_divergence' in results:
    print(f"Acceleration Distribution Match: KL={results['acceleration_kl_divergence']:.3f}")
    print(f"Acceleration Mean Error: {results['acceleration_mean_error']:.3f} m/s²")

# Braking validation
if 'braking_distance_error' in results:
    print(f"Braking Distance Error: {results['braking_distance_error']:.1f}%")
    if 'deceleration_mean' in results:
        print(f"Mean Deceleration: {results['deceleration_mean']:.2f} m/s²")

# Lateral dynamics
if 'lateral_accel_max_sim' in results:
    print(f"Max Lateral Accel (sim): {results['lateral_accel_max_sim']:.2f} m/s²")
    print(f"Max Lateral Accel (real): {results['lateral_accel_max_real']:.2f} m/s²")

# Smoothness
if 'longitudinal_jerk_mean_sim' in results:
    jerk_sim = results['longitudinal_jerk_mean_sim']
    jerk_real = results['longitudinal_jerk_mean_real']
    print(f"Longitudinal Jerk - Sim: {jerk_sim:.2f}, Real: {jerk_real:.2f} m/s³")

# Speed distribution
if 'speed_kl_divergence' in results:
    print(f"Speed Distribution Match: KL={results['speed_kl_divergence']:.3f}")
    print(f"Speed Mean Error: {results['speed_mean_error']:.2f} m/s")

# Reaction time
if 'reaction_time_mean' in results:
    print(f"Estimated Reaction Time: {results['reaction_time_mean']:.2f}s ± {results.get('reaction_time_std', 0):.2f}s")
```

### Scenario-Specific Validation

#### Highway Merging
```python
# Focus on acceleration and lateral dynamics
results = calculate_vehicle_dynamics_quality(
    sim_traj, real_traj,
    maneuver_type='acceleration',
    metrics=['acceleration_profile', 'lateral_dynamics', 'trajectory_smoothness']
)

# Expect smooth acceleration and moderate lateral movement
assert results['acceleration_mean_error'] < 1.0  # m/s²
assert results.get('lateral_accel_max_sim', 0) < 5.0  # m/s² (highway merge)
assert results.get('longitudinal_jerk_mean_sim', 0) < 3.0  # m/s³ (comfortable)
```

#### Emergency Braking
```python
# Focus on braking performance
results = calculate_vehicle_dynamics_quality(
    sim_traj, real_traj,
    maneuver_type='braking',
    metrics=['braking_distance']
)

# Check safety-critical braking
if 'deceleration_mean' in results:
    assert 6.0 <= results['deceleration_mean'] <= 10.0  # Emergency braking range
if 'braking_distance_error' in results:
    assert abs(results['braking_distance_error']) < 15.0  # Within 15% is acceptable
```

#### Urban Lane Changes
```python
# Focus on lateral dynamics
results = calculate_vehicle_dynamics_quality(
    sim_traj, real_traj,
    maneuver_type='lane_change',
    metrics=['lateral_dynamics', 'trajectory_smoothness']
)

# Validate realistic lane change
lateral_accel = results.get('lateral_accel_max_sim', 0)
assert 2.0 <= lateral_accel <= 4.0  # Typical urban lane change

lateral_jerk = results.get('lateral_jerk_mean_sim', 0)
assert lateral_jerk < 5.0  # Should be smooth
```

### Interpretation Guidelines

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **Overall Score** | > 80 | 60-80 | 40-60 | < 40 |
| **Acceleration KL** | < 0.3 | 0.3-0.8 | 0.8-1.5 | > 1.5 |
| **Braking Distance Error** | < 10% | 10-20% | 20-30% | > 30% |
| **Lateral Accel Error** | < 15% | 15-30% | 30-50% | > 50% |
| **Longitudinal Jerk** | < 2 m/s³ | 2-4 m/s³ | 4-6 m/s³ | > 6 m/s³ |
| **Speed KL Divergence** | < 0.2 | 0.2-0.5 | 0.5-1.0 | > 1.0 |
| **Reaction Time** | 0.7-1.5s | 1.5-2.0s | 2.0-2.5s | > 2.5s |

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Unrealistic acceleration | KL divergence > 1.0 | Calibrate powertrain model, check mass/power ratio |
| Excessive braking distance | Distance error > 25% | Adjust tire friction model, brake force limits |
| Jerky motion | Jerk > 5 m/s³ | Add motion smoothing, use higher-order planners |
| Wrong lateral behavior | Accel error > 40% | Tune steering model, check tire slip curves |
| Incorrect speeds | Speed KL > 0.8 | Adjust speed controller, check traffic flow model |
| Instant reactions | Reaction time < 0.5s | Add human-like delays to control inputs |
| Sliding vehicles | Lateral accel > 10 m/s² | Fix tire model, check friction coefficients |

### Best Practices for Dynamics Validation

1. **Match Scenario Types**: Use appropriate real-world data
   - Highway: High speeds, smooth acceleration
   - Urban: Frequent stops, lower speeds
   - Emergency: Maximum braking/steering

2. **Control Timestep**: Consistent dt critical for acceleration computation
   ```python
   # Ensure consistent timestep
   dt = 0.1  # 10 Hz is typical
   assert np.allclose(np.diff(timestamps), dt, atol=0.01)
   ```

3. **Validate Physics Limits**: Check for violations
   - Longitudinal accel: Typically < 4 m/s² (normal), < 10 m/s² (emergency)
   - Lateral accel: Typically < 8 m/s² (depends on speed and tire friction)
   - Jerk: < 5 m/s³ for passenger comfort

4. **Consider Vehicle Types**: Different vehicles have different dynamics
   - Sports car: Higher acceleration, lateral capability
   - SUV: Lower lateral limits, longer braking
   - Truck: Much lower acceleration, much longer braking

5. **Test Edge Cases**:
   ```python
   # Test with stationary vehicles
   # Test with single vehicle
   # Test with incomplete trajectories
   # Test with minimal timesteps
   ```

6. **Compare Distributions**: Don't just check means
   - Use KL divergence for overall pattern matching
   - Check variance and extremes
   - Validate temporal correlations

## Best Practices

### 1. Establish Baselines

**Collect real-world data:**
```python
# Record sensor data in various conditions
real_camera_data = collect_real_camera_images(scenarios)
real_lidar_data = collect_real_lidar_scans(scenarios)
```

**Evaluate simulation:**
```python
# Compare on same scenarios
quality = calculate_camera_image_quality(sim_data, real_data)
assert quality['psnr'] > 25, "Image quality too poor"
```

### 2. Scenario Coverage

**Test diverse conditions:**
- Time of day (dawn, day, dusk, night)
- Weather (clear, rain, fog, snow)
- Traffic density (sparse, moderate, dense)
- Road types (highway, urban, rural)

### 3. Sensor Configuration

**Match real hardware:**
```python
# Configure simulator to match real sensors
lidar_config = {
    'channels': 64,
    'range': 100.0,
    'rotation_frequency': 10,  # Hz
    'points_per_second': 1.3e6,
    'noise_std': 0.02  # meters
}
```

### 4. Iterative Refinement

**Identify and fix issues:**
1. Run quality metrics
2. Identify largest gaps
3. Improve simulator (materials, lighting, physics)
4. Re-evaluate
5. Repeat until gaps acceptable

### 5. Validation Pipeline

```python
def validate_simulation(sim_data, real_data):
    """Complete validation pipeline."""
    
    results = {}
    
    # Camera quality
    cam_quality = calculate_camera_image_quality(sim_data['camera'], real_data['camera'])
    results['camera_psnr'] = cam_quality['psnr']
    
    # LiDAR quality
    lidar_quality = calculate_lidar_point_cloud_quality(sim_data['lidar'], real_data['lidar'])
    results['lidar_chamfer'] = lidar_quality['chamfer_distance']
    
    # Noise characteristics
    noise_quality = calculate_sensor_noise_characteristics(
        sim_data['noise_samples'],
        real_data['noise_samples']
    )
    results['noise_std_ratio'] = noise_quality['noise_std_ratio']
    
    # Sim2Real gap
    gap = calculate_perception_sim2real_gap(sim_data['detections'], real_data['detections'])
    results['performance_drop'] = gap['performance_drop_pct']
    
    # Summary
    passed = (
        results['camera_psnr'] > 25 and
        results['lidar_chamfer'] < 0.5 and
        abs(results['noise_std_ratio'] - 1.0) < 0.3 and
        results['performance_drop'] < 20
    )
    
    return results, passed
```

### 6. Document Limitations

**Be transparent about sim quality:**
- Known artifacts (e.g., "no rain simulation")
- Performance gaps (e.g., "15% drop in night scenarios")
- Mitigation strategies (e.g., "domain randomization applied")

## Simulator Comparisons

### Popular AV Simulators

| Simulator | Strengths | Weaknesses | Quality Focus |
|-----------|-----------|------------|---------------|
| **CARLA** | Open-source, good graphics | Limited physics | Visual realism |
| **LGSVL** | Multi-sensor, ROS integration | Discontinued | Sensor accuracy |
| **AirSim** | Photorealistic, UE4 engine | Resource intensive | Camera quality |
| **Waymo Sim** | Real-world scenarios | Closed-source | Behavior realism |
| **nuPlan** | Real sensor logs + replay | Not full sim | Data-driven |
| **MetaDrive** | Lightweight, procedural | Simple graphics | Scalability |

### Quality Assessment by Simulator

**Example results (illustrative):**

```
Simulator: CARLA
  Camera PSNR: 28 dB (Good)
  LiDAR Chamfer: 0.35m (Good)
  Performance Drop: 18% (Acceptable)

Simulator: AirSim  
  Camera PSNR: 32 dB (Excellent)
  LiDAR Chamfer: 0.45m (Acceptable)
  Performance Drop: 22% (Acceptable)

Simulator: MetaDrive
  Camera PSNR: 22 dB (Poor)
  LiDAR Chamfer: 0.25m (Good)
  Performance Drop: 28% (Significant)
```

## References

1. **CARLA: An Open Urban Driving Simulator**
   - Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017)
   - Conference on Robot Learning (CoRL) 2017
   - https://arxiv.org/abs/1711.03938
   - https://carla.org
   - https://github.com/carla-simulator/carla
   - Open-source simulator with flexible sensor configuration and evaluation metrics

2. **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**
   - Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017)
   - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2017
   - https://arxiv.org/abs/1703.06907
   - Techniques for improving sim-to-real transfer through randomization

3. **Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey**
   - Zhao, W., Queralta, J. P., & Westerlund, T. (2020)
   - IEEE Conference on Automation Science and Engineering (CASE) 2020
   - https://arxiv.org/abs/2009.13303
   - Comprehensive survey of simulation-to-reality transfer methods

4. **Simulating LiDAR Point Cloud for Autonomous Driving using Real-world Scenes and Traffic Flows**
   - Fang, J., Zhou, D., Yan, F., Zhao, T., Zhang, F., Ma, Y., Wang, L., & Yang, R. (2018)
   - https://arxiv.org/abs/1811.07112
   - Realistic LiDAR simulation for autonomous driving validation

5. **Real-to-Sim: Predicting Residual Errors of Robotic Systems with Sparse Data using a Learning-based Unscented Kalman Filter**
   - Kurup, A., & Borrelli, F. (2021)
   - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2021
   - https://arxiv.org/abs/2103.10403
   - Learning-based approach for modeling sensor noise and errors



### Weather Simulation

1. **Simulating Photo-realistic Snow and Fog on Existing Images for Enhanced CNN Training and Evaluation**
   - Hahner, M., Dai, D., Sakaridis, C., Zaech, J. N., & Van Gool, L. (2019)
   - https://arxiv.org/abs/1812.00606
   - Fog/snow augmentation for autonomous driving

2. **Seeing Through Fog Without Seeing Fog**
   - Bijelic, M., Gruber, T., & Ritter, W. (2020)
   - https://arxiv.org/abs/2004.08637
   - Adverse weather perception

### Physics/Dynamics

3. **Learning to Drive from Simulation**
   - Pan, X., You, Y., Wang, Z., & Lu, C. (2020)
   - https://arxiv.org/abs/1608.01230
   - Vehicle dynamics in simulation

4. **nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles**
   - Caesar, H., et al. (2021)
   - https://arxiv.org/abs/2106.11810
   - Closed-loop planning with physics

### Semantic Validation

5. **SceneGen: Learning to Generate Realistic Traffic Scenes**
   - Tan, S., Wong, K., Wang, S., Manivasagam, S., Ren, M., & Urtasun, R. (2021)
   - https://arxiv.org/abs/2101.06541
   - Realistic scene generation

## See Also

- [END_TO_END_METRICS.md](END_TO_END_METRICS.md) - Planning and driving evaluation
- [DETECTION_METRICS.md](DETECTION_METRICS.md) - Object detection metrics
- [LOCALIZATION_METRICS.md](LOCALIZATION_METRICS.md) - Ego pose metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
