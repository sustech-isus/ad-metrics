"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_weather_simulation_quality(
    sim_data: Dict[str, np.ndarray],
    real_data: Dict[str, np.ndarray],
    weather_type: str = 'rain',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate weather and environmental simulation quality.
    
    Validates realism of weather effects (rain, fog, snow) and environmental conditions
    (lighting, shadows) in simulation compared to real-world data. Critical for
    testing perception robustness across diverse weather scenarios.
    
    Args:
        sim_data: Dictionary containing simulated weather data:
            - 'intensity': Weather intensity values (e.g., rain rate mm/h, fog density)
            - 'visibility': Visibility range in meters
            - 'images': Optional camera images showing weather effects (N, H, W, C)
            - 'lighting': Lighting intensity values (0-1 normalized)
            - 'shadow_coverage': Fraction of scene covered by shadows (0-1)
            - 'reflectance': Surface reflectance values for wet conditions
        real_data: Dictionary with same structure as sim_data from real-world
        weather_type: Type of weather to validate. Options:
            - 'rain': Rain simulation quality
            - 'fog': Fog/haze simulation quality  
            - 'snow': Snow simulation quality
            - 'lighting': Day/night/dusk lighting transitions
            - 'shadows': Shadow realism and coverage
        metrics: List of metrics to compute. Options:
            - 'intensity_distribution': KL divergence of weather intensity
            - 'visibility_range': Visibility distance statistics
            - 'temporal_consistency': Frame-to-frame weather stability
            - 'spatial_distribution': Spatial pattern similarity
            - 'particle_density': Density of rain/snow particles
            - 'lighting_histogram': Lighting distribution similarity
            - 'shadow_realism': Shadow edge sharpness and coverage
            
    Returns:
        Dictionary with requested metrics:
            - intensity_kl_divergence: KL divergence of intensity distributions
            - visibility_mean_error: Mean absolute error in visibility (meters)
            - visibility_std_ratio: Ratio of visibility std dev
            - temporal_stability: Frame-to-frame correlation (0-1)
            - spatial_correlation: Spatial pattern correlation coefficient
            - particle_density_ratio: Ratio of particle densities
            - lighting_kl_divergence: KL divergence of lighting histograms
            - shadow_coverage_error: Absolute error in shadow coverage
            - shadow_edge_sharpness: Shadow edge gradient similarity
            - overall_realism_score: Combined realism score (0-100)
        
    Example:
        >>> # Rain simulation validation
        >>> sim_data = {
        ...     'intensity': np.random.gamma(2, 5, 1000),  # Rain rate mm/h
        ...     'visibility': np.random.normal(200, 50, 1000),  # Visibility in meters
        ...     'images': np.random.rand(10, 224, 224, 3) * 255
        ... }
        >>> real_data = {
        ...     'intensity': np.random.gamma(2.1, 4.8, 1000),
        ...     'visibility': np.random.normal(195, 55, 1000),
        ...     'images': np.random.rand(10, 224, 224, 3) * 255
        ... }
        >>> quality = calculate_weather_simulation_quality(
        ...     sim_data, real_data, weather_type='rain'
        ... )
        >>> print(f"Rain intensity realism: {quality['intensity_kl_divergence']:.3f}")
        
    Notes:
        - Weather intensity thresholds (typical ranges):
            * Light rain: 0-2.5 mm/h
            * Moderate rain: 2.5-10 mm/h
            * Heavy rain: 10-50 mm/h
            * Light fog: visibility 500-1000m
            * Moderate fog: visibility 200-500m
            * Dense fog: visibility < 200m
        - Temporal stability > 0.9 indicates good frame-to-frame consistency
        - Spatial correlation > 0.8 indicates realistic weather patterns
        - Overall realism score > 75 is considered good for production use
    """
    if metrics is None:
        metrics = [
            'intensity_distribution', 'visibility_range', 'temporal_consistency',
            'spatial_distribution'
        ]
    
    # Handle empty metrics list - use defaults
    if len(metrics) == 0:
        metrics = [
            'intensity_distribution', 'visibility_range', 'temporal_consistency',
            'spatial_distribution'
        ]
    
    results = {}
    
    # Intensity distribution comparison
    if 'intensity_distribution' in metrics and 'intensity' in sim_data and 'intensity' in real_data:
        sim_intensity = sim_data['intensity'].flatten()
        real_intensity = real_data['intensity'].flatten()
        
        # Remove zeros/invalid values
        sim_intensity = sim_intensity[sim_intensity > 0]
        real_intensity = real_intensity[real_intensity > 0]
        
        # Check if we have valid data after filtering
        if len(sim_intensity) > 0 and len(real_intensity) > 0:
            # Calculate KL divergence using histograms
            bins = np.linspace(0, max(sim_intensity.max(), real_intensity.max()), 50)
            sim_hist, _ = np.histogram(sim_intensity, bins=bins, density=True)
            real_hist, _ = np.histogram(real_intensity, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            sim_hist = sim_hist + 1e-10
            real_hist = real_hist + 1e-10
            sim_hist = sim_hist / sim_hist.sum()
            real_hist = real_hist / real_hist.sum()
            
            kl_div = stats.entropy(sim_hist, real_hist)
            results['intensity_kl_divergence'] = float(kl_div)
            results['intensity_mean_sim'] = float(sim_intensity.mean())
            results['intensity_mean_real'] = float(real_intensity.mean())
            results['intensity_std_ratio'] = float(sim_intensity.std() / (real_intensity.std() + 1e-6))
    
    # Visibility range validation
    if 'visibility_range' in metrics and 'visibility' in sim_data and 'visibility' in real_data:
        sim_vis = sim_data['visibility'].flatten()
        real_vis = real_data['visibility'].flatten()
        
        results['visibility_mean_error'] = float(np.abs(sim_vis.mean() - real_vis.mean()))
        results['visibility_std_ratio'] = float(sim_vis.std() / (real_vis.std() + 1e-6))
        results['visibility_median_sim'] = float(np.median(sim_vis))
        results['visibility_median_real'] = float(np.median(real_vis))
        
        # Statistical test for distribution similarity
        ks_stat, ks_pval = stats.ks_2samp(sim_vis, real_vis)
        results['visibility_ks_statistic'] = float(ks_stat)
        results['visibility_ks_pvalue'] = float(ks_pval)
    
    # Temporal consistency (frame-to-frame stability)
    if 'temporal_consistency' in metrics and 'intensity' in sim_data:
        # Reshape to time series if needed
        sim_intensity = sim_data['intensity'].flatten()
        
        if len(sim_intensity) > 1:
            # Calculate autocorrelation at lag 1
            sim_diffs = np.diff(sim_intensity)
            temporal_stability = 1.0 - (np.std(sim_diffs) / (np.std(sim_intensity) + 1e-6))
            results['temporal_stability'] = float(np.clip(temporal_stability, 0, 1))
            
            # Frame-to-frame correlation
            if len(sim_intensity) > 2:
                corr = np.corrcoef(sim_intensity[:-1], sim_intensity[1:])[0, 1]
                results['frame_to_frame_correlation'] = float(corr)
    
    # Spatial distribution (for image-based weather effects)
    if 'spatial_distribution' in metrics and 'images' in sim_data and 'images' in real_data:
        sim_imgs = sim_data['images']
        real_imgs = real_data['images']
        
        if sim_imgs.shape == real_imgs.shape and len(sim_imgs.shape) == 4:
            # Calculate spatial gradients
            sim_gradients = np.gradient(sim_imgs.mean(axis=-1).mean(axis=0))
            real_gradients = np.gradient(real_imgs.mean(axis=-1).mean(axis=0))
            
            # Correlation of gradient magnitudes
            sim_grad_mag = np.sqrt(sim_gradients[0]**2 + sim_gradients[1]**2).flatten()
            real_grad_mag = np.sqrt(real_gradients[0]**2 + real_gradients[1]**2).flatten()
            
            if len(sim_grad_mag) > 1 and len(real_grad_mag) > 1:
                spatial_corr = np.corrcoef(sim_grad_mag, real_grad_mag)[0, 1]
                results['spatial_correlation'] = float(spatial_corr)
    
    # Particle density (for rain/snow)
    if 'particle_density' in metrics and weather_type in ['rain', 'snow']:
        if 'images' in sim_data and 'images' in real_data:
            # Estimate particle density from high-frequency components
            sim_imgs = sim_data['images']
            real_imgs = real_data['images']
            
            # Use variance of pixel values as proxy for particle density
            sim_variance = np.var(sim_imgs, axis=(1, 2, 3))
            real_variance = np.var(real_imgs, axis=(1, 2, 3))
            
            results['particle_density_ratio'] = float(
                sim_variance.mean() / (real_variance.mean() + 1e-6)
            )
            results['particle_density_sim'] = float(sim_variance.mean())
            results['particle_density_real'] = float(real_variance.mean())
    
    # Lighting histogram (for day/night/dusk transitions)
    if 'lighting_histogram' in metrics and 'lighting' in sim_data and 'lighting' in real_data:
        sim_lighting = sim_data['lighting'].flatten()
        real_lighting = real_data['lighting'].flatten()
        
        # Histogram comparison
        bins = np.linspace(0, 1, 20)
        sim_hist, _ = np.histogram(sim_lighting, bins=bins, density=True)
        real_hist, _ = np.histogram(real_lighting, bins=bins, density=True)
        
        sim_hist = sim_hist + 1e-10
        real_hist = real_hist + 1e-10
        sim_hist = sim_hist / sim_hist.sum()
        real_hist = real_hist / real_hist.sum()
        
        lighting_kl = stats.entropy(sim_hist, real_hist)
        results['lighting_kl_divergence'] = float(lighting_kl)
        results['lighting_mean_sim'] = float(sim_lighting.mean())
        results['lighting_mean_real'] = float(real_lighting.mean())
    
    # Shadow realism
    if 'shadow_realism' in metrics and 'shadow_coverage' in sim_data and 'shadow_coverage' in real_data:
        sim_shadows = sim_data['shadow_coverage'].flatten()
        real_shadows = real_data['shadow_coverage'].flatten()
        
        results['shadow_coverage_error'] = float(
            np.abs(sim_shadows.mean() - real_shadows.mean())
        )
        results['shadow_coverage_sim'] = float(sim_shadows.mean())
        results['shadow_coverage_real'] = float(real_shadows.mean())
        
        # Shadow edge sharpness (if images available)
        if 'images' in sim_data and 'images' in real_data:
            sim_imgs = sim_data['images']
            real_imgs = real_data['images']
            
            # Calculate edge gradients (average across frames and channels first)
            sim_gray = sim_imgs.mean(axis=-1)  # (N, H, W)
            real_gray = real_imgs.mean(axis=-1)  # (N, H, W)
            
            # Compute gradients along spatial dimensions
            sim_grad_y = np.gradient(sim_gray, axis=1)  # Vertical gradient
            sim_grad_x = np.gradient(sim_gray, axis=2)  # Horizontal gradient
            real_grad_y = np.gradient(real_gray, axis=1)
            real_grad_x = np.gradient(real_gray, axis=2)
            
            # Edge strength is magnitude of gradient
            sim_edge_strength = np.sqrt(sim_grad_y**2 + sim_grad_x**2).mean()
            real_edge_strength = np.sqrt(real_grad_y**2 + real_grad_x**2).mean()
            
            # Avoid division by zero
            if real_edge_strength > 1e-6:
                results['shadow_edge_sharpness'] = float(
                    sim_edge_strength / real_edge_strength
                )
            else:
                results['shadow_edge_sharpness'] = 1.0  # Both have no edges
    
    # Overall realism score (0-100)
    # Combines multiple metrics into a single quality score
    if len(results) > 0:
        score_components = []
        
        # Intensity distribution (lower KL = better, normalize)
        if 'intensity_kl_divergence' in results:
            intensity_score = 100 * np.exp(-results['intensity_kl_divergence'])
            score_components.append(intensity_score)
        
        # Visibility accuracy (lower error = better)
        if 'visibility_mean_error' in results:
            # Normalize by typical visibility range (1000m)
            vis_score = 100 * (1 - np.clip(results['visibility_mean_error'] / 1000, 0, 1))
            score_components.append(vis_score)
        
        # Temporal stability (higher = better)
        if 'temporal_stability' in results:
            temp_score = 100 * results['temporal_stability']
            score_components.append(temp_score)
        
        # Spatial correlation (higher = better)
        if 'spatial_correlation' in results:
            spatial_score = 100 * (results['spatial_correlation'] + 1) / 2  # Map -1,1 to 0,100
            score_components.append(spatial_score)
        
        if score_components:
            results['overall_realism_score'] = float(np.mean(score_components))
    
    return results


