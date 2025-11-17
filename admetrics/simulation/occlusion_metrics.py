"""
Occlusion and Visibility Quality Metrics for Simulation Validation.

Evaluates the realism of occlusion patterns and visibility distributions in simulated
detection data compared to real-world sensor observations.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats


def calculate_occlusion_visibility_quality(
    sim_detections: Dict[str, np.ndarray],
    real_detections: Dict[str, np.ndarray],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate occlusion and visibility realism in simulation.
    
    Validates whether simulated detections exhibit realistic occlusion patterns,
    visibility distributions, and detection range characteristics compared to
    real-world sensor data.
    
    Args:
        sim_detections: Simulated detection data with:
            - 'positions': Array of detection positions (N, 3) [x, y, z]
            - 'occlusion_levels': Array of occlusion percentages (N,) in range [0, 1]
            - 'truncation_levels': Array of truncation percentages (N,) in range [0, 1]
            - 'visibility_scores': Array of visibility scores (N,) in range [0, 1]
            - 'detection_ranges': Array of distances from sensor (N,) in meters
            - 'object_sizes': Array of object sizes (N, 3) [length, width, height]
        real_detections: Real-world detection data, same format
        metrics: List of metrics to compute. Options:
            - 'occlusion_distribution': Occlusion level distribution matching
            - 'truncation_distribution': Truncation level distribution
            - 'visibility_distribution': Visibility score distribution
            - 'range_visibility': Detection range vs visibility correlation
            - 'occlusion_by_distance': Occlusion patterns at different ranges
            - 'all': Compute all metrics
            
    Returns:
        Dictionary with metrics:
            - occlusion_kl_divergence: KL divergence of occlusion distributions
            - occlusion_mean_error: Mean absolute error in occlusion levels
            - truncation_kl_divergence: KL divergence of truncation distributions
            - truncation_mean_error: Mean absolute error in truncation levels
            - visibility_kl_divergence: KL divergence of visibility scores
            - visibility_correlation: Correlation between sim/real visibility
            - range_visibility_correlation_sim: Range-visibility correlation (sim)
            - range_visibility_correlation_real: Range-visibility correlation (real)
            - near_occlusion_ratio: Ratio of occluded objects at close range (<20m)
            - far_occlusion_ratio: Ratio of occluded objects at far range (>50m)
            - partial_occlusion_frequency_sim: Frequency of partially occluded objects (sim)
            - partial_occlusion_frequency_real: Frequency of partially occluded objects (real)
            - full_occlusion_frequency_sim: Frequency of fully occluded objects (sim)
            - full_occlusion_frequency_real: Frequency of fully occluded objects (real)
            - overall_occlusion_quality_score: Composite quality score (0-100)
    """
    results = {}
    
    if metrics is None or metrics == 'all':
        metrics = ['occlusion_distribution', 'truncation_distribution', 
                  'visibility_distribution', 'range_visibility', 'occlusion_by_distance']
    
    # 1. Occlusion Distribution Validation
    if 'occlusion_distribution' in metrics:
        if 'occlusion_levels' in sim_detections and 'occlusion_levels' in real_detections:
            sim_occ = np.array(sim_detections['occlusion_levels'])
            real_occ = np.array(real_detections['occlusion_levels'])
            
            if len(sim_occ) > 0 and len(real_occ) > 0:
                # Occlusion level distribution (0%, 10%, 20%, ..., 100%)
                bins = np.linspace(0, 1, 11)
                sim_occ_hist, _ = np.histogram(sim_occ, bins=bins, density=True)
                real_occ_hist, _ = np.histogram(real_occ, bins=bins, density=True)
                
                # Normalize
                sim_occ_dist = (sim_occ_hist + 1e-10) / (sim_occ_hist.sum() + 1e-10 * len(sim_occ_hist))
                real_occ_dist = (real_occ_hist + 1e-10) / (real_occ_hist.sum() + 1e-10 * len(real_occ_hist))
                
                # KL divergence
                kl_div = float(np.sum(real_occ_dist * np.log((real_occ_dist + 1e-10) / (sim_occ_dist + 1e-10))))
                results['occlusion_kl_divergence'] = kl_div
                results['occlusion_mean_error'] = float(np.abs(sim_occ.mean() - real_occ.mean()))
                
                # Partial vs full occlusion frequencies
                # Partial: 0.1 < occlusion < 0.9
                # Full: occlusion >= 0.9
                sim_partial = np.sum((sim_occ > 0.1) & (sim_occ < 0.9)) / len(sim_occ)
                real_partial = np.sum((real_occ > 0.1) & (real_occ < 0.9)) / len(real_occ)
                results['partial_occlusion_frequency_sim'] = float(sim_partial)
                results['partial_occlusion_frequency_real'] = float(real_partial)
                
                sim_full = np.sum(sim_occ >= 0.9) / len(sim_occ)
                real_full = np.sum(real_occ >= 0.9) / len(real_occ)
                results['full_occlusion_frequency_sim'] = float(sim_full)
                results['full_occlusion_frequency_real'] = float(real_full)
    
    # 2. Truncation Distribution
    if 'truncation_distribution' in metrics:
        if 'truncation_levels' in sim_detections and 'truncation_levels' in real_detections:
            sim_trunc = np.array(sim_detections['truncation_levels'])
            real_trunc = np.array(real_detections['truncation_levels'])
            
            if len(sim_trunc) > 0 and len(real_trunc) > 0:
                bins = np.linspace(0, 1, 11)
                sim_trunc_hist, _ = np.histogram(sim_trunc, bins=bins, density=True)
                real_trunc_hist, _ = np.histogram(real_trunc, bins=bins, density=True)
                
                sim_trunc_dist = (sim_trunc_hist + 1e-10) / (sim_trunc_hist.sum() + 1e-10 * len(sim_trunc_hist))
                real_trunc_dist = (real_trunc_hist + 1e-10) / (real_trunc_hist.sum() + 1e-10 * len(real_trunc_hist))
                
                kl_div = float(np.sum(real_trunc_dist * np.log((real_trunc_dist + 1e-10) / (sim_trunc_dist + 1e-10))))
                results['truncation_kl_divergence'] = kl_div
                results['truncation_mean_error'] = float(np.abs(sim_trunc.mean() - real_trunc.mean()))
    
    # 3. Visibility Distribution
    if 'visibility_distribution' in metrics:
        if 'visibility_scores' in sim_detections and 'visibility_scores' in real_detections:
            sim_vis = np.array(sim_detections['visibility_scores'])
            real_vis = np.array(real_detections['visibility_scores'])
            
            if len(sim_vis) > 0 and len(real_vis) > 0:
                bins = np.linspace(0, 1, 11)
                sim_vis_hist, _ = np.histogram(sim_vis, bins=bins, density=True)
                real_vis_hist, _ = np.histogram(real_vis, bins=bins, density=True)
                
                sim_vis_dist = (sim_vis_hist + 1e-10) / (sim_vis_hist.sum() + 1e-10 * len(sim_vis_hist))
                real_vis_dist = (real_vis_hist + 1e-10) / (real_vis_hist.sum() + 1e-10 * len(real_vis_hist))
                
                kl_div = float(np.sum(real_vis_dist * np.log((real_vis_dist + 1e-10) / (sim_vis_dist + 1e-10))))
                results['visibility_kl_divergence'] = kl_div
                
                # Correlation between sim and real visibility
                # (if we can match objects)
                if len(sim_vis) == len(real_vis):
                    correlation = np.corrcoef(sim_vis, real_vis)[0, 1]
                    results['visibility_correlation'] = float(correlation)
    
    # 4. Range-Visibility Correlation
    if 'range_visibility' in metrics:
        if 'detection_ranges' in sim_detections and 'visibility_scores' in sim_detections:
            sim_ranges = np.array(sim_detections['detection_ranges'])
            sim_vis = np.array(sim_detections['visibility_scores'])
            
            if len(sim_ranges) > 0 and len(sim_vis) > 0 and len(sim_ranges) == len(sim_vis):
                # Visibility should decrease with distance
                corr_sim = np.corrcoef(sim_ranges, sim_vis)[0, 1]
                results['range_visibility_correlation_sim'] = float(corr_sim)
        
        if 'detection_ranges' in real_detections and 'visibility_scores' in real_detections:
            real_ranges = np.array(real_detections['detection_ranges'])
            real_vis = np.array(real_detections['visibility_scores'])
            
            if len(real_ranges) > 0 and len(real_vis) > 0 and len(real_ranges) == len(real_vis):
                corr_real = np.corrcoef(real_ranges, real_vis)[0, 1]
                results['range_visibility_correlation_real'] = float(corr_real)
    
    # 5. Occlusion by Distance
    if 'occlusion_by_distance' in metrics:
        if 'detection_ranges' in sim_detections and 'occlusion_levels' in sim_detections:
            sim_ranges = np.array(sim_detections['detection_ranges'])
            sim_occ = np.array(sim_detections['occlusion_levels'])
            
            if len(sim_ranges) > 0 and len(sim_occ) > 0 and len(sim_ranges) == len(sim_occ):
                # Near range: < 20m
                near_mask_sim = sim_ranges < 20
                if np.sum(near_mask_sim) > 0:
                    near_occ_sim = np.mean(sim_occ[near_mask_sim] > 0.3)  # Threshold for "occluded"
                    results['near_occlusion_ratio_sim'] = float(near_occ_sim)
                
                # Far range: > 50m
                far_mask_sim = sim_ranges > 50
                if np.sum(far_mask_sim) > 0:
                    far_occ_sim = np.mean(sim_occ[far_mask_sim] > 0.3)
                    results['far_occlusion_ratio_sim'] = float(far_occ_sim)
        
        if 'detection_ranges' in real_detections and 'occlusion_levels' in real_detections:
            real_ranges = np.array(real_detections['detection_ranges'])
            real_occ = np.array(real_detections['occlusion_levels'])
            
            if len(real_ranges) > 0 and len(real_occ) > 0 and len(real_ranges) == len(real_occ):
                near_mask_real = real_ranges < 20
                if np.sum(near_mask_real) > 0:
                    near_occ_real = np.mean(real_occ[near_mask_real] > 0.3)
                    results['near_occlusion_ratio_real'] = float(near_occ_real)
                
                far_mask_real = real_ranges > 50
                if np.sum(far_mask_real) > 0:
                    far_occ_real = np.mean(real_occ[far_mask_real] > 0.3)
                    results['far_occlusion_ratio_real'] = float(far_occ_real)
    
    # 6. Overall Occlusion Quality Score
    score_components = []
    
    # Occlusion distribution (lower KL = better)
    if 'occlusion_kl_divergence' in results:
        # KL < 0.2 = excellent, 0.2-0.5 = good, > 0.8 = poor
        occ_score = 100 * (1 - np.clip(results['occlusion_kl_divergence'] / 0.8, 0, 1))
        score_components.append(occ_score)
    
    # Truncation distribution (lower KL = better)
    if 'truncation_kl_divergence' in results:
        trunc_score = 100 * (1 - np.clip(results['truncation_kl_divergence'] / 0.8, 0, 1))
        score_components.append(trunc_score)
    
    # Visibility distribution (lower KL = better)
    if 'visibility_kl_divergence' in results:
        vis_score = 100 * (1 - np.clip(results['visibility_kl_divergence'] / 0.8, 0, 1))
        score_components.append(vis_score)
    
    # Range-visibility correlation (should be negative and similar)
    if 'range_visibility_correlation_sim' in results and 'range_visibility_correlation_real' in results:
        corr_error = abs(results['range_visibility_correlation_sim'] - results['range_visibility_correlation_real'])
        corr_score = 100 * (1 - np.clip(corr_error / 0.5, 0, 1))
        score_components.append(corr_score)
    
    # Partial occlusion frequency (should be similar)
    if 'partial_occlusion_frequency_sim' in results and 'partial_occlusion_frequency_real' in results:
        partial_error = abs(results['partial_occlusion_frequency_sim'] - results['partial_occlusion_frequency_real'])
        partial_score = 100 * (1 - np.clip(partial_error / 0.3, 0, 1))
        score_components.append(partial_score)
    
    if score_components:
        results['overall_occlusion_quality_score'] = float(np.mean(score_components))
    
    return results
