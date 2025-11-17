"""
Tests for occlusion and visibility quality metrics.
"""

import pytest
import numpy as np
from admetrics.simulation.occlusion_metrics import calculate_occlusion_visibility_quality


def test_identical_occlusion_distributions():
    """Test that identical occlusion distributions give low KL divergence."""
    np.random.seed(42)
    
    # Create identical distributions
    occlusion_levels = np.random.beta(2, 5, 100)  # Skewed toward low occlusion
    
    sim_detections = {
        'occlusion_levels': occlusion_levels.copy(),
    }
    real_detections = {
        'occlusion_levels': occlusion_levels.copy(),
    }
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections, 
                                                     ['occlusion_distribution'])
    
    assert 'occlusion_kl_divergence' in results
    assert results['occlusion_kl_divergence'] < 0.1  # Should be very low
    assert results['occlusion_mean_error'] < 0.01


def test_different_occlusion_distributions():
    """Test that different occlusion distributions give high KL divergence."""
    np.random.seed(42)
    
    # Simulation: mostly low occlusion
    sim_occ = np.random.beta(2, 5, 100)
    
    # Real: mostly high occlusion
    real_occ = np.random.beta(5, 2, 100)
    
    sim_detections = {'occlusion_levels': sim_occ}
    real_detections = {'occlusion_levels': real_occ}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_distribution'])
    
    assert results['occlusion_kl_divergence'] > 0.3
    assert results['occlusion_mean_error'] > 0.1


def test_partial_occlusion_frequency():
    """Test partial occlusion frequency calculation."""
    # Simulation: 50% partial occlusion (0.1 < occ < 0.9)
    sim_occ = np.array([0.05, 0.2, 0.5, 0.8, 0.95] * 20)
    
    # Real: similar pattern
    real_occ = np.array([0.1, 0.25, 0.45, 0.75, 0.9] * 20)
    
    sim_detections = {'occlusion_levels': sim_occ}
    real_detections = {'occlusion_levels': real_occ}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_distribution'])
    
    assert 'partial_occlusion_frequency_sim' in results
    assert 'partial_occlusion_frequency_real' in results
    assert 0.5 < results['partial_occlusion_frequency_sim'] < 0.7
    assert 0.5 < results['partial_occlusion_frequency_real'] < 0.7


def test_full_occlusion_frequency():
    """Test full occlusion frequency calculation."""
    # Many fully occluded objects (>= 0.9)
    sim_occ = np.array([0.9, 0.95, 1.0, 0.5, 0.2] * 20)
    real_occ = np.array([0.92, 0.96, 0.99, 0.6, 0.3] * 20)
    
    sim_detections = {'occlusion_levels': sim_occ}
    real_detections = {'occlusion_levels': real_occ}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_distribution'])
    
    assert 'full_occlusion_frequency_sim' in results
    assert 'full_occlusion_frequency_real' in results
    # 3 out of 5 = 60% fully occluded
    assert 0.5 < results['full_occlusion_frequency_sim'] < 0.7
    assert 0.5 < results['full_occlusion_frequency_real'] < 0.7


def test_truncation_distribution():
    """Test truncation level distribution matching."""
    np.random.seed(42)
    
    # Similar truncation patterns
    sim_trunc = np.random.beta(3, 7, 100)
    real_trunc = np.random.beta(3, 7, 100)
    
    sim_detections = {'truncation_levels': sim_trunc}
    real_detections = {'truncation_levels': real_trunc}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['truncation_distribution'])
    
    assert 'truncation_kl_divergence' in results
    assert results['truncation_kl_divergence'] < 0.5  # Relaxed threshold
    assert 'truncation_mean_error' in results


def test_visibility_distribution():
    """Test visibility score distribution matching."""
    np.random.seed(42)
    
    # High visibility scores
    sim_vis = np.random.beta(5, 2, 100)
    real_vis = np.random.beta(5, 2, 100)
    
    sim_detections = {'visibility_scores': sim_vis}
    real_detections = {'visibility_scores': real_vis}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['visibility_distribution'])
    
    assert 'visibility_kl_divergence' in results
    assert results['visibility_kl_divergence'] < 0.5  # Relaxed threshold


def test_visibility_correlation():
    """Test correlation between sim and real visibility (when matched)."""
    np.random.seed(42)
    
    # Highly correlated visibility scores
    base_vis = np.random.uniform(0.3, 0.9, 50)
    sim_vis = base_vis + np.random.normal(0, 0.05, 50)
    real_vis = base_vis + np.random.normal(0, 0.05, 50)
    
    sim_detections = {'visibility_scores': np.clip(sim_vis, 0, 1)}
    real_detections = {'visibility_scores': np.clip(real_vis, 0, 1)}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['visibility_distribution'])
    
    assert 'visibility_correlation' in results
    assert results['visibility_correlation'] > 0.8  # High correlation


def test_range_visibility_correlation():
    """Test that visibility decreases with range."""
    np.random.seed(42)
    
    # Simulate realistic range-visibility relationship
    sim_ranges = np.random.uniform(5, 100, 100)
    # Visibility decreases with distance
    sim_vis = 1.0 - (sim_ranges / 100) + np.random.normal(0, 0.1, 100)
    sim_vis = np.clip(sim_vis, 0.1, 1.0)
    
    real_ranges = np.random.uniform(5, 100, 100)
    real_vis = 1.0 - (real_ranges / 100) + np.random.normal(0, 0.1, 100)
    real_vis = np.clip(real_vis, 0.1, 1.0)
    
    sim_detections = {'detection_ranges': sim_ranges, 'visibility_scores': sim_vis}
    real_detections = {'detection_ranges': real_ranges, 'visibility_scores': real_vis}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['range_visibility'])
    
    assert 'range_visibility_correlation_sim' in results
    assert 'range_visibility_correlation_real' in results
    # Both should be negative (visibility decreases with range)
    assert results['range_visibility_correlation_sim'] < -0.5
    assert results['range_visibility_correlation_real'] < -0.5


def test_occlusion_by_distance():
    """Test occlusion patterns at different ranges."""
    np.random.seed(42)
    
    # Near objects: less occlusion
    near_ranges = np.random.uniform(5, 20, 50)
    near_occ = np.random.beta(2, 8, 50)  # Mostly low occlusion
    
    # Far objects: more occlusion
    far_ranges = np.random.uniform(50, 100, 50)
    far_occ = np.random.beta(5, 3, 50)  # Mostly high occlusion
    
    sim_ranges = np.concatenate([near_ranges, far_ranges])
    sim_occ = np.concatenate([near_occ, far_occ])
    
    # Similar pattern for real data
    real_near_ranges = np.random.uniform(5, 20, 50)
    real_near_occ = np.random.beta(2, 8, 50)
    real_far_ranges = np.random.uniform(50, 100, 50)
    real_far_occ = np.random.beta(5, 3, 50)
    
    real_ranges = np.concatenate([real_near_ranges, real_far_ranges])
    real_occ = np.concatenate([real_near_occ, real_far_occ])
    
    sim_detections = {'detection_ranges': sim_ranges, 'occlusion_levels': sim_occ}
    real_detections = {'detection_ranges': real_ranges, 'occlusion_levels': real_occ}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_by_distance'])
    
    assert 'near_occlusion_ratio_sim' in results
    assert 'far_occlusion_ratio_sim' in results
    assert 'near_occlusion_ratio_real' in results
    assert 'far_occlusion_ratio_real' in results
    
    # Far range should have more occlusion
    assert results['far_occlusion_ratio_sim'] > results['near_occlusion_ratio_sim']
    assert results['far_occlusion_ratio_real'] > results['near_occlusion_ratio_real']


def test_overall_quality_score_high():
    """Test overall quality score for well-matched distributions."""
    np.random.seed(42)
    
    # Create similar distributions
    sim_occ = np.random.beta(3, 5, 100)
    real_occ = np.random.beta(3, 5, 100)
    
    sim_trunc = np.random.beta(2, 6, 100)
    real_trunc = np.random.beta(2, 6, 100)
    
    sim_vis = np.random.beta(6, 2, 100)
    real_vis = np.random.beta(6, 2, 100)
    
    sim_ranges = np.random.uniform(10, 80, 100)
    real_ranges = np.random.uniform(10, 80, 100)
    
    sim_detections = {
        'occlusion_levels': sim_occ,
        'truncation_levels': sim_trunc,
        'visibility_scores': sim_vis,
        'detection_ranges': sim_ranges,
    }
    real_detections = {
        'occlusion_levels': real_occ,
        'truncation_levels': real_trunc,
        'visibility_scores': real_vis,
        'detection_ranges': real_ranges,
    }
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections)
    
    assert 'overall_occlusion_quality_score' in results
    assert results['overall_occlusion_quality_score'] > 60  # Relaxed threshold


def test_overall_quality_score_low():
    """Test overall quality score for poorly-matched distributions."""
    np.random.seed(42)
    
    # Create very different distributions
    sim_occ = np.random.beta(2, 8, 100)  # Low occlusion
    real_occ = np.random.beta(8, 2, 100)  # High occlusion
    
    sim_trunc = np.random.beta(2, 8, 100)
    real_trunc = np.random.beta(8, 2, 100)
    
    sim_vis = np.random.beta(8, 2, 100)  # High visibility
    real_vis = np.random.beta(2, 8, 100)  # Low visibility
    
    sim_ranges = np.random.uniform(10, 50, 100)
    real_ranges = np.random.uniform(10, 50, 100)
    
    sim_detections = {
        'occlusion_levels': sim_occ,
        'truncation_levels': sim_trunc,
        'visibility_scores': sim_vis,
        'detection_ranges': sim_ranges,
    }
    real_detections = {
        'occlusion_levels': real_occ,
        'truncation_levels': real_trunc,
        'visibility_scores': real_vis,
        'detection_ranges': real_ranges,
    }
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections)
    
    assert results['overall_occlusion_quality_score'] < 60


def test_empty_data():
    """Test handling of empty detection data."""
    sim_detections = {'occlusion_levels': np.array([])}
    real_detections = {'occlusion_levels': np.array([])}
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_distribution'])
    
    # Should return empty results without crashing
    assert isinstance(results, dict)


def test_missing_fields():
    """Test handling of missing fields."""
    sim_detections = {'occlusion_levels': np.random.rand(50)}
    real_detections = {}  # Missing occlusion_levels
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections,
                                                     ['occlusion_distribution'])
    
    # Should handle gracefully
    assert isinstance(results, dict)


def test_all_metrics():
    """Test computing all metrics at once."""
    np.random.seed(42)
    
    # Full dataset
    sim_detections = {
        'occlusion_levels': np.random.beta(3, 5, 100),
        'truncation_levels': np.random.beta(2, 6, 100),
        'visibility_scores': np.random.beta(6, 2, 100),
        'detection_ranges': np.random.uniform(5, 100, 100),
    }
    real_detections = {
        'occlusion_levels': np.random.beta(3, 5, 100),
        'truncation_levels': np.random.beta(2, 6, 100),
        'visibility_scores': np.random.beta(6, 2, 100),
        'detection_ranges': np.random.uniform(5, 100, 100),
    }
    
    results = calculate_occlusion_visibility_quality(sim_detections, real_detections, 'all')
    
    # Should have many metrics
    assert 'occlusion_kl_divergence' in results
    assert 'truncation_kl_divergence' in results
    assert 'visibility_kl_divergence' in results
    assert 'overall_occlusion_quality_score' in results
