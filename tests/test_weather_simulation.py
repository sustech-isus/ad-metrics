"""
Tests for weather simulation quality metrics.

Validates weather and environmental simulation quality functions including rain,
fog, snow, lighting, and shadow realism.
"""

import numpy as np
import pytest
from admetrics.simulation import calculate_weather_simulation_quality


class TestWeatherIntensityDistribution:
    """Tests for weather intensity distribution validation."""
    
    def test_rain_intensity_similar_distributions(self):
        """Test rain intensity with similar sim and real distributions."""
        # Both follow gamma distribution (typical for rain)
        sim_data = {
            'intensity': np.random.gamma(2, 5, 1000),  # Rain rate mm/h
            'visibility': np.random.normal(200, 50, 1000)
        }
        real_data = {
            'intensity': np.random.gamma(2.1, 4.9, 1000),
            'visibility': np.random.normal(195, 52, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution']
        )
        
        assert 'intensity_kl_divergence' in result
        assert result['intensity_kl_divergence'] < 0.5  # Low divergence for similar distributions
        assert 'intensity_mean_sim' in result
        assert 'intensity_mean_real' in result
        assert 'intensity_std_ratio' in result
        assert 0.5 < result['intensity_std_ratio'] < 2.0
    
    def test_fog_density_validation(self):
        """Test fog density distribution comparison."""
        # Fog typically has different distribution than rain
        sim_data = {
            'intensity': np.random.exponential(2, 1000),  # Fog density
            'visibility': np.random.normal(300, 100, 1000)
        }
        real_data = {
            'intensity': np.random.exponential(2.2, 1000),
            'visibility': np.random.normal(290, 110, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='fog',
            metrics=['intensity_distribution']
        )
        
        assert 'intensity_kl_divergence' in result
        assert result['intensity_kl_divergence'] >= 0
        assert result['intensity_mean_sim'] > 0
        assert result['intensity_mean_real'] > 0
    
    def test_very_different_distributions(self):
        """Test with very different sim/real distributions."""
        sim_data = {
            'intensity': np.random.uniform(0, 10, 1000),
            'visibility': np.random.normal(500, 50, 1000)
        }
        real_data = {
            'intensity': np.random.gamma(5, 2, 1000),
            'visibility': np.random.normal(200, 100, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution']
        )
        
        # Should have high KL divergence for very different distributions
        assert result['intensity_kl_divergence'] > 0.1


class TestVisibilityRange:
    """Tests for visibility range validation."""
    
    def test_visibility_mean_error(self):
        """Test visibility mean absolute error calculation."""
        sim_data = {
            'intensity': np.random.gamma(2, 5, 500),
            'visibility': np.random.normal(250, 40, 500)
        }
        real_data = {
            'intensity': np.random.gamma(2, 5, 500),
            'visibility': np.random.normal(200, 45, 500)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='fog',
            metrics=['visibility_range']
        )
        
        assert 'visibility_mean_error' in result
        assert 'visibility_std_ratio' in result
        assert 'visibility_median_sim' in result
        assert 'visibility_median_real' in result
        assert 'visibility_ks_statistic' in result
        assert 'visibility_ks_pvalue' in result
        assert result['visibility_mean_error'] >= 0
        assert 0 <= result['visibility_ks_statistic'] <= 1
        assert 0 <= result['visibility_ks_pvalue'] <= 1
    
    def test_visibility_statistical_test(self):
        """Test KS statistical test for visibility distributions."""
        # Identical distributions should pass KS test
        visibility_data = np.random.normal(300, 50, 1000)
        sim_data = {'visibility': visibility_data + np.random.randn(1000) * 5}
        real_data = {'visibility': visibility_data + np.random.randn(1000) * 5}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='fog',
            metrics=['visibility_range']
        )
        
        # High p-value indicates similar distributions
        assert result['visibility_ks_pvalue'] > 0.01
    
    def test_heavy_fog_low_visibility(self):
        """Test validation with heavy fog (low visibility)."""
        sim_data = {
            'intensity': np.random.gamma(5, 3, 500),  # Dense fog
            'visibility': np.random.normal(150, 30, 500)  # Low visibility
        }
        real_data = {
            'intensity': np.random.gamma(5.2, 2.9, 500),
            'visibility': np.random.normal(145, 35, 500)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='fog',
            metrics=['visibility_range']
        )
        
        assert result['visibility_median_sim'] < 200  # Heavy fog
        assert result['visibility_median_real'] < 200
        assert result['visibility_mean_error'] < 50  # Good accuracy


class TestTemporalConsistency:
    """Tests for temporal consistency validation."""
    
    def test_temporal_stability(self):
        """Test temporal stability calculation."""
        # Create temporally smooth intensity data
        t = np.linspace(0, 10, 100)
        sim_data = {
            'intensity': 5 + 2 * np.sin(t) + np.random.randn(100) * 0.1
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, {}, weather_type='rain',
            metrics=['temporal_consistency']
        )
        
        assert 'temporal_stability' in result
        assert 'frame_to_frame_correlation' in result
        assert 0 <= result['temporal_stability'] <= 1
        assert -1 <= result['frame_to_frame_correlation'] <= 1
    
    def test_high_temporal_stability(self):
        """Test with high temporal stability (smooth transitions)."""
        t = np.linspace(0, 10, 200)
        sim_data = {
            'intensity': 10 + 3 * np.sin(t * 0.5)  # Smooth sinusoidal variation
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, {}, weather_type='rain',
            metrics=['temporal_consistency']
        )
        
        # Smooth data should have high stability
        assert result['temporal_stability'] > 0.8
        assert result['frame_to_frame_correlation'] > 0.9
    
    def test_low_temporal_stability(self):
        """Test with low temporal stability (noisy/jittery)."""
        sim_data = {
            'intensity': np.random.randn(100) * 10  # High noise, low correlation
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, {}, weather_type='rain',
            metrics=['temporal_consistency']
        )
        
        # Noisy data should have lower stability
        assert result['temporal_stability'] < 0.9


class TestSpatialDistribution:
    """Tests for spatial distribution validation."""
    
    def test_spatial_correlation(self):
        """Test spatial pattern correlation."""
        # Create images with similar spatial patterns
        sim_imgs = np.random.rand(10, 64, 64, 3) * 100
        real_imgs = sim_imgs + np.random.randn(10, 64, 64, 3) * 5
        
        sim_data = {'images': sim_imgs}
        real_data = {'images': real_imgs}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['spatial_distribution']
        )
        
        assert 'spatial_correlation' in result
        assert -1 <= result['spatial_correlation'] <= 1
    
    def test_similar_spatial_patterns(self):
        """Test with similar spatial gradient patterns."""
        # Create structured pattern
        x, y = np.meshgrid(np.linspace(0, 10, 64), np.linspace(0, 10, 64))
        pattern = np.sin(x) * np.cos(y)
        
        sim_imgs = np.stack([pattern] * 10)[:, :, :, np.newaxis]
        sim_imgs = np.repeat(sim_imgs, 3, axis=-1)
        real_imgs = sim_imgs + np.random.randn(*sim_imgs.shape) * 0.1
        
        sim_data = {'images': sim_imgs}
        real_data = {'images': real_imgs}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='fog',
            metrics=['spatial_distribution']
        )
        
        # Similar patterns should have high correlation
        assert result['spatial_correlation'] > 0.7


class TestParticleDensity:
    """Tests for rain/snow particle density validation."""
    
    def test_rain_particle_density(self):
        """Test rain particle density estimation."""
        # Create images with particle-like variance
        sim_imgs = np.random.rand(20, 64, 64, 3) * 150
        real_imgs = np.random.rand(20, 64, 64, 3) * 140
        
        sim_data = {'images': sim_imgs}
        real_data = {'images': real_imgs}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['particle_density']
        )
        
        assert 'particle_density_ratio' in result
        assert 'particle_density_sim' in result
        assert 'particle_density_real' in result
        assert result['particle_density_ratio'] > 0
    
    def test_snow_particle_density(self):
        """Test snow particle density estimation."""
        sim_imgs = np.random.rand(15, 64, 64, 3) * 200
        real_imgs = np.random.rand(15, 64, 64, 3) * 210
        
        sim_data = {'images': sim_imgs}
        real_data = {'images': real_imgs}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='snow',
            metrics=['particle_density']
        )
        
        assert 'particle_density_ratio' in result
        # Ratio should be close to 1 for similar densities
        assert 0.5 < result['particle_density_ratio'] < 2.0


class TestLightingHistogram:
    """Tests for lighting condition validation."""
    
    def test_daylight_distribution(self):
        """Test daylight intensity distribution."""
        # Daylight: high lighting values
        sim_data = {
            'lighting': np.random.beta(8, 2, 1000)  # Skewed towards high values
        }
        real_data = {
            'lighting': np.random.beta(7.5, 2.2, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='lighting',
            metrics=['lighting_histogram']
        )
        
        assert 'lighting_kl_divergence' in result
        assert 'lighting_mean_sim' in result
        assert 'lighting_mean_real' in result
        assert result['lighting_mean_sim'] > 0.6  # Daylight
        assert result['lighting_mean_real'] > 0.6
    
    def test_night_lighting_distribution(self):
        """Test night lighting distribution."""
        # Night: low lighting values
        sim_data = {
            'lighting': np.random.beta(2, 8, 1000)  # Skewed towards low values
        }
        real_data = {
            'lighting': np.random.beta(2.2, 7.8, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='lighting',
            metrics=['lighting_histogram']
        )
        
        assert result['lighting_mean_sim'] < 0.4  # Night
        assert result['lighting_mean_real'] < 0.4
        assert result['lighting_kl_divergence'] < 0.5  # Similar distributions
    
    def test_dusk_dawn_transitions(self):
        """Test dusk/dawn lighting transitions."""
        # Dusk/dawn: uniform-ish distribution
        sim_data = {
            'lighting': np.random.beta(3, 3, 1000)  # More uniform
        }
        real_data = {
            'lighting': np.random.beta(3.2, 2.9, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='lighting',
            metrics=['lighting_histogram']
        )
        
        assert 0.3 < result['lighting_mean_sim'] < 0.7  # Mid-range
        assert 0.3 < result['lighting_mean_real'] < 0.7


class TestShadowRealism:
    """Tests for shadow realism validation."""
    
    def test_shadow_coverage(self):
        """Test shadow coverage validation."""
        sim_data = {
            'shadow_coverage': np.random.beta(3, 7, 500),  # ~30% coverage
            'images': np.random.rand(10, 64, 64, 3) * 200
        }
        real_data = {
            'shadow_coverage': np.random.beta(3.2, 6.8, 500),
            'images': np.random.rand(10, 64, 64, 3) * 190
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='shadows',
            metrics=['shadow_realism']
        )
        
        assert 'shadow_coverage_error' in result
        assert 'shadow_coverage_sim' in result
        assert 'shadow_coverage_real' in result
        assert 'shadow_edge_sharpness' in result
        assert result['shadow_coverage_error'] >= 0
        assert 0 <= result['shadow_coverage_sim'] <= 1
        assert 0 <= result['shadow_coverage_real'] <= 1
    
    def test_shadow_edge_sharpness(self):
        """Test shadow edge sharpness calculation."""
        # Create images with sharp edges (shadows)
        sim_imgs = np.ones((10, 64, 64, 3)) * 100
        sim_imgs[:, :, :32, :] = 50  # Sharp shadow edge
        
        real_imgs = np.ones((10, 64, 64, 3)) * 100
        real_imgs[:, :, :32, :] = 55  # Similar sharp edge
        
        sim_data = {
            'shadow_coverage': np.array([0.5] * 500),
            'images': sim_imgs
        }
        real_data = {
            'shadow_coverage': np.array([0.5] * 500),
            'images': real_imgs
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='shadows',
            metrics=['shadow_realism']
        )
        
        assert result['shadow_edge_sharpness'] > 0
        # Similar edge sharpness
        assert 0.5 < result['shadow_edge_sharpness'] < 2.0


class TestOverallRealismScore:
    """Tests for overall realism score calculation."""
    
    def test_overall_score_with_multiple_metrics(self):
        """Test overall realism score combining multiple metrics."""
        sim_data = {
            'intensity': np.random.gamma(2, 5, 1000),
            'visibility': np.random.normal(250, 50, 1000),
            'images': np.random.rand(10, 64, 64, 3) * 200
        }
        real_data = {
            'intensity': np.random.gamma(2.1, 4.9, 1000),
            'visibility': np.random.normal(240, 52, 1000),
            'images': np.random.rand(10, 64, 64, 3) * 190
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution', 'visibility_range', 
                    'temporal_consistency', 'spatial_distribution']
        )
        
        assert 'overall_realism_score' in result
        assert 0 <= result['overall_realism_score'] <= 100
    
    def test_high_realism_score(self):
        """Test high realism score with very similar data."""
        # Nearly identical data should give high score
        base_intensity = np.random.gamma(3, 4, 1000)
        sim_data = {
            'intensity': base_intensity + np.random.randn(1000) * 0.1,
            'visibility': np.random.normal(300, 50, 1000)
        }
        real_data = {
            'intensity': base_intensity + np.random.randn(1000) * 0.1,
            'visibility': np.random.normal(302, 51, 1000)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution', 'visibility_range']
        )
        
        # Very similar data should have high realism score
        assert result['overall_realism_score'] > 70


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_metrics_list(self):
        """Test with empty metrics list."""
        sim_data = {'intensity': np.random.gamma(2, 5, 100)}
        real_data = {'intensity': np.random.gamma(2, 5, 100)}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain', metrics=[]
        )
        
        # Should still return default metrics
        assert len(result) > 0
    
    def test_missing_data_fields(self):
        """Test with missing data fields."""
        sim_data = {'intensity': np.random.gamma(2, 5, 100)}
        real_data = {'intensity': np.random.gamma(2, 5, 100)}
        
        # Request metric that requires missing field
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['visibility_range']  # 'visibility' field missing
        )
        
        # Should not crash, just skip unavailable metrics
        assert 'visibility_mean_error' not in result
    
    def test_single_sample(self):
        """Test with single sample (edge case)."""
        sim_data = {'intensity': np.array([5.0])}
        real_data = {'intensity': np.array([5.5])}
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution']
        )
        
        # Should handle gracefully
        assert 'intensity_mean_sim' in result
        assert 'intensity_mean_real' in result
    
    def test_zero_intensity_values(self):
        """Test with zero intensity values (no weather)."""
        sim_data = {
            'intensity': np.zeros(100),
            'visibility': np.random.normal(1000, 50, 100)  # Clear weather
        }
        real_data = {
            'intensity': np.zeros(100),
            'visibility': np.random.normal(1000, 50, 100)
        }
        
        result = calculate_weather_simulation_quality(
            sim_data, real_data, weather_type='rain',
            metrics=['intensity_distribution', 'visibility_range']
        )
        
        # Should handle zero values gracefully
        assert 'visibility_mean_error' in result
    
    def test_different_weather_types(self):
        """Test all supported weather types."""
        sim_data = {
            'intensity': np.random.gamma(2, 5, 500),
            'visibility': np.random.normal(300, 50, 500)
        }
        real_data = {
            'intensity': np.random.gamma(2, 5, 500),
            'visibility': np.random.normal(300, 50, 500)
        }
        
        weather_types = ['rain', 'fog', 'snow', 'lighting', 'shadows']
        
        for weather_type in weather_types:
            result = calculate_weather_simulation_quality(
                sim_data, real_data, weather_type=weather_type
            )
            assert len(result) > 0
