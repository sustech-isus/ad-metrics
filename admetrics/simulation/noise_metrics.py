"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_sensor_noise_characteristics(
    sim_measurements: np.ndarray,
    real_measurements: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compare noise characteristics between simulated and real sensors.
    
    Evaluates whether simulated sensor noise matches real-world noise patterns.
    
    Args:
        sim_measurements: Repeated measurements from simulation, shape (N, D)
        real_measurements: Repeated measurements from real sensor, shape (M, D)
        ground_truth: Optional ground truth values, shape (D,)
        
    Returns:
        Dictionary with noise statistics:
            - noise_std_ratio: Ratio of standard deviations
            - noise_distribution_ks: Kolmogorov-Smirnov test statistic
            - bias_sim: Mean error from ground truth (if provided)
            - bias_real: Mean error from ground truth (if provided)
            - snr_ratio: Signal-to-noise ratio comparison
            
    Example:
        >>> # Repeated measurements of same target
        >>> sim_meas = np.random.randn(100, 3) * 0.1 + np.array([10, 0, 0])
        >>> real_meas = np.random.randn(100, 3) * 0.15 + np.array([10, 0, 0])
        >>> gt = np.array([10, 0, 0])
        >>> quality = sensor_noise_characteristics(sim_meas, real_meas, gt)
    """
    results = {}
    
    # Standard deviation comparison
    sim_std = np.std(sim_measurements, axis=0)
    real_std = np.std(real_measurements, axis=0)
    
    results['noise_std_sim'] = float(np.mean(sim_std))
    results['noise_std_real'] = float(np.mean(real_std))
    results['noise_std_ratio'] = float(np.mean(sim_std) / (np.mean(real_std) + 1e-6))
    
    # Distribution comparison (Kolmogorov-Smirnov test)
    # Flatten and compare overall distributions
    sim_flat = sim_measurements.flatten()
    real_flat = real_measurements.flatten()
    
    ks_statistic, ks_pvalue = stats.ks_2samp(sim_flat, real_flat)
    results['noise_distribution_ks_statistic'] = float(ks_statistic)
    results['noise_distribution_ks_pvalue'] = float(ks_pvalue)
    
    # Bias analysis (if ground truth available)
    if ground_truth is not None:
        sim_errors = sim_measurements - ground_truth
        real_errors = real_measurements - ground_truth
        
        results['bias_sim'] = float(np.mean(np.linalg.norm(sim_errors, axis=1)))
        results['bias_real'] = float(np.mean(np.linalg.norm(real_errors, axis=1)))
        results['bias_ratio'] = float(results['bias_sim'] / (results['bias_real'] + 1e-6))
        
        # Signal-to-noise ratio
        signal_power_sim = np.mean(ground_truth ** 2)
        noise_power_sim = np.mean(sim_errors ** 2)
        snr_sim = 10 * np.log10(signal_power_sim / (noise_power_sim + 1e-10))
        
        noise_power_real = np.mean(real_errors ** 2)
        snr_real = 10 * np.log10(signal_power_sim / (noise_power_real + 1e-10))
        
        results['snr_sim_db'] = float(snr_sim)
        results['snr_real_db'] = float(snr_real)
        results['snr_ratio'] = float(snr_sim / (snr_real + 1e-6))
    
    return results


