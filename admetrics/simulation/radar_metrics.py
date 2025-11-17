"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_radar_quality(
    sim_detections: np.ndarray,
    real_detections: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate radar simulation quality.
    
    Compares simulated radar detections to real radar data.
    
    Args:
        sim_detections: Simulated radar detections, shape (N, D) where D includes:
            [x, y, z, velocity, rcs] - position, radial velocity, radar cross-section
        real_detections: Real radar detections, same format
        
    Returns:
        Dictionary with quality metrics:
            - detection_density: Ratio of detection counts
            - velocity_distribution: KL divergence of velocity distributions
            - rcs_distribution: Radar cross-section distribution similarity
            - spatial_accuracy: Chamfer distance of detection positions
            
    Example:
        >>> sim_radar = np.random.randn(50, 5)
        >>> real_radar = np.random.randn(60, 5)
        >>> quality = radar_quality(sim_radar, real_radar)
    """
    results = {}
    
    # Detection density
    results['detection_density_ratio'] = float(len(sim_detections) / (len(real_detections) + 1e-6))
    results['detection_count_sim'] = len(sim_detections)
    results['detection_count_real'] = len(real_detections)
    
    if len(sim_detections) == 0 or len(real_detections) == 0:
        return results
    
    # Spatial accuracy (Chamfer distance on positions)
    sim_positions = sim_detections[:, :3]
    real_positions = real_detections[:, :3]
    
    dist_matrix = cdist(sim_positions, real_positions)
    chamfer_sim = np.mean(np.min(dist_matrix, axis=1))
    chamfer_real = np.mean(np.min(dist_matrix, axis=0))
    results['spatial_chamfer'] = float((chamfer_sim + chamfer_real) / 2)
    
    # Velocity distribution (if available)
    if sim_detections.shape[1] >= 4 and real_detections.shape[1] >= 4:
        sim_velocities = sim_detections[:, 3]
        real_velocities = real_detections[:, 3]
        
        bins = np.linspace(-30, 30, 40)  # m/s range
        sim_vel_hist, _ = np.histogram(sim_velocities, bins=bins, density=True)
        real_vel_hist, _ = np.histogram(real_velocities, bins=bins, density=True)
        
        sim_vel_hist = sim_vel_hist + 1e-10
        real_vel_hist = real_vel_hist + 1e-10
        sim_vel_hist = sim_vel_hist / np.sum(sim_vel_hist)
        real_vel_hist = real_vel_hist / np.sum(real_vel_hist)
        
        results['velocity_distribution_kl'] = float(stats.entropy(sim_vel_hist, real_vel_hist))
    
    # RCS distribution (if available)
    if sim_detections.shape[1] >= 5 and real_detections.shape[1] >= 5:
        sim_rcs = sim_detections[:, 4]
        real_rcs = real_detections[:, 4]
        
        # RCS in dBsm, typically ranges from -10 to 40
        bins = np.linspace(-10, 40, 30)
        sim_rcs_hist, _ = np.histogram(sim_rcs, bins=bins, density=True)
        real_rcs_hist, _ = np.histogram(real_rcs, bins=bins, density=True)
        
        sim_rcs_hist = sim_rcs_hist + 1e-10
        real_rcs_hist = real_rcs_hist + 1e-10
        sim_rcs_hist = sim_rcs_hist / np.sum(sim_rcs_hist)
        real_rcs_hist = real_rcs_hist / np.sum(real_rcs_hist)
        
        results['rcs_distribution_kl'] = float(stats.entropy(sim_rcs_hist, real_rcs_hist))
    
    return results


