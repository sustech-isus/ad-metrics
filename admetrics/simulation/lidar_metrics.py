"""
LiDAR Point Cloud Quality Metrics for Simulation Validation.

Evaluates the geometric fidelity and accuracy of simulated LiDAR point clouds
compared to real-world LiDAR data.
"""

import numpy as np
from typing import Dict
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_lidar_point_cloud_quality(
    sim_points: np.ndarray,
    real_points: np.ndarray,
    max_range: float = 100.0
) -> Dict[str, float]:
    """
    Evaluate LiDAR point cloud simulation quality.
    
    Compares simulated LiDAR scans to real-world scans to assess geometric fidelity.
    
    Args:
        sim_points: Simulated point cloud, shape (N, 3) or (N, 4) for [x, y, z] or [x, y, z, intensity]
        real_points: Real-world point cloud, shape (M, 3) or (M, 4)
        max_range: Maximum sensor range in meters
        
    Returns:
        Dictionary with quality metrics:
            - chamfer_distance: Average nearest-neighbor distance (bidirectional)
            - point_density: Ratio of simulated to real point counts
            - range_distribution: KL divergence of range histograms
            - intensity_correlation: Correlation of intensity values (if available)
            - vertical_distribution: Distribution of vertical angles
            
    Example:
        >>> sim_pc = np.random.randn(1000, 3) * 10
        >>> real_pc = sim_pc + np.random.randn(*sim_pc.shape) * 0.5
        >>> quality = calculate_lidar_point_cloud_quality(sim_pc, real_pc)
    """
    results = {}
    
    # Extract XYZ coordinates
    sim_xyz = sim_points[:, :3]
    real_xyz = real_points[:, :3]
    
    # Chamfer Distance (symmetric)
    # Subsample for efficiency if too many points
    if len(sim_xyz) > 10000:
        sim_sample = sim_xyz[np.random.choice(len(sim_xyz), 10000, replace=False)]
    else:
        sim_sample = sim_xyz
        
    if len(real_xyz) > 10000:
        real_sample = real_xyz[np.random.choice(len(real_xyz), 10000, replace=False)]
    else:
        real_sample = real_xyz
    
    # Sim to real distances
    dist_sim_to_real = cdist(sim_sample, real_sample)
    min_dist_sim = np.min(dist_sim_to_real, axis=1)
    chamfer_sim_to_real = np.mean(min_dist_sim)
    
    # Real to sim distances
    dist_real_to_sim = cdist(real_sample, sim_sample)
    min_dist_real = np.min(dist_real_to_sim, axis=1)
    chamfer_real_to_sim = np.mean(min_dist_real)
    
    results['chamfer_distance'] = float((chamfer_sim_to_real + chamfer_real_to_sim) / 2)
    
    # Point density comparison
    results['point_density_ratio'] = float(len(sim_xyz) / (len(real_xyz) + 1e-6))
    results['point_count_sim'] = len(sim_xyz)
    results['point_count_real'] = len(real_xyz)
    
    # Range distribution (distance from origin)
    sim_ranges = np.linalg.norm(sim_xyz, axis=1)
    real_ranges = np.linalg.norm(real_xyz, axis=1)
    
    bins = np.linspace(0, max_range, 50)
    sim_range_hist, _ = np.histogram(sim_ranges, bins=bins, density=True)
    real_range_hist, _ = np.histogram(real_ranges, bins=bins, density=True)
    
    # Add epsilon and normalize
    sim_range_hist = sim_range_hist + 1e-10
    real_range_hist = real_range_hist + 1e-10
    sim_range_hist = sim_range_hist / np.sum(sim_range_hist)
    real_range_hist = real_range_hist / np.sum(real_range_hist)
    
    results['range_distribution_kl'] = float(stats.entropy(sim_range_hist, real_range_hist))
    
    # Intensity correlation (if available)
    if sim_points.shape[1] >= 4 and real_points.shape[1] >= 4:
        # Match points for intensity comparison
        if len(sim_sample) <= len(real_sample):
            dist_matrix = cdist(sim_sample, real_sample)
            matched_real_idx = np.argmin(dist_matrix, axis=1)
            
            sim_intensities = sim_points[np.random.choice(len(sim_xyz), len(sim_sample), replace=False), 3]
            real_intensities = real_points[matched_real_idx, 3]
            
            if len(sim_intensities) > 0 and len(real_intensities) > 0:
                correlation = np.corrcoef(sim_intensities, real_intensities)[0, 1]
                results['intensity_correlation'] = float(correlation)
    
    # Vertical angle distribution
    sim_vertical_angles = np.arctan2(sim_xyz[:, 2], np.linalg.norm(sim_xyz[:, :2], axis=1))
    real_vertical_angles = np.arctan2(real_xyz[:, 2], np.linalg.norm(real_xyz[:, :2], axis=1))
    
    angle_bins = np.linspace(-np.pi/2, np.pi/2, 30)
    sim_angle_hist, _ = np.histogram(sim_vertical_angles, bins=angle_bins, density=True)
    real_angle_hist, _ = np.histogram(real_vertical_angles, bins=angle_bins, density=True)
    
    sim_angle_hist = sim_angle_hist + 1e-10
    real_angle_hist = real_angle_hist + 1e-10
    sim_angle_hist = sim_angle_hist / np.sum(sim_angle_hist)
    real_angle_hist = real_angle_hist / np.sum(real_angle_hist)
    
    results['vertical_angle_kl'] = float(stats.entropy(sim_angle_hist, real_angle_hist))
    
    return results
