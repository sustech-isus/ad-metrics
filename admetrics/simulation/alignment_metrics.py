"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_multimodal_sensor_alignment(
    camera_detections: np.ndarray,
    lidar_detections: np.ndarray,
    camera_to_lidar_transform: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate alignment quality between different sensor modalities.
    
    Critical for sensor fusion validation in simulation.
    
    Args:
        camera_detections: 3D bounding boxes from camera, shape (N, 7) [x, y, z, l, w, h, yaw]
        lidar_detections: 3D bounding boxes from LiDAR, shape (M, 7)
        camera_to_lidar_transform: Optional 4x4 transformation matrix
        
    Returns:
        Dictionary with alignment metrics:
            - spatial_alignment_error: Mean position error for matched detections
            - detection_agreement_rate: Fraction of detections that match
            - size_consistency: Mean size difference for matched detections
            - temporal_sync_error: Timestamp synchronization error (if available)
            
    Example:
        >>> cam_det = np.random.randn(10, 7)
        >>> lidar_det = cam_det + np.random.randn(*cam_det.shape) * 0.2
        >>> quality = multimodal_sensor_alignment(cam_det, lidar_det)
    """
    results = {}
    
    if len(camera_detections) == 0 or len(lidar_detections) == 0:
        results['detection_agreement_rate'] = 0.0
        return results
    
    # Apply transformation if provided
    if camera_to_lidar_transform is not None:
        # Transform camera detections to LiDAR frame
        cam_positions = camera_detections[:, :3]
        ones = np.ones((len(cam_positions), 1))
        cam_positions_h = np.hstack([cam_positions, ones])
        transformed_positions = (camera_to_lidar_transform @ cam_positions_h.T).T[:, :3]
        camera_detections = camera_detections.copy()
        camera_detections[:, :3] = transformed_positions
    
    # Match detections using Hungarian algorithm (simplified: greedy matching)
    cam_centers = camera_detections[:, :3]
    lidar_centers = lidar_detections[:, :3]
    
    dist_matrix = cdist(cam_centers, lidar_centers)
    
    # Greedy matching with threshold
    matches = []
    match_threshold = 2.0  # meters
    
    used_lidar = set()
    for cam_idx in range(len(camera_detections)):
        distances = dist_matrix[cam_idx]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < match_threshold and min_idx not in used_lidar:
            matches.append((cam_idx, min_idx, distances[min_idx]))
            used_lidar.add(min_idx)
    
    # Detection agreement rate
    total_detections = max(len(camera_detections), len(lidar_detections))
    results['detection_agreement_rate'] = float(len(matches) / total_detections)
    results['matched_detections'] = len(matches)
    results['camera_only_detections'] = len(camera_detections) - len(matches)
    results['lidar_only_detections'] = len(lidar_detections) - len(matches)
    
    if len(matches) == 0:
        return results
    
    # Spatial alignment error
    spatial_errors = [dist for _, _, dist in matches]
    results['spatial_alignment_error'] = float(np.mean(spatial_errors))
    results['spatial_alignment_std'] = float(np.std(spatial_errors))
    
    # Size consistency (L, W, H)
    cam_sizes = camera_detections[[m[0] for m in matches], 3:6]
    lidar_sizes = lidar_detections[[m[1] for m in matches], 3:6]
    
    size_errors = np.abs(cam_sizes - lidar_sizes)
    results['size_consistency_error'] = float(np.mean(size_errors))
    results['length_error'] = float(np.mean(size_errors[:, 0]))
    results['width_error'] = float(np.mean(size_errors[:, 1]))
    results['height_error'] = float(np.mean(size_errors[:, 2]))
    
    # Orientation consistency
    cam_yaws = camera_detections[[m[0] for m in matches], 6]
    lidar_yaws = lidar_detections[[m[1] for m in matches], 6]
    
    yaw_errors = np.abs(np.arctan2(np.sin(cam_yaws - lidar_yaws), np.cos(cam_yaws - lidar_yaws)))
    results['orientation_error_rad'] = float(np.mean(yaw_errors))
    results['orientation_error_deg'] = float(np.degrees(np.mean(yaw_errors)))
    
    return results


