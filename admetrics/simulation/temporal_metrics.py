"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_temporal_consistency(
    detections_sequence: List[np.ndarray],
    fps: float = 10.0
) -> Dict[str, float]:
    """
    Evaluate temporal consistency of sensor detections.
    
    Simulated sensors should maintain realistic temporal coherence.
    
    Args:
        detections_sequence: List of detection arrays over time, each shape (N_t, D)
        fps: Frames per second of sensor
        
    Returns:
        Dictionary with temporal metrics:
            - detection_count_variance: Variance in number of detections
            - motion_smoothness: Smoothness of tracked object motion
            - appearance_disappearance_rate: Rate of flickering detections
            - frame_to_frame_consistency: Average detection overlap between frames
            
    Example:
        >>> sequence = [np.random.randn(10, 7) for _ in range(20)]
        >>> quality = temporal_consistency(sequence)
    """
    results = {}
    
    if len(detections_sequence) < 2:
        return results
    
    # Detection count variance
    detection_counts = [len(dets) for dets in detections_sequence]
    results['detection_count_mean'] = float(np.mean(detection_counts))
    results['detection_count_std'] = float(np.std(detection_counts))
    results['detection_count_variance'] = float(np.var(detection_counts))
    
    # Frame-to-frame consistency (IoU-based matching rate)
    frame_consistencies = []
    for i in range(len(detections_sequence) - 1):
        curr_dets = detections_sequence[i]
        next_dets = detections_sequence[i + 1]
        
        if len(curr_dets) == 0 or len(next_dets) == 0:
            continue
        
        # Simple distance-based matching
        curr_centers = curr_dets[:, :3] if curr_dets.shape[1] >= 3 else curr_dets
        next_centers = next_dets[:, :3] if next_dets.shape[1] >= 3 else next_dets
        
        dist_matrix = cdist(curr_centers, next_centers)
        
        # Count matches within threshold (accounting for motion at fps)
        max_motion = 20.0 / fps  # Assume max 20 m/s motion
        matches = np.sum(np.min(dist_matrix, axis=1) < max_motion)
        
        consistency = matches / max(len(curr_dets), len(next_dets))
        frame_consistencies.append(consistency)
    
    if len(frame_consistencies) > 0:
        results['frame_to_frame_consistency'] = float(np.mean(frame_consistencies))
        results['consistency_std'] = float(np.std(frame_consistencies))
    
    # Appearance/disappearance rate (flickering)
    # Count new and lost detections per frame
    new_detections = []
    lost_detections = []
    
    for i in range(len(detections_sequence) - 1):
        curr_count = len(detections_sequence[i])
        next_count = len(detections_sequence[i + 1])
        
        # Simplified: just count changes
        if next_count > curr_count:
            new_detections.append(next_count - curr_count)
        elif curr_count > next_count:
            lost_detections.append(curr_count - next_count)
    
    if len(new_detections) > 0:
        results['avg_new_detections_per_frame'] = float(np.mean(new_detections))
    if len(lost_detections) > 0:
        results['avg_lost_detections_per_frame'] = float(np.mean(lost_detections))
    
    flicker_rate = (len(new_detections) + len(lost_detections)) / (len(detections_sequence) - 1)
    results['flicker_rate'] = float(flicker_rate)
    
    return results


