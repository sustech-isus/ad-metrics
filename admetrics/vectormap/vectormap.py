"""
Vector Map Detection and Evaluation Metrics.

Metrics for evaluating HD map vector extraction from sensor data, including:
- Lane line detection (centerlines, boundaries)
- Road boundary detection
- Crosswalk/pedestrian crossing detection
- Topology estimation (connectivity, lane relationships)

Used in: nuScenes Map, Argoverse 2 Map, OpenLane-V2, MapTR, HDMapNet, VectorMapNet
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.spatial.distance import directed_hausdorff, cdist
from collections import defaultdict


def calculate_chamfer_distance_polyline(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    max_distance: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate Chamfer Distance between two polylines.
    
    Primary metric for polyline/curve matching in vector map evaluation.
    
    Args:
        pred_polyline: Predicted polyline points, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline points, shape (M, 2) or (M, 3)
        max_distance: Optional maximum distance threshold for filtering outliers
        
    Returns:
        Dictionary with:
            - chamfer_distance: Average bidirectional chamfer distance
            - chamfer_pred_to_gt: Pred -> GT distance
            - chamfer_gt_to_pred: GT -> Pred distance
            - precision: Fraction of pred points within threshold
            - recall: Fraction of GT points within threshold
            
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0.1]])
        >>> gt = np.array([[0, 0], [1, 0], [2, 0]])
        >>> result = chamfer_distance_polyline(pred, gt)
        >>> print(f"Chamfer: {result['chamfer_distance']:.3f}m")
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'chamfer_distance': float('inf'),
            'chamfer_pred_to_gt': float('inf'),
            'chamfer_gt_to_pred': float('inf'),
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Compute pairwise distances
    dist_matrix = cdist(pred_polyline, gt_polyline)
    
    # Pred to GT: for each predicted point, find nearest GT point
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    chamfer_pred_to_gt = np.mean(min_dist_pred_to_gt)
    
    # GT to Pred: for each GT point, find nearest predicted point
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)
    chamfer_gt_to_pred = np.mean(min_dist_gt_to_pred)
    
    # Symmetric Chamfer distance
    chamfer_distance = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2
    
    # Precision and recall (if threshold provided)
    precision = recall = None
    if max_distance is not None:
        precision = np.mean(min_dist_pred_to_gt <= max_distance)
        recall = np.mean(min_dist_gt_to_pred <= max_distance)
    
    return {
        'chamfer_distance': float(chamfer_distance),
        'chamfer_pred_to_gt': float(chamfer_pred_to_gt),
        'chamfer_gt_to_pred': float(chamfer_gt_to_pred),
        'precision': float(precision) if precision is not None else None,
        'recall': float(recall) if recall is not None else None
    }


def calculate_frechet_distance(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray
) -> float:
    """
    Calculate Fréchet Distance between two polylines.
    
    Measures similarity of curves considering ordering and continuity.
    Also known as "dog-leash distance" - better than Chamfer for curves.
    
    Args:
        pred_polyline: Predicted polyline points, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline points, shape (M, 2) or (M, 3)
        
    Returns:
        Fréchet distance (meters)
        
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0]])
        >>> gt = np.array([[0, 0.1], [1, 0.1], [2, 0.1]])
        >>> dist = frechet_distance(pred, gt)
    """
    n = len(pred_polyline)
    m = len(gt_polyline)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Dynamic programming matrix
    ca = np.full((n, m), -1.0)
    
    def _compute_frechet(i: int, j: int) -> float:
        """Recursive computation with memoization."""
        if ca[i, j] > -1:
            return ca[i, j]
        
        # Distance between current points
        d = np.linalg.norm(pred_polyline[i] - gt_polyline[j])
        
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_compute_frechet(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_compute_frechet(0, j-1), d)
        else:
            ca[i, j] = max(
                min(
                    _compute_frechet(i-1, j),
                    _compute_frechet(i-1, j-1),
                    _compute_frechet(i, j-1)
                ),
                d
            )
        
        return ca[i, j]
    
    return float(_compute_frechet(n-1, m-1))


def calculate_polyline_iou(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    width: float = 1.0,
    num_samples: int = 100
) -> float:
    """
    Calculate IoU between two polylines by treating them as thin rectangles.
    
    Useful for lane line matching where width represents lane width tolerance.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2)
        gt_polyline: Ground truth polyline, shape (M, 2)
        width: Width of the polyline corridor (meters)
        num_samples: Number of points to sample along polylines for IoU calculation
        
    Returns:
        IoU score [0, 1]
        
    Example:
        >>> pred = np.array([[0, 0], [10, 0]])
        >>> gt = np.array([[0, 0.2], [10, 0.2]])
        >>> iou = polyline_iou(pred, gt, width=1.0)
    """
    if len(pred_polyline) < 2 or len(gt_polyline) < 2:
        return 0.0
    
    # Sample points uniformly along both polylines
    def sample_polyline(polyline, n_samples):
        """Sample n points uniformly along polyline."""
        # Calculate cumulative distances
        segments = np.diff(polyline, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]
        
        if total_length == 0:
            return polyline[:1]
        
        # Sample at uniform intervals
        sample_distances = np.linspace(0, total_length, n_samples)
        sampled_points = []
        
        for dist in sample_distances:
            # Find which segment this distance falls on
            seg_idx = np.searchsorted(cumulative_lengths, dist) - 1
            seg_idx = max(0, min(seg_idx, len(polyline) - 2))
            
            # Interpolate within segment
            seg_start_dist = cumulative_lengths[seg_idx]
            seg_length = segment_lengths[seg_idx]
            
            if seg_length > 0:
                t = (dist - seg_start_dist) / seg_length
                t = np.clip(t, 0, 1)
                point = polyline[seg_idx] + t * segments[seg_idx]
            else:
                point = polyline[seg_idx]
            
            sampled_points.append(point)
        
        return np.array(sampled_points)
    
    # Sample both polylines
    pred_samples = sample_polyline(pred_polyline, num_samples)
    gt_samples = sample_polyline(gt_polyline, num_samples)
    
    # Compute distances between all sample pairs
    dist_matrix = cdist(pred_samples, gt_samples)
    
    # Count intersection: points within width threshold
    pred_in_gt = np.any(dist_matrix <= width, axis=1)
    gt_in_pred = np.any(dist_matrix <= width, axis=0)
    
    intersection = np.sum(pred_in_gt) + np.sum(gt_in_pred)
    union = 2 * num_samples  # Both polylines sampled with num_samples points
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(np.clip(iou, 0, 1))


def calculate_lane_detection_metrics(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    distance_threshold: float = 1.5,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive lane detection evaluation metrics.
    
    Used in: OpenLane, OpenLane-V2, Argoverse 2 HD Map Challenge.
    
    Args:
        pred_lanes: List of predicted lane polylines, each shape (N, 2) or (N, 3)
        gt_lanes: List of ground truth lane polylines, each shape (M, 2) or (M, 3)
        distance_threshold: Maximum chamfer distance for matching (meters)
        iou_threshold: Minimum IoU for matching
        
    Returns:
        Dictionary with:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1_score: Harmonic mean of precision and recall
            - chamfer_distance: Mean chamfer distance for matched lanes
            - tp: Number of true positives
            - fp: Number of false positives
            - fn: Number of false negatives
            
    Example:
        >>> pred = [np.array([[0, 0], [10, 0]]), np.array([[0, 3], [10, 3]])]
        >>> gt = [np.array([[0, 0.1], [10, 0.1]]), np.array([[0, 3.1], [10, 3.1]])]
        >>> metrics = lane_detection_metrics(pred, gt)
    """
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'chamfer_distance': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    
    if len(gt_lanes) == 0:
        return {
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0,
            'chamfer_distance': float('inf'),
            'tp': 0,
            'fp': len(pred_lanes),
            'fn': 0
        }
    
    if len(pred_lanes) == 0:
        return {
            'precision': 1.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'chamfer_distance': float('inf'),
            'tp': 0,
            'fp': 0,
            'fn': len(gt_lanes)
        }
    
    # Compute pairwise Chamfer distances between all pred-GT lane pairs
    chamfer_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
    
    for i, pred_lane in enumerate(pred_lanes):
        for j, gt_lane in enumerate(gt_lanes):
            result = calculate_chamfer_distance_polyline(pred_lane, gt_lane)
            chamfer_matrix[i, j] = result['chamfer_distance']
    
    # Greedy matching: match pred to GT based on minimum Chamfer distance
    matched_pred = set()
    matched_gt = set()
    matched_distances = []
    
    # Sort all pairs by chamfer distance
    pairs = []
    for i in range(len(pred_lanes)):
        for j in range(len(gt_lanes)):
            pairs.append((chamfer_matrix[i, j], i, j))
    pairs.sort()
    
    # Greedily match
    for dist, pred_idx, gt_idx in pairs:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            if dist <= distance_threshold:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                matched_distances.append(dist)
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_lanes) - tp
    fn = len(gt_lanes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_chamfer = np.mean(matched_distances) if len(matched_distances) > 0 else float('inf')
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'chamfer_distance': float(mean_chamfer),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_topology_metrics(
    pred_topology: Dict[str, List[int]],
    gt_topology: Dict[str, List[int]],
    lane_matches: Dict[int, int]
) -> Dict[str, float]:
    """
    Evaluate lane topology/connectivity prediction.
    
    Measures how well the model predicts lane-to-lane relationships:
    - Lane successors (which lanes follow current lane)
    - Lane predecessors (which lanes lead to current lane)
    - Left/right neighbors
    
    Args:
        pred_topology: Predicted topology, e.g., {'successors': [lane_ids], 'left_neighbor': [id]}
        gt_topology: Ground truth topology, same format
        lane_matches: Mapping from predicted lane ID to GT lane ID
        
    Returns:
        Dictionary with:
            - topology_precision: Correct connections / predicted connections
            - topology_recall: Correct connections / GT connections
            - topology_f1: F1 score
            
    Example:
        >>> pred_topo = {'successors': [1, 2], 'left_neighbor': [3]}
        >>> gt_topo = {'successors': [1, 2], 'left_neighbor': [3, 4]}
        >>> matches = {0: 0, 1: 1, 2: 2, 3: 3}
        >>> metrics = topology_metrics(pred_topo, gt_topo, matches)
    """
    # Count correct and total topology connections
    correct_connections = 0
    pred_connections = 0
    gt_connections = 0
    
    # Check each topology type (successors, predecessors, left_neighbor, right_neighbor)
    topology_types = set(list(pred_topology.keys()) + list(gt_topology.keys()))
    
    for topo_type in topology_types:
        pred_ids = set(pred_topology.get(topo_type, []))
        gt_ids = set(gt_topology.get(topo_type, []))
        
        # Map predicted IDs to GT space
        mapped_pred_ids = set()
        for pred_id in pred_ids:
            if pred_id in lane_matches:
                mapped_pred_ids.add(lane_matches[pred_id])
        
        # Count matches
        correct = len(mapped_pred_ids & gt_ids)
        correct_connections += correct
        pred_connections += len(pred_ids)
        gt_connections += len(gt_ids)
    
    # Calculate metrics
    precision = correct_connections / pred_connections if pred_connections > 0 else 0.0
    recall = correct_connections / gt_connections if gt_connections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'topology_precision': float(precision),
        'topology_recall': float(recall),
        'topology_f1': float(f1),
        'correct_connections': correct_connections,
        'pred_connections': pred_connections,
        'gt_connections': gt_connections
    }


def calculate_endpoint_error(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray
) -> Dict[str, float]:
    """
    Calculate error at polyline endpoints.
    
    Important for lane merging/splitting scenarios and topology.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline, shape (M, 2) or (M, 3)
        
    Returns:
        Dictionary with:
            - start_error: Distance between start points
            - end_error: Distance between end points
            - mean_endpoint_error: Average of start and end errors
            
    Example:
        >>> pred = np.array([[0, 0], [10, 0]])
        >>> gt = np.array([[0.1, 0], [10.1, 0]])
        >>> errors = endpoint_error(pred, gt)
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'start_error': float('inf'),
            'end_error': float('inf'),
            'mean_endpoint_error': float('inf')
        }
    
    start_error = np.linalg.norm(pred_polyline[0] - gt_polyline[0])
    end_error = np.linalg.norm(pred_polyline[-1] - gt_polyline[-1])
    mean_error = (start_error + end_error) / 2
    
    return {
        'start_error': float(start_error),
        'end_error': float(end_error),
        'mean_endpoint_error': float(mean_error)
    }


def calculate_direction_accuracy(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Evaluate directional accuracy of polyline (tangent vector alignment).
    
    Measures whether the predicted lane follows the correct direction.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline, shape (M, 2) or (M, 3)
        num_samples: Number of points to sample for direction comparison
        
    Returns:
        Dictionary with:
            - mean_direction_error: Mean angle error in radians
            - mean_direction_error_deg: Mean angle error in degrees
            - direction_accuracy: Fraction of samples with error < 15°
            
    Example:
        >>> pred = np.array([[0, 0], [10, 1]])
        >>> gt = np.array([[0, 0], [10, 0]])
        >>> acc = direction_accuracy(pred, gt)
    """
    if len(pred_polyline) < 2 or len(gt_polyline) < 2:
        return {
            'mean_direction_error': float('inf'),
            'mean_direction_error_deg': float('inf'),
            'direction_accuracy': 0.0
        }
    
    # Sample points along pred polyline
    pred_length = np.sum(np.linalg.norm(np.diff(pred_polyline, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_polyline, axis=0), axis=1))
    
    if pred_length == 0 or gt_length == 0:
        return {
            'mean_direction_error': float('inf'),
            'mean_direction_error_deg': float('inf'),
            'direction_accuracy': 0.0
        }
    
    # Get tangent vectors at sampled points
    direction_errors = []
    
    for i in range(num_samples):
        # Sample parameter t
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        
        # Get tangent at this point for both polylines
        pred_idx = int(t * (len(pred_polyline) - 1))
        gt_idx = int(t * (len(gt_polyline) - 1))
        
        pred_idx = min(pred_idx, len(pred_polyline) - 2)
        gt_idx = min(gt_idx, len(gt_polyline) - 2)
        
        # Tangent vectors
        pred_tangent = pred_polyline[pred_idx + 1] - pred_polyline[pred_idx]
        gt_tangent = gt_polyline[gt_idx + 1] - gt_polyline[gt_idx]
        
        # Normalize
        pred_tangent_norm = pred_tangent / (np.linalg.norm(pred_tangent) + 1e-8)
        gt_tangent_norm = gt_tangent / (np.linalg.norm(gt_tangent) + 1e-8)
        
        # Angle between vectors (handle 2D and 3D)
        dot_product = np.dot(pred_tangent_norm, gt_tangent_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error = np.arccos(dot_product)
        
        direction_errors.append(angle_error)
    
    mean_error_rad = np.mean(direction_errors)
    mean_error_deg = np.degrees(mean_error_rad)
    
    # Accuracy: fraction within 15 degrees
    accuracy = np.mean([err < np.radians(15) for err in direction_errors])
    
    return {
        'mean_direction_error': float(mean_error_rad),
        'mean_direction_error_deg': float(mean_error_deg),
        'direction_accuracy': float(accuracy)
    }


def calculate_vectormap_ap(
    pred_lanes: List[Dict],
    gt_lanes: List[Dict],
    distance_thresholds: List[float] = [0.5, 1.0, 1.5],
    confidence_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate Average Precision for vector map detection.
    
    Similar to object detection AP but using Chamfer distance for matching.
    
    Args:
        pred_lanes: List of predicted lane dicts with 'polyline' (np.ndarray) and optional 'score'
        gt_lanes: List of ground truth lane dicts with 'polyline' (np.ndarray)
        distance_thresholds: Chamfer distance thresholds for matching (meters)
        confidence_scores: Optional confidence scores for predictions (if not in pred_lanes)
        
    Returns:
        Dictionary with AP at different thresholds:
            - ap_0.5: AP at 0.5m threshold
            - ap_1.0: AP at 1.0m threshold
            - ap_1.5: AP at 1.5m threshold
            - map: Mean AP across all thresholds
            
    Example:
        >>> pred = [{'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}]
        >>> gt = [{'polyline': np.array([[0, 0.1], [10, 0.1]])}]
        >>> ap = vectormap_ap(pred, gt)
    """
    results = {}
    
    # Extract confidence scores
    if confidence_scores is None:
        scores = [lane.get('score', 1.0) for lane in pred_lanes]
    else:
        scores = confidence_scores
    
    # Sort predictions by confidence (descending)
    if len(scores) > 0:
        sorted_indices = np.argsort(scores)[::-1]
        pred_lanes = [pred_lanes[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
    
    # Calculate AP for each threshold
    aps = []
    for threshold in distance_thresholds:
        # Track which GT lanes have been matched
        matched_gt = set()
        tp = []
        fp = []
        
        for pred_idx, pred_lane in enumerate(pred_lanes):
            pred_polyline = pred_lane['polyline']
            
            # Find best matching GT lane
            best_chamfer = float('inf')
            best_gt_idx = -1
            
            for gt_idx, gt_lane in enumerate(gt_lanes):
                if gt_idx in matched_gt:
                    continue
                
                gt_polyline = gt_lane['polyline']
                result = calculate_chamfer_distance_polyline(pred_polyline, gt_polyline)
                chamfer = result['chamfer_distance']
                
                if chamfer < best_chamfer:
                    best_chamfer = chamfer
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_chamfer <= threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_lanes) if len(gt_lanes) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calculate AP (area under PR curve)
        # Use 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
        results[f'ap_{threshold:.1f}'] = float(ap)
    
    # Mean AP
    results['map'] = float(np.mean(aps))
    
    return results


def calculate_chamfer_distance_3d(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    max_distance: Optional[float] = None,
    weight_z: float = 1.0
) -> Dict[str, float]:
    """
    Calculate 3D Chamfer Distance for 3D lane polylines with elevation.
    
    Used in OpenLane-V2 for 3D lane detection evaluation.
    
    Args:
        pred_polyline: Predicted 3D polyline, shape (N, 3) with [x, y, z]
        gt_polyline: Ground truth 3D polyline, shape (M, 3) with [x, y, z]
        max_distance: Optional maximum distance threshold for filtering
        weight_z: Weight for z-axis (elevation) differences, default 1.0
        
    Returns:
        Dictionary with:
            - chamfer_distance_3d: 3D chamfer distance
            - chamfer_distance_xy: 2D (horizontal) chamfer distance
            - elevation_error: Mean absolute elevation error
            - chamfer_pred_to_gt: Forward distance
            - chamfer_gt_to_pred: Backward distance
            
    Example:
        >>> pred = np.array([[0, 0, 0], [10, 0, 0.5]])
        >>> gt = np.array([[0, 0, 0.1], [10, 0, 0.4]])
        >>> result = calculate_chamfer_distance_3d(pred, gt)
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'chamfer_distance_3d': float('inf'),
            'chamfer_distance_xy': float('inf'),
            'elevation_error': float('inf'),
            'chamfer_pred_to_gt': float('inf'),
            'chamfer_gt_to_pred': float('inf')
        }
    
    # Ensure 3D input
    if pred_polyline.shape[1] != 3 or gt_polyline.shape[1] != 3:
        raise ValueError("Polylines must be 3D (N, 3) for 3D Chamfer distance")
    
    # Weight z-axis if specified (ensure float type)
    pred_weighted = pred_polyline.astype(np.float64).copy()
    gt_weighted = gt_polyline.astype(np.float64).copy()
    pred_weighted[:, 2] *= weight_z
    gt_weighted[:, 2] *= weight_z
    
    # Compute 3D distance matrix
    dist_matrix_3d = cdist(pred_weighted, gt_weighted)
    
    # Pred to GT
    min_dist_pred_to_gt = np.min(dist_matrix_3d, axis=1)
    chamfer_pred_to_gt = np.mean(min_dist_pred_to_gt)
    
    # GT to Pred
    min_dist_gt_to_pred = np.min(dist_matrix_3d, axis=0)
    chamfer_gt_to_pred = np.mean(min_dist_gt_to_pred)
    
    # Symmetric 3D Chamfer
    chamfer_3d = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2
    
    # Also compute 2D (XY) chamfer for reference
    dist_matrix_xy = cdist(pred_polyline[:, :2], gt_polyline[:, :2])
    chamfer_xy = (np.mean(np.min(dist_matrix_xy, axis=1)) + 
                  np.mean(np.min(dist_matrix_xy, axis=0))) / 2
    
    # Elevation error: find closest XY points and compare Z
    closest_gt_indices = np.argmin(dist_matrix_xy, axis=1)
    elevation_diffs = np.abs(pred_polyline[:, 2] - gt_polyline[closest_gt_indices, 2])
    elevation_error = np.mean(elevation_diffs)
    
    return {
        'chamfer_distance_3d': float(chamfer_3d),
        'chamfer_distance_xy': float(chamfer_xy),
        'elevation_error': float(elevation_error),
        'chamfer_pred_to_gt': float(chamfer_pred_to_gt),
        'chamfer_gt_to_pred': float(chamfer_gt_to_pred)
    }


def calculate_frechet_distance_3d(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    weight_z: float = 1.0
) -> Dict[str, float]:
    """
    Calculate 3D Fréchet Distance for 3D lane polylines.
    
    Extends Fréchet distance to 3D space with elevation awareness.
    
    Args:
        pred_polyline: Predicted 3D polyline, shape (N, 3)
        gt_polyline: Ground truth 3D polyline, shape (M, 3)
        weight_z: Weight for z-axis differences
        
    Returns:
        Dictionary with:
            - frechet_distance_3d: 3D Fréchet distance
            - frechet_distance_xy: 2D Fréchet distance (for reference)
            
    Example:
        >>> pred = np.array([[0, 0, 0], [5, 0, 1], [10, 0, 2]])
        >>> gt = np.array([[0, 0, 0.1], [5, 0, 1.1], [10, 0, 2.1]])
        >>> result = calculate_frechet_distance_3d(pred, gt)
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'frechet_distance_3d': float('inf'),
            'frechet_distance_xy': float('inf')
        }
    
    if pred_polyline.shape[1] != 3 or gt_polyline.shape[1] != 3:
        raise ValueError("Polylines must be 3D (N, 3)")
    
    # Weight z-axis (ensure float type)
    pred_weighted = pred_polyline.astype(np.float64).copy()
    gt_weighted = gt_polyline.astype(np.float64).copy()
    pred_weighted[:, 2] *= weight_z
    gt_weighted[:, 2] *= weight_z
    
    # Compute 3D Fréchet distance using dynamic programming
    n = len(pred_weighted)
    m = len(gt_weighted)
    
    # Distance matrix
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = np.linalg.norm(pred_weighted[i] - gt_weighted[j])
    
    # DP table for Fréchet distance
    ca = np.full((n, m), -1.0)
    
    def compute_frechet(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        
        if i == 0 and j == 0:
            ca[i, j] = dist_matrix[i, j]
        elif i > 0 and j == 0:
            ca[i, j] = max(compute_frechet(i-1, 0), dist_matrix[i, 0])
        elif i == 0 and j > 0:
            ca[i, j] = max(compute_frechet(0, j-1), dist_matrix[0, j])
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(compute_frechet(i-1, j), compute_frechet(i-1, j-1), compute_frechet(i, j-1)),
                dist_matrix[i, j]
            )
        else:
            ca[i, j] = float('inf')
        
        return ca[i, j]
    
    frechet_3d = compute_frechet(n-1, m-1)
    
    # Also compute 2D Fréchet for reference
    frechet_2d = calculate_frechet_distance(pred_polyline[:, :2], gt_polyline[:, :2])
    
    return {
        'frechet_distance_3d': float(frechet_3d),
        'frechet_distance_xy': float(frechet_2d)
    }


def calculate_online_lane_segment_metric(
    pred_lanes_sequence: List[List[np.ndarray]],
    gt_lanes_sequence: List[List[np.ndarray]],
    distance_threshold: float = 1.0,
    iou_threshold: float = 0.5,
    consistency_weight: float = 0.3
) -> Dict[str, float]:
    """
    Calculate Online Lane Segment (OLS) metric for temporal consistency.
    
    Evaluates both detection quality and temporal tracking consistency
    across frames. Used in OpenLane-V2 for online HD map construction.
    
    Args:
        pred_lanes_sequence: List of predicted lanes per frame, each frame is List[np.ndarray]
        gt_lanes_sequence: List of ground truth lanes per frame
        distance_threshold: Chamfer distance threshold for matching
        iou_threshold: IoU threshold for matching
        consistency_weight: Weight for temporal consistency [0, 1]
        
    Returns:
        Dictionary with:
            - ols: Overall Online Lane Segment score
            - detection_score: Per-frame detection F1 (average)
            - consistency_score: Temporal tracking consistency
            - avg_precision: Average precision across frames
            - avg_recall: Average recall across frames
            - id_switches: Number of identity switches
            
    Example:
        >>> # Frame 1, 2, 3 predictions and ground truth
        >>> pred_seq = [[lane1_f1, lane2_f1], [lane1_f2, lane2_f2], ...]
        >>> gt_seq = [[lane1_f1, lane2_f1], [lane1_f2, lane2_f2], ...]
        >>> result = calculate_online_lane_segment_metric(pred_seq, gt_seq)
    """
    if len(pred_lanes_sequence) != len(gt_lanes_sequence):
        raise ValueError("Prediction and GT sequences must have same length")
    
    if len(pred_lanes_sequence) == 0:
        return {
            'ols': 0.0,
            'detection_score': 0.0,
            'consistency_score': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'id_switches': 0
        }
    
    num_frames = len(pred_lanes_sequence)
    frame_f1_scores = []
    frame_precisions = []
    frame_recalls = []
    
    # Track lane IDs across frames for consistency
    prev_matches = {}  # pred_idx -> gt_idx for previous frame
    id_switches = 0
    consistent_tracks = 0
    total_tracks = 0
    
    for frame_idx in range(num_frames):
        pred_lanes = pred_lanes_sequence[frame_idx]
        gt_lanes = gt_lanes_sequence[frame_idx]
        
        # Calculate per-frame detection metrics
        frame_metrics = calculate_lane_detection_metrics(
            pred_lanes, gt_lanes,
            distance_threshold=distance_threshold,
            iou_threshold=iou_threshold
        )
        
        frame_f1_scores.append(frame_metrics['f1_score'])
        frame_precisions.append(frame_metrics['precision'])
        frame_recalls.append(frame_metrics['recall'])
        
        # Track consistency: check if same lanes matched in consecutive frames
        if frame_idx > 0 and len(pred_lanes) > 0 and len(gt_lanes) > 0:
            # Compute current frame matches
            chamfer_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
            for i, pred_lane in enumerate(pred_lanes):
                for j, gt_lane in enumerate(gt_lanes):
                    result = calculate_chamfer_distance_polyline(pred_lane, gt_lane)
                    chamfer_matrix[i, j] = result['chamfer_distance']
            
            # Find current matches
            current_matches = {}
            matched_pred = set()
            matched_gt = set()
            
            pairs = []
            for i in range(len(pred_lanes)):
                for j in range(len(gt_lanes)):
                    pairs.append((chamfer_matrix[i, j], i, j))
            pairs.sort()
            
            for dist, pred_idx, gt_idx in pairs:
                if pred_idx not in matched_pred and gt_idx not in matched_gt:
                    if dist <= distance_threshold:
                        current_matches[pred_idx] = gt_idx
                        matched_pred.add(pred_idx)
                        matched_gt.add(gt_idx)
            
            # Compare with previous frame
            for pred_idx, gt_idx in current_matches.items():
                total_tracks += 1
                if pred_idx in prev_matches:
                    if prev_matches[pred_idx] == gt_idx:
                        consistent_tracks += 1
                    else:
                        id_switches += 1
            
            prev_matches = current_matches
        elif len(pred_lanes) > 0 and len(gt_lanes) > 0:
            # Initialize tracking for first frame
            chamfer_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
            for i, pred_lane in enumerate(pred_lanes):
                for j, gt_lane in enumerate(gt_lanes):
                    result = calculate_chamfer_distance_polyline(pred_lane, gt_lane)
                    chamfer_matrix[i, j] = result['chamfer_distance']
            
            matched_pred = set()
            matched_gt = set()
            pairs = []
            for i in range(len(pred_lanes)):
                for j in range(len(gt_lanes)):
                    pairs.append((chamfer_matrix[i, j], i, j))
            pairs.sort()
            
            for dist, pred_idx, gt_idx in pairs:
                if pred_idx not in matched_pred and gt_idx not in matched_gt:
                    if dist <= distance_threshold:
                        prev_matches[pred_idx] = gt_idx
                        matched_pred.add(pred_idx)
                        matched_gt.add(gt_idx)
    
    # Calculate scores
    avg_f1 = np.mean(frame_f1_scores) if frame_f1_scores else 0.0
    avg_precision = np.mean(frame_precisions) if frame_precisions else 0.0
    avg_recall = np.mean(frame_recalls) if frame_recalls else 0.0
    
    # Consistency score: ratio of consistent tracks
    consistency_score = consistent_tracks / total_tracks if total_tracks > 0 else 1.0
    
    # OLS: weighted combination of detection and consistency
    detection_score = avg_f1
    ols = (1 - consistency_weight) * detection_score + consistency_weight * consistency_score
    
    return {
        'ols': float(ols),
        'detection_score': float(detection_score),
        'consistency_score': float(consistency_score),
        'avg_precision': float(avg_precision),
        'avg_recall': float(avg_recall),
        'id_switches': int(id_switches)
    }


def calculate_per_category_metrics(
    pred_lanes: List[Dict[str, Any]],
    gt_lanes: List[Dict[str, Any]],
    categories: List[str],
    distance_threshold: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """
    Calculate separate metrics for different lane/map element categories.
    
    Supports evaluation of:
    - Lane dividers (dashed, solid)
    - Road boundaries (curbs, edges)
    - Crosswalks
    - Stop lines
    - Other map elements
    
    Args:
        pred_lanes: List of predicted lane dicts with 'polyline' and 'category'
        gt_lanes: List of GT lane dicts with 'polyline' and 'category'
        categories: List of category names to evaluate
        distance_threshold: Matching threshold in meters
        
    Returns:
        Dictionary with per-category metrics:
            {
                'lane_divider': {'precision': ..., 'recall': ..., 'f1': ..., 'ap': ...},
                'road_edge': {...},
                'crosswalk': {...},
                ...
                'overall': {...}  # Averaged across all categories
            }
            
    Example:
        >>> pred = [
        ...     {'polyline': np.array([[0,0],[10,0]]), 'category': 'lane_divider', 'score': 0.9},
        ...     {'polyline': np.array([[0,3],[10,3]]), 'category': 'road_edge', 'score': 0.8}
        ... ]
        >>> gt = [
        ...     {'polyline': np.array([[0,0.1],[10,0.1]]), 'category': 'lane_divider'},
        ...     {'polyline': np.array([[0,3.1],[10,3.1]]), 'category': 'road_edge'}
        ... ]
        >>> result = calculate_per_category_metrics(pred, gt, ['lane_divider', 'road_edge'])
    """
    results = {}
    
    # Group by category
    pred_by_category = defaultdict(list)
    gt_by_category = defaultdict(list)
    
    for pred in pred_lanes:
        category = pred.get('category', 'unknown')
        pred_by_category[category].append(pred)
    
    for gt in gt_lanes:
        category = gt.get('category', 'unknown')
        gt_by_category[category].append(gt)
    
    # Evaluate each category
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_aps = []
    
    for category in categories:
        pred_cat = pred_by_category.get(category, [])
        gt_cat = gt_by_category.get(category, [])
        
        # Extract polylines
        pred_polylines = [p['polyline'] for p in pred_cat]
        gt_polylines = [g['polyline'] for g in gt_cat]
        
        if len(gt_polylines) == 0 and len(pred_polylines) == 0:
            # No GT and no predictions for this category
            results[category] = {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'ap': 1.0,
                'num_pred': 0,
                'num_gt': 0
            }
            continue
        
        # Calculate detection metrics
        det_metrics = calculate_lane_detection_metrics(
            pred_polylines, gt_polylines,
            distance_threshold=distance_threshold
        )
        
        # Calculate AP for this category
        ap_result = calculate_vectormap_ap(
            pred_cat, gt_cat,
            distance_thresholds=[distance_threshold]
        )
        
        results[category] = {
            'precision': det_metrics['precision'],
            'recall': det_metrics['recall'],
            'f1_score': det_metrics['f1_score'],
            'ap': ap_result.get(f'ap_{distance_threshold:.1f}', 0.0),
            'num_pred': len(pred_polylines),
            'num_gt': len(gt_polylines)
        }
        
        if len(gt_polylines) > 0:  # Only include in average if GT exists
            all_precisions.append(det_metrics['precision'])
            all_recalls.append(det_metrics['recall'])
            all_f1s.append(det_metrics['f1_score'])
            all_aps.append(ap_result.get(f'ap_{distance_threshold:.1f}', 0.0))
    
    # Overall metrics (macro-average across categories with GT)
    if len(all_precisions) > 0:
        results['overall'] = {
            'precision': float(np.mean(all_precisions)),
            'recall': float(np.mean(all_recalls)),
            'f1_score': float(np.mean(all_f1s)),
            'map': float(np.mean(all_aps)),
            'num_categories': len(all_precisions)
        }
    else:
        results['overall'] = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'map': 0.0,
            'num_categories': 0
        }
    
    return results

