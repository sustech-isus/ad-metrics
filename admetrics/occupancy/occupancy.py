"""
3D Occupancy Prediction Metrics.

This module provides metrics for evaluating 3D occupancy prediction models
used in autonomous driving. Occupancy prediction predicts which voxels in
a 3D grid are occupied by different semantic classes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from scipy.ndimage import label as connected_components


def calculate_occupancy_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    class_id: Optional[int] = None,
    ignore_index: int = 255
) -> float:
    """
    Calculate Intersection over Union (IoU) for 3D occupancy predictions.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        class_id: Specific class ID to compute IoU for. If None, treats as binary occupancy
        ignore_index: Label value to ignore in calculation (e.g., unknown/invalid voxels)
    
    Returns:
        IoU score between 0 and 1
    
    Example:
        >>> pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        >>> gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        >>> iou = occupancy_iou(pred, gt, class_id=1)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    # Create mask for valid voxels (not ignored)
    valid_mask = gt_occupancy != ignore_index
    
    if class_id is not None:
        # Calculate IoU for specific class
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
    else:
        # Binary occupancy (occupied vs free)
        pred_mask = (pred_occupancy > 0) & valid_mask
        gt_mask = (gt_occupancy > 0) & valid_mask
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def calculate_mean_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    ignore_classes: Optional[List[int]] = None
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Calculate mean Intersection over Union (mIoU) across all classes.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        num_classes: Total number of semantic classes (including background/free space)
        ignore_index: Label value to ignore in calculation
        ignore_classes: List of class IDs to exclude from mIoU calculation
    
    Returns:
        Dictionary containing:
            - 'mIoU': Mean IoU across all valid classes
            - 'class_iou': Dictionary mapping class_id to IoU score
            - 'valid_classes': Number of classes with non-zero union
    
    Example:
        >>> pred = np.random.randint(0, 3, size=(10, 10, 10))
        >>> gt = np.random.randint(0, 3, size=(10, 10, 10))
        >>> result = mean_iou(pred, gt, num_classes=3)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    ignore_classes = ignore_classes or []
    valid_mask = gt_occupancy != ignore_index
    
    class_ious = {}
    valid_class_count = 0
    
    for class_id in range(num_classes):
        if class_id in ignore_classes:
            continue
        
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union > 0:
            class_ious[class_id] = float(intersection / union)
            valid_class_count += 1
        else:
            # Class not present in ground truth or prediction
            class_ious[class_id] = np.nan
    
    # Calculate mIoU only over classes that appear in the data
    valid_ious = [iou for iou in class_ious.values() if not np.isnan(iou)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    
    return {
        'mIoU': miou,
        'class_iou': class_ious,
        'valid_classes': valid_class_count
    }


def calculate_occupancy_precision_recall(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    class_id: Optional[int] = None,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for occupancy predictions.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        class_id: Specific class ID to evaluate. If None, treats as binary occupancy
        ignore_index: Label value to ignore in calculation
    
    Returns:
        Dictionary containing:
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1-score
            - 'true_positives': Number of true positives
            - 'false_positives': Number of false positives
            - 'false_negatives': Number of false negatives
    
    Example:
        >>> pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        >>> gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        >>> metrics = occupancy_precision_recall(pred, gt, class_id=1)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    valid_mask = gt_occupancy != ignore_index
    
    if class_id is not None:
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
    else:
        pred_mask = (pred_occupancy > 0) & valid_mask
        gt_mask = (gt_occupancy > 0) & valid_mask
    
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def calculate_scene_completion(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    free_class: int = 0,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Calculate Scene Completion (SC) metrics for 3D occupancy.
    
    Scene Completion evaluates how well the model completes the scene by
    predicting both occupied and free space correctly.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        free_class: Class ID representing free/empty space (default: 0)
        ignore_index: Label value to ignore in calculation
    
    Returns:
        Dictionary containing:
            - 'SC_IoU': IoU for scene completion (occupied space)
            - 'SC_Precision': Precision for occupied voxels
            - 'SC_Recall': Recall for occupied voxels
            - 'SSC_mIoU': Semantic Scene Completion mIoU (excludes free space)
            - 'completion_ratio': Ratio of predicted occupied to GT occupied voxels
    
    Example:
        >>> pred = np.random.randint(0, 3, size=(10, 10, 10))
        >>> gt = np.random.randint(0, 3, size=(10, 10, 10))
        >>> sc_metrics = scene_completion(pred, gt, free_class=0)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    valid_mask = gt_occupancy != ignore_index
    
    # Binary occupancy: occupied (any non-free class) vs free
    pred_occupied = (pred_occupancy != free_class) & valid_mask
    gt_occupied = (gt_occupancy != free_class) & valid_mask
    
    # Scene Completion IoU (binary: occupied vs free)
    intersection = np.logical_and(pred_occupied, gt_occupied).sum()
    union = np.logical_or(pred_occupied, gt_occupied).sum()
    sc_iou = float(intersection / union) if union > 0 else 0.0
    
    # Precision and Recall for occupied voxels
    tp = intersection
    fp = np.logical_and(pred_occupied, ~gt_occupied).sum()
    fn = np.logical_and(~pred_occupied, gt_occupied).sum()
    
    sc_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    sc_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Completion ratio
    pred_occupied_count = pred_occupied.sum()
    gt_occupied_count = gt_occupied.sum()
    completion_ratio = float(pred_occupied_count / gt_occupied_count) if gt_occupied_count > 0 else 0.0
    
    # Semantic Scene Completion mIoU (for occupied voxels only)
    # Get unique classes excluding free space
    occupied_classes = np.unique(gt_occupancy[gt_occupied])
    occupied_classes = occupied_classes[occupied_classes != free_class]
    
    if len(occupied_classes) > 0:
        class_ious = []
        for class_id in occupied_classes:
            pred_class = (pred_occupancy == class_id) & valid_mask
            gt_class = (gt_occupancy == class_id) & valid_mask
            
            class_intersection = np.logical_and(pred_class, gt_class).sum()
            class_union = np.logical_or(pred_class, gt_class).sum()
            
            if class_union > 0:
                class_ious.append(class_intersection / class_union)
        
        ssc_miou = float(np.mean(class_ious)) if class_ious else 0.0
    else:
        ssc_miou = 0.0
    
    return {
        'SC_IoU': sc_iou,
        'SC_Precision': sc_precision,
        'SC_Recall': sc_recall,
        'SSC_mIoU': ssc_miou,
        'completion_ratio': completion_ratio
    }


def calculate_chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    bidirectional: bool = True
) -> Dict[str, float]:
    """
    Calculate Chamfer Distance between predicted and ground truth point clouds.
    
    Chamfer Distance measures the average distance from each point in one set
    to its nearest neighbor in the other set. Useful for evaluating occupancy
    prediction quality at surface boundaries.
    
    Args:
        pred_points: Predicted occupied voxel centers, shape (N, 3)
        gt_points: Ground truth occupied voxel centers, shape (M, 3)
        bidirectional: If True, compute symmetric Chamfer Distance (both directions)
    
    Returns:
        Dictionary containing:
            - 'chamfer_distance': Average Chamfer Distance
            - 'pred_to_gt': Average distance from predicted to GT points
            - 'gt_to_pred': Average distance from GT to predicted points (if bidirectional)
    
    Example:
        >>> pred_pts = np.random.rand(100, 3)
        >>> gt_pts = np.random.rand(120, 3)
        >>> cd = chamfer_distance(pred_pts, gt_pts)
    """
    if pred_points.shape[1] != 3 or gt_points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {
            'chamfer_distance': np.inf,
            'pred_to_gt': np.inf,
            'gt_to_pred': np.inf if bidirectional else None
        }
    
    # Compute pairwise distances
    distances = cdist(pred_points, gt_points, metric='euclidean')
    
    # Pred to GT: for each predicted point, find nearest GT point
    pred_to_gt = np.mean(np.min(distances, axis=1))
    
    result = {
        'pred_to_gt': float(pred_to_gt)
    }
    
    if bidirectional:
        # GT to Pred: for each GT point, find nearest predicted point
        gt_to_pred = np.mean(np.min(distances, axis=0))
        result['gt_to_pred'] = float(gt_to_pred)
        result['chamfer_distance'] = float((pred_to_gt + gt_to_pred) / 2)
    else:
        result['chamfer_distance'] = float(pred_to_gt)
        result['gt_to_pred'] = None
    
    return result


def calculate_surface_distance(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    voxel_size: float = 1.0,
    percentile: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate surface distance metrics between predicted and GT occupancy.
    
    Measures how well the predicted occupancy boundaries match the ground truth
    boundaries by computing distances between surface voxels.
    
    Args:
        pred_occupancy: Predicted binary occupancy grid, shape (X, Y, Z)
        gt_occupancy: Ground truth binary occupancy grid, shape (X, Y, Z)
        voxel_size: Physical size of each voxel in meters
        percentile: If specified, compute percentile distance (e.g., 95 for 95th percentile)
    
    Returns:
        Dictionary containing:
            - 'mean_surface_distance': Average distance between surfaces
            - 'median_surface_distance': Median distance
            - 'std_surface_distance': Standard deviation of distances
            - 'percentile_distance': Percentile distance if percentile is specified
            - 'max_surface_distance': Maximum distance
    
    Example:
        >>> pred = np.random.randint(0, 2, size=(20, 20, 20))
        >>> gt = np.random.randint(0, 2, size=(20, 20, 20))
        >>> sd = surface_distance(pred, gt, voxel_size=0.2)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    # Convert to binary if not already
    pred_binary = (pred_occupancy > 0).astype(bool)
    gt_binary = (gt_occupancy > 0).astype(bool)
    
    # Find surface voxels (voxels with at least one free neighbor)
    def get_surface_voxels(occupancy: np.ndarray) -> np.ndarray:
        """Extract coordinates of surface voxels."""
        from scipy.ndimage import binary_erosion
        
        # Surface = occupied voxels that are not completely surrounded
        eroded = binary_erosion(occupancy)
        surface = occupancy & ~eroded
        
        return np.argwhere(surface)
    
    pred_surface = get_surface_voxels(pred_binary)
    gt_surface = get_surface_voxels(gt_binary)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return {
            'mean_surface_distance': np.inf,
            'median_surface_distance': np.inf,
            'std_surface_distance': 0.0,
            'max_surface_distance': np.inf,
            'percentile_distance': np.inf if percentile else None
        }
    
    # Convert voxel coordinates to physical coordinates
    pred_surface_coords = pred_surface * voxel_size
    gt_surface_coords = gt_surface * voxel_size
    
    # Compute distances
    distances = cdist(pred_surface_coords, gt_surface_coords, metric='euclidean')
    
    # For each predicted surface voxel, find distance to nearest GT surface voxel
    min_distances = np.min(distances, axis=1)
    
    result = {
        'mean_surface_distance': float(np.mean(min_distances)),
        'median_surface_distance': float(np.median(min_distances)),
        'std_surface_distance': float(np.std(min_distances)),
        'max_surface_distance': float(np.max(min_distances))
    }
    
    if percentile is not None:
        result['percentile_distance'] = float(np.percentile(min_distances, percentile))
    else:
        result['percentile_distance'] = None
    
    return result


def calculate_f_score(pred_occupancy: np.ndarray, gt_occupancy: np.ndarray, beta: float = 1.0, class_id: Optional[int] = None, ignore_index: int = 255) -> float:
    """
    Compute F-score for occupancy predictions (wrapper around precision/recall).
    """
    pr = calculate_occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=class_id, ignore_index=ignore_index)
    precision = pr['precision']
    recall = pr['recall']
    if precision + recall == 0:
        return 0.0
    return (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)


def calculate_hausdorff_distance(*args, **kwargs):
    """
    Placeholder alias for Hausdorff distance.
    
    This implementation currently delegates to :func:`calculate_chamfer_distance`,
    which is commonly used as an alternative metric for point cloud comparison.
    True Hausdorff distance (max of min distances) is not implemented.
    
    For point cloud quality evaluation, use :func:`calculate_chamfer_distance` directly.
    """
    return calculate_chamfer_distance(*args, **kwargs)


def calculate_ray_iou(*args, **kwargs):
    """
    Legacy compatibility alias for ray-based IoU evaluation.
    
    Note: True ray-based IoU (as used in some Occ3D evaluations) is not
    fully implemented. This function delegates to :func:`calculate_mean_iou`
    for standard voxel-based mIoU calculation.
    
    For standard occupancy evaluation, use :func:`calculate_mean_iou` directly.
    
    Args:
        Same as calculate_mean_iou
        
    Returns:
        Same as calculate_mean_iou
    """
    # Delegate to mean IoU for compatibility
    return calculate_mean_iou(*args, **kwargs)


def calculate_depth_error(*args, **kwargs):
    """
    Placeholder for depth error metrics.
    
    Depth error evaluation is not implemented in the occupancy module.
    For depth estimation metrics, consider using dedicated depth evaluation
    tools or implementing custom depth comparison logic.
    
    Raises:
        NotImplementedError: This metric is not available
    """
    raise NotImplementedError("calculate_depth_error is not implemented in this module")


def calculate_absrel_error(*args, **kwargs):
    """
    Placeholder for absolute relative error (AbsRel) metrics.
    
    AbsRel error evaluation is not implemented in the occupancy module.
    For depth estimation metrics, consider using dedicated depth evaluation
    tools or implementing custom depth comparison logic.
    
    Raises:
        NotImplementedError: This metric is not available
    """
    raise NotImplementedError("calculate_absrel_error is not implemented in this module")


def calculate_occupancy_metrics(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    num_classes: int = 2,
    pred_points: Optional[np.ndarray] = None,
    gt_points: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Convenience wrapper that returns a collection of occupancy metrics.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z)
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z)
        num_classes: Number of semantic classes (default: 2 for binary)
        pred_points: Optional predicted occupied voxel centers (N, 3)
        gt_points: Optional ground truth occupied voxel centers (M, 3)
    
    Returns:
        Dictionary with keys for IoU, mean IoU, scene completion and chamfer.
        
    Example:
        >>> pred = np.random.randint(0, 2, size=(10, 10, 10))
        >>> gt = np.random.randint(0, 2, size=(10, 10, 10))
        >>> metrics = calculate_occupancy_metrics(pred, gt, num_classes=2)
    """
    metrics = {}
    metrics['occupancy_iou'] = calculate_occupancy_iou(pred_occupancy, gt_occupancy)
    metrics['mean_iou'] = calculate_mean_iou(pred_occupancy, gt_occupancy, num_classes=num_classes)
    metrics['scene_completion'] = calculate_scene_completion(pred_occupancy, gt_occupancy)
    
    # Only calculate Chamfer distance if point clouds are provided
    if pred_points is not None and gt_points is not None:
        metrics['chamfer'] = calculate_chamfer_distance(pred_points, gt_points)
    else:
        metrics['chamfer'] = None
    
    return metrics


def calculate_visibility_weighted_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    visibility_mask: Optional[np.ndarray] = None,
    num_classes: int = 16,
    ignore_index: int = 255
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Calculate visibility-weighted IoU for occupancy predictions.
    
    This metric downweights regions that are occluded or not visible from
    the sensor viewpoint, providing a more fair evaluation for camera-based
    or LiDAR-based occupancy prediction methods.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        visibility_mask: Binary mask of shape (X, Y, Z) indicating visible voxels.
                        If None, uses distance-based visibility (closer = more visible)
        num_classes: Total number of semantic classes
        ignore_index: Label value to ignore in calculation
    
    Returns:
        Dictionary containing:
            - 'visibility_weighted_mIoU': Weighted mean IoU
            - 'class_iou': Dictionary of per-class visibility-weighted IoU
            - 'visible_voxel_ratio': Ratio of visible voxels used
    
    Example:
        >>> pred = np.random.randint(0, 16, size=(200, 200, 16))
        >>> gt = np.random.randint(0, 16, size=(200, 200, 16))
        >>> # Create visibility mask (e.g., from ray-casting or distance)
        >>> vis_mask = create_visibility_mask(gt)
        >>> result = calculate_visibility_weighted_iou(pred, gt, vis_mask, num_classes=16)
        >>> print(f"Visibility-weighted mIoU: {result['visibility_weighted_mIoU']:.4f}")
    
    References:
        - Occ3D benchmark uses visibility-aware evaluation
        - Accounts for occlusion in camera-based occupancy prediction
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    # Create visibility mask if not provided (distance-based)
    if visibility_mask is None:
        # Simple distance-based visibility: closer voxels are more visible
        # Assumes ego vehicle at center bottom
        center_x, center_y = pred_occupancy.shape[0] // 2, pred_occupancy.shape[1] // 2
        x, y, z = np.meshgrid(
            np.arange(pred_occupancy.shape[0]),
            np.arange(pred_occupancy.shape[1]),
            np.arange(pred_occupancy.shape[2]),
            indexing='ij'
        )
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2 + z**2)
        max_dist = np.sqrt(pred_occupancy.shape[0]**2 + pred_occupancy.shape[1]**2 + pred_occupancy.shape[2]**2)
        visibility_mask = 1.0 - (distances / max_dist)  # Closer = higher weight
    else:
        visibility_mask = visibility_mask.astype(float)
    
    valid_mask = (gt_occupancy != ignore_index)
    
    class_ious = {}
    weighted_ious = []
    
    for class_id in range(num_classes):
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
        
        # Weight by visibility
        intersection = np.logical_and(pred_mask, gt_mask) * visibility_mask
        union_mask = np.logical_or(pred_mask, gt_mask)
        union = union_mask * visibility_mask
        
        intersection_sum = intersection.sum()
        union_sum = union.sum()
        
        if union_sum > 0:
            iou = float(intersection_sum / union_sum)
            class_ious[class_id] = iou
            weighted_ious.append(iou)
        else:
            class_ious[class_id] = np.nan
    
    # Calculate weighted mIoU
    weighted_miou = float(np.mean(weighted_ious)) if weighted_ious else 0.0
    visible_ratio = float(visibility_mask[valid_mask].sum() / valid_mask.sum()) if valid_mask.sum() > 0 else 0.0
    
    return {
        'visibility_weighted_mIoU': weighted_miou,
        'class_iou': class_ious,
        'visible_voxel_ratio': visible_ratio
    }


def calculate_panoptic_quality(
    pred_occupancy: np.ndarray,
    pred_instances: np.ndarray,
    gt_occupancy: np.ndarray,
    gt_instances: np.ndarray,
    num_classes: int = 16,
    ignore_index: int = 255,
    stuff_classes: Optional[List[int]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate Panoptic Quality (PQ) for instance-aware occupancy prediction.
    
    Panoptic Quality combines semantic segmentation (stuff classes like road, building)
    with instance segmentation (thing classes like vehicles, pedestrians).
    
    PQ = (TP) / (TP + 0.5*FP + 0.5*FN) for matched instances
    
    Args:
        pred_occupancy: Predicted semantic occupancy grid (X, Y, Z) with class labels
        pred_instances: Predicted instance IDs (X, Y, Z), 0 = no instance
        gt_occupancy: Ground truth semantic occupancy grid (X, Y, Z)
        gt_instances: Ground truth instance IDs (X, Y, Z), 0 = no instance
        num_classes: Total number of semantic classes
        ignore_index: Label value to ignore
        stuff_classes: List of stuff class IDs (e.g., [0, 1, 2] for free, road, sidewalk).
                      If None, treats all classes as things
    
    Returns:
        Dictionary containing:
            - 'PQ': Overall Panoptic Quality [0, 1]
            - 'SQ': Segmentation Quality (IoU of matched instances)
            - 'RQ': Recognition Quality (F1 of detection)
            - 'PQ_stuff': PQ for stuff classes
            - 'PQ_thing': PQ for thing classes
            - 'per_class_pq': Dictionary of per-class PQ scores
    
    Example:
        >>> pred_sem = np.random.randint(0, 16, size=(200, 200, 16))
        >>> pred_inst = np.random.randint(0, 100, size=(200, 200, 16))
        >>> gt_sem = np.random.randint(0, 16, size=(200, 200, 16))
        >>> gt_inst = np.random.randint(0, 100, size=(200, 200, 16))
        >>> pq = calculate_panoptic_quality(pred_sem, pred_inst, gt_sem, gt_inst,
        ...                                  num_classes=16, stuff_classes=[0, 1, 2])
        >>> print(f"PQ: {pq['PQ']:.4f}, SQ: {pq['SQ']:.4f}, RQ: {pq['RQ']:.4f}")
    
    References:
        - Panoptic Segmentation (Kirillov et al., CVPR 2019)
        - Adapted for 3D occupancy prediction with instances
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(f"Occupancy shape mismatch")
    if pred_instances.shape != gt_instances.shape:
        raise ValueError(f"Instance shape mismatch")
    
    stuff_classes = stuff_classes or []
    thing_classes = [c for c in range(num_classes) if c not in stuff_classes]
    
    valid_mask = gt_occupancy != ignore_index
    
    per_class_pq = {}
    per_class_sq = {}
    per_class_rq = {}
    
    stuff_pq_scores = []
    thing_pq_scores = []
    
    for class_id in range(num_classes):
        # Get voxels for this class
        gt_class_mask = (gt_occupancy == class_id) & valid_mask
        pred_class_mask = (pred_occupancy == class_id) & valid_mask
        
        if class_id in stuff_classes:
            # Stuff classes: treat as semantic segmentation (no instances)
            intersection = np.logical_and(gt_class_mask, pred_class_mask).sum()
            union = np.logical_or(gt_class_mask, pred_class_mask).sum()
            
            if union > 0:
                iou = intersection / union
                pq = sq = iou
                rq = 1.0
                stuff_pq_scores.append(pq)
            else:
                pq = sq = rq = 0.0
        else:
            # Thing classes: instance-level matching
            gt_inst_in_class = gt_instances[gt_class_mask]
            pred_inst_in_class = pred_instances[pred_class_mask]
            
            gt_unique_ids = np.unique(gt_inst_in_class)
            gt_unique_ids = gt_unique_ids[gt_unique_ids > 0]  # Exclude background
            
            pred_unique_ids = np.unique(pred_inst_in_class)
            pred_unique_ids = pred_unique_ids[pred_unique_ids > 0]
            
            if len(gt_unique_ids) == 0 and len(pred_unique_ids) == 0:
                pq = sq = rq = 1.0
            elif len(gt_unique_ids) == 0 or len(pred_unique_ids) == 0:
                pq = sq = rq = 0.0
            else:
                # Match instances using IoU threshold (0.5)
                iou_threshold = 0.5
                matched_pairs = []
                matched_ious = []
                
                for gt_id in gt_unique_ids:
                    gt_inst_mask = (gt_instances == gt_id) & gt_class_mask
                    
                    best_iou = 0.0
                    best_pred_id = None
                    
                    for pred_id in pred_unique_ids:
                        pred_inst_mask = (pred_instances == pred_id) & pred_class_mask
                        
                        intersection = np.logical_and(gt_inst_mask, pred_inst_mask).sum()
                        union = np.logical_or(gt_inst_mask, pred_inst_mask).sum()
                        
                        if union > 0:
                            iou = intersection / union
                            if iou > best_iou and iou >= iou_threshold:
                                best_iou = iou
                                best_pred_id = pred_id
                    
                    if best_pred_id is not None:
                        matched_pairs.append((gt_id, best_pred_id))
                        matched_ious.append(best_iou)
                
                tp = len(matched_pairs)
                fp = len(pred_unique_ids) - tp
                fn = len(gt_unique_ids) - tp
                
                # Segmentation Quality (average IoU of matched instances)
                sq = float(np.mean(matched_ious)) if matched_ious else 0.0
                
                # Recognition Quality (F1 score)
                rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
                
                # Panoptic Quality
                pq = sq * rq
                
                thing_pq_scores.append(pq)
        
        per_class_pq[class_id] = pq
        per_class_sq[class_id] = sq
        per_class_rq[class_id] = rq
    
    # Overall metrics
    all_pq_scores = stuff_pq_scores + thing_pq_scores
    overall_pq = float(np.mean(all_pq_scores)) if all_pq_scores else 0.0
    overall_sq = float(np.mean([sq for sq in per_class_sq.values() if sq > 0])) if per_class_sq else 0.0
    overall_rq = float(np.mean([rq for rq in per_class_rq.values() if rq > 0])) if per_class_rq else 0.0
    
    pq_stuff = float(np.mean(stuff_pq_scores)) if stuff_pq_scores else 0.0
    pq_thing = float(np.mean(thing_pq_scores)) if thing_pq_scores else 0.0
    
    return {
        'PQ': overall_pq,
        'SQ': overall_sq,
        'RQ': overall_rq,
        'PQ_stuff': pq_stuff,
        'PQ_thing': pq_thing,
        'per_class_pq': per_class_pq,
        'per_class_sq': per_class_sq,
        'per_class_rq': per_class_rq
    }


def calculate_video_panoptic_quality(
    pred_occupancy_sequence: List[np.ndarray],
    pred_instances_sequence: List[np.ndarray],
    gt_occupancy_sequence: List[np.ndarray],
    gt_instances_sequence: List[np.ndarray],
    num_classes: int = 16,
    ignore_index: int = 255,
    stuff_classes: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate Video Panoptic Quality (VPQ) for temporal occupancy prediction.
    
    VPQ extends Panoptic Quality to video sequences, measuring both spatial
    accuracy and temporal consistency of instance tracking across frames.
    
    Args:
        pred_occupancy_sequence: List of predicted semantic grids, each (X, Y, Z)
        pred_instances_sequence: List of predicted instance IDs, each (X, Y, Z)
        gt_occupancy_sequence: List of ground truth semantic grids, each (X, Y, Z)
        gt_instances_sequence: List of ground truth instance IDs, each (X, Y, Z)
        num_classes: Total number of semantic classes
        ignore_index: Label value to ignore
        stuff_classes: List of stuff class IDs
    
    Returns:
        Dictionary containing:
            - 'VPQ': Video Panoptic Quality [0, 1]
            - 'STQ': Spatial-Temporal Quality
            - 'AQ': Association Quality (temporal consistency)
            - 'per_frame_pq': List of PQ scores for each frame
    
    Example:
        >>> # 10-frame sequence
        >>> pred_seq = [np.random.randint(0, 16, (200, 200, 16)) for _ in range(10)]
        >>> pred_inst_seq = [np.random.randint(0, 100, (200, 200, 16)) for _ in range(10)]
        >>> gt_seq = [np.random.randint(0, 16, (200, 200, 16)) for _ in range(10)]
        >>> gt_inst_seq = [np.random.randint(0, 100, (200, 200, 16)) for _ in range(10)]
        >>> vpq = calculate_video_panoptic_quality(pred_seq, pred_inst_seq, 
        ...                                         gt_seq, gt_inst_seq, num_classes=16)
        >>> print(f"VPQ: {vpq['VPQ']:.4f}, STQ: {vpq['STQ']:.4f}")
    
    References:
        - Video Panoptic Segmentation (Kim et al., CVPR 2020)
        - Adapted for 4D occupancy prediction (3D space + time)
    """
    if len(pred_occupancy_sequence) != len(gt_occupancy_sequence):
        raise ValueError("Sequence length mismatch")
    
    num_frames = len(pred_occupancy_sequence)
    
    # Calculate per-frame PQ
    per_frame_pq = []
    per_frame_sq = []
    per_frame_rq = []
    
    for t in range(num_frames):
        pq_result = calculate_panoptic_quality(
            pred_occupancy_sequence[t],
            pred_instances_sequence[t],
            gt_occupancy_sequence[t],
            gt_instances_sequence[t],
            num_classes=num_classes,
            ignore_index=ignore_index,
            stuff_classes=stuff_classes
        )
        per_frame_pq.append(pq_result['PQ'])
        per_frame_sq.append(pq_result['SQ'])
        per_frame_rq.append(pq_result['RQ'])
    
    # Calculate Association Quality (temporal consistency)
    # Measure how well instance IDs are maintained across frames
    association_scores = []
    
    for t in range(num_frames - 1):
        # Compare instance assignments between consecutive frames
        curr_gt_inst = gt_instances_sequence[t]
        next_gt_inst = gt_instances_sequence[t + 1]
        curr_pred_inst = pred_instances_sequence[t]
        next_pred_inst = pred_instances_sequence[t + 1]
        
        # Find overlapping instances in GT
        gt_ids_curr = np.unique(curr_gt_inst)
        gt_ids_curr = gt_ids_curr[gt_ids_curr > 0]
        
        temporal_consistency = []
        
        for gt_id in gt_ids_curr:
            # Check if this GT instance exists in next frame
            if gt_id in next_gt_inst:
                # Get voxels for this instance in both frames
                curr_gt_mask = (curr_gt_inst == gt_id)
                next_gt_mask = (next_gt_inst == gt_id)
                
                # Find predicted instance IDs in current frame
                pred_ids_at_gt = curr_pred_inst[curr_gt_mask]
                if len(pred_ids_at_gt) > 0:
                    # Most common predicted ID for this GT instance
                    curr_pred_id = np.bincount(pred_ids_at_gt[pred_ids_at_gt > 0])
                    if len(curr_pred_id) > 0:
                        curr_pred_id = np.argmax(curr_pred_id)
                        
                        # Check if same predicted ID is used in next frame
                        pred_ids_at_gt_next = next_pred_inst[next_gt_mask]
                        if curr_pred_id in pred_ids_at_gt_next:
                            consistency = (pred_ids_at_gt_next == curr_pred_id).sum() / len(pred_ids_at_gt_next)
                            temporal_consistency.append(consistency)
        
        if temporal_consistency:
            association_scores.append(np.mean(temporal_consistency))
    
    # Compute overall metrics
    avg_pq = float(np.mean(per_frame_pq)) if per_frame_pq else 0.0
    avg_sq = float(np.mean(per_frame_sq)) if per_frame_sq else 0.0
    avg_rq = float(np.mean(per_frame_rq)) if per_frame_rq else 0.0
    aq = float(np.mean(association_scores)) if association_scores else 0.0
    
    # STQ (Spatial-Temporal Quality)
    stq = np.sqrt(avg_sq * aq) if (avg_sq > 0 and aq > 0) else 0.0
    
    # VPQ combines spatial quality and temporal consistency
    vpq = avg_pq * aq
    
    return {
        'VPQ': float(vpq),
        'STQ': float(stq),
        'AQ': aq,
        'avg_PQ': avg_pq,
        'avg_SQ': avg_sq,
        'avg_RQ': avg_rq,
        'per_frame_pq': per_frame_pq
    }

