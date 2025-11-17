"""
Multi-Object Tracking (MOT) metrics for 3D object tracking evaluation.

Implements CLEAR MOT metrics (MOTA, MOTP) and other tracking-specific metrics
including ID switches, fragmentations, and HOTA.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


def calculate_mota(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Multiple Object Tracking Accuracy (MOTA).
    
    MOTA = 1 - (FN + FP + ID_SW) / GT
    
    Note: For pure detection (single frame), ID switches are 0.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with MOTA and components
    """
    from admetrics.detection.confusion import calculate_tp_fp_fn
    
    counts = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold)
    
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    num_gt = len(ground_truth)
    
    if num_gt == 0:
        return {
            'mota': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_gt': 0
        }
    
    # For detection only (no tracking), ID switches = 0
    id_switches = 0
    
    mota = 1 - (fn + fp + id_switches) / num_gt
    
    return {
        'mota': float(mota),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'id_switches': id_switches,
        'num_gt': num_gt
    }


def calculate_motp(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    distance_type: str = "euclidean"
) -> Dict[str, float]:
    """
    Calculate Multiple Object Tracking Precision (MOTP).
    
    MOTP = sum(distance_i) / num_TP
    
    Average distance error for all true positive detections.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold for matching
        distance_type: Type of distance metric to use
    
    Returns:
        Dictionary with MOTP and related metrics
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    
    # Match predictions to ground truth
    gt_matched = [False] * len(ground_truth)
    distances = []
    
    for pred in predictions:
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            if pred.get('class') != gt.get('class'):
                continue
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            gt_matched[max_gt_idx] = True
            
            # Calculate distance
            dist = calculate_center_distance(
                pred['box'],
                ground_truth[max_gt_idx]['box'],
                distance_type=distance_type
            )
            distances.append(dist)
    
    if len(distances) == 0:
        return {
            'motp': 0.0,
            'mean_distance': 0.0,
            'num_tp': 0
        }
    
    motp = np.mean(distances)
    
    return {
        'motp': float(motp),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'num_tp': len(distances)
    }


def calculate_clearmot_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate CLEAR MOT metrics (combines MOTA and MOTP).
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with both MOTA and MOTP metrics
    """
    mota_result = calculate_mota(predictions, ground_truth, iou_threshold)
    motp_result = calculate_motp(predictions, ground_truth, iou_threshold)
    
    return {
        **mota_result,
        **motp_result
    }


def calculate_multi_frame_mota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate MOTA across multiple frames with ID switch tracking.
    
    MOTA = 1 - (FN + FP + IDSW) / total_GT
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> list of predictions
                          Each prediction should have 'box', 'track_id', 'class'
        frame_ground_truth: Dictionary mapping frame_id -> list of ground truth
                           Each GT should have 'box', 'track_id', 'class'
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with tracking metrics including:
            - mota: Multiple Object Tracking Accuracy
            - motp: Multiple Object Tracking Precision
            - num_matches: Total true positives
            - num_false_positives: Total false positives
            - num_misses: Total false negatives (missed detections)
            - num_switches: Total ID switches
            - num_fragmentations: Total track fragmentations
            - mostly_tracked: Number of mostly tracked GT trajectories
            - mostly_lost: Number of mostly lost GT trajectories
            - partially_tracked: Number of partially tracked GT trajectories
    
    Example:
        >>> frame_preds = {
        ...     0: [{'box': [0,0,0,4,2,1.5,0], 'track_id': 1, 'class': 'car'}],
        ...     1: [{'box': [1,0,0,4,2,1.5,0], 'track_id': 1, 'class': 'car'}]
        ... }
        >>> frame_gt = {
        ...     0: [{'box': [0,0,0,4,2,1.5,0], 'track_id': 100, 'class': 'car'}],
        ...     1: [{'box': [1,0,0,4,2,1.5,0], 'track_id': 100, 'class': 'car'}]
        ... }
        >>> metrics = calculate_multi_frame_mota(frame_preds, frame_gt)
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance
    
    # Tracking state
    num_matches = 0
    num_false_positives = 0
    num_misses = 0
    num_switches = 0
    total_gt = 0
    total_distance = 0.0
    
    # Track GT -> Pred mapping for ID switch detection
    gt_to_pred_mapping = {}  # {gt_track_id: pred_track_id}
    
    # Track coverage for each GT trajectory
    gt_track_frames = defaultdict(int)  # {gt_track_id: num_frames_detected}
    gt_track_total_frames = defaultdict(int)  # {gt_track_id: total_frames}
    
    # Process frames in order
    for frame_id in sorted(frame_predictions.keys()):
        if frame_id not in frame_ground_truth:
            # All predictions are FP if no GT in this frame
            num_false_positives += len(frame_predictions[frame_id])
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        total_gt += len(gts)
        
        # Count total frames for each GT track
        for gt in gts:
            gt_id = gt.get('track_id')
            if gt_id is not None:
                gt_track_total_frames[gt_id] += 1
        
        # Match predictions to ground truth using Hungarian assignment
        matches, unmatched_preds, unmatched_gts = _match_frame(
            preds, gts, iou_threshold
        )
        
        # Process matches
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            num_matches += 1
            
            # Calculate distance for MOTP
            dist = calculate_center_distance(pred['box'], gt['box'])
            total_distance += dist
            
            # Check for ID switch
            gt_id = gt.get('track_id')
            pred_id = pred.get('track_id')
            
            if gt_id is not None and pred_id is not None:
                gt_track_frames[gt_id] += 1
                
                if gt_id in gt_to_pred_mapping:
                    # Check if predicted ID matches previous assignment
                    if gt_to_pred_mapping[gt_id] != pred_id:
                        num_switches += 1
                        gt_to_pred_mapping[gt_id] = pred_id
                else:
                    # First time seeing this GT track
                    gt_to_pred_mapping[gt_id] = pred_id
        
        num_false_positives += len(unmatched_preds)
        num_misses += len(unmatched_gts)
    
    # Calculate MOTA
    if total_gt == 0:
        mota = 0.0
        motp = 0.0
    else:
        mota = 1.0 - (num_misses + num_false_positives + num_switches) / total_gt
        motp = total_distance / num_matches if num_matches > 0 else 0.0
    
    # Calculate trajectory-level metrics
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0
    
    for gt_id, frames_detected in gt_track_frames.items():
        total_frames = gt_track_total_frames[gt_id]
        ratio = frames_detected / total_frames if total_frames > 0 else 0
        
        if ratio >= 0.8:
            mostly_tracked += 1
        elif ratio >= 0.2:
            partially_tracked += 1
        else:
            mostly_lost += 1
    
    # Count fragmentations (GT tracks that were lost and then re-found)
    num_fragmentations = _count_fragmentations(frame_predictions, frame_ground_truth, iou_threshold)
    
    return {
        'mota': float(mota),
        'motp': float(motp),
        'num_matches': num_matches,
        'num_false_positives': num_false_positives,
        'num_misses': num_misses,
        'num_switches': num_switches,
        'num_fragmentations': num_fragmentations,
        'total_gt': total_gt,
        'precision': num_matches / (num_matches + num_false_positives) if (num_matches + num_false_positives) > 0 else 0.0,
        'recall': num_matches / (num_matches + num_misses) if (num_matches + num_misses) > 0 else 0.0,
        'mostly_tracked': mostly_tracked,
        'partially_tracked': partially_tracked,
        'mostly_lost': mostly_lost,
        'num_gt_trajectories': len(gt_track_total_frames)
    }


def _match_frame(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predictions to ground truth in a single frame using Hungarian algorithm.
    
    Args:
        predictions: List of predictions with 'box' and 'class'
        ground_truth: List of ground truth with 'box' and 'class'
        iou_threshold: Minimum IoU for valid match
    
    Returns:
        matches: List of (pred_idx, gt_idx) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    from admetrics.detection.iou import calculate_iou_3d
    from scipy.optimize import linear_sum_assignment
    
    if len(predictions) == 0:
        return [], [], list(range(len(ground_truth)))
    
    if len(ground_truth) == 0:
        return [], list(range(len(predictions))), []
    
    # Compute IoU cost matrix
    cost_matrix = np.zeros((len(predictions), len(ground_truth)))
    
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            # Only match same class
            if pred.get('class') != gt.get('class'):
                cost_matrix[i, j] = 0
            else:
                iou = calculate_iou_3d(pred['box'], gt['box'])
                cost_matrix[i, j] = iou
    
    # Hungarian algorithm maximizes, so use negative cost
    cost_matrix_neg = -cost_matrix
    
    # Solve assignment problem
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix_neg)
    
    # Filter by IoU threshold
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] >= iou_threshold:
            matches.append((pred_idx, gt_idx))
    
    # Find unmatched
    matched_pred_indices = {m[0] for m in matches}
    matched_gt_indices = {m[1] for m in matches}
    
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred_indices]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gt_indices]
    
    return matches, unmatched_preds, unmatched_gts


def _count_fragmentations(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float
) -> int:
    """
    Count the number of times a ground truth track is fragmented.
    
    A fragmentation occurs when a GT track is matched, then unmatched,
    then matched again.
    
    Args:
        frame_predictions: Frame-indexed predictions
        frame_ground_truth: Frame-indexed ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Number of fragmentations
    """
    # Track the state of each GT trajectory
    gt_track_states = defaultdict(list)  # {gt_track_id: [matched, matched, unmatched, matched, ...]}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            # All GTs are unmatched
            for gt in frame_ground_truth[frame_id]:
                gt_id = gt.get('track_id')
                if gt_id is not None:
                    gt_track_states[gt_id].append(False)
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, _, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        # Mark matched GTs
        matched_gt_indices = {m[1] for m in matches}
        
        for gt_idx, gt in enumerate(gts):
            gt_id = gt.get('track_id')
            if gt_id is not None:
                is_matched = gt_idx in matched_gt_indices
                gt_track_states[gt_id].append(is_matched)
    
    # Count fragmentations
    num_fragmentations = 0
    
    for gt_id, states in gt_track_states.items():
        # Look for pattern: matched -> unmatched -> matched
        was_matched = False
        was_broken = False
        
        for is_matched in states:
            if was_matched and not is_matched:
                was_broken = True
            elif was_broken and is_matched:
                num_fragmentations += 1
                was_broken = False
            
            if is_matched:
                was_matched = True
    
    return num_fragmentations


def calculate_hota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Higher Order Tracking Accuracy (HOTA).
    
    HOTA = sqrt(DetA * AssA)
    
    Where:
    - DetA: Detection Accuracy
    - AssA: Association Accuracy
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for detection matching
    
    Returns:
        Dictionary with HOTA and sub-metrics
    
    Reference:
        "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking"
        Luiten et al., IJCV 2021
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Calculate detection accuracy
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Calculate association accuracy
    # Track correspondences across frames
    pred_track_to_gt = defaultdict(set)  # {pred_track_id: set of gt_track_ids}
    gt_track_to_pred = defaultdict(set)  # {gt_track_id: set of pred_track_ids}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            gts = frame_ground_truth[frame_id]
            total_fn += len(gts)
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
        
        # Track associations
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is not None and gt_id is not None:
                pred_track_to_gt[pred_id].add(gt_id)
                gt_track_to_pred[gt_id].add(pred_id)
    
    # Calculate DetA (Detection Accuracy)
    if total_tp + total_fp + total_fn == 0:
        det_a = 0.0
    else:
        det_a = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
    
    # Calculate AssA (Association Accuracy)
    # For each GT trajectory, find the best matching predicted trajectory
    total_tpa = 0  # True Positive Associations
    total_gt_trajectories = len(gt_track_to_pred)
    
    for gt_id, pred_ids in gt_track_to_pred.items():
        if len(pred_ids) == 0:
            continue
        
        # Find pred_id with maximum overlap
        max_overlap = 0
        for pred_id in pred_ids:
            # Count frames where both tracks are matched to each other
            # This is simplified; full HOTA uses Jaccard similarity
            overlap = 1 / (len(pred_ids) + len(pred_track_to_gt.get(pred_id, set())) - 1)
            max_overlap = max(max_overlap, overlap)
        
        total_tpa += max_overlap
    
    if total_gt_trajectories == 0:
        ass_a = 0.0
    else:
        ass_a = total_tpa / total_gt_trajectories
    
    # HOTA is geometric mean of DetA and AssA
    hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0.0
    
    return {
        'hota': float(hota),
        'det_a': float(det_a),
        'ass_a': float(ass_a),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_id_f1(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate ID-based precision, recall, and F1 score.
    
    Evaluates how well track identities are preserved across frames.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with IDP, IDR, IDF1 metrics
    """
    # Count correct ID assignments
    idtp = 0  # ID true positives
    idfp = 0  # ID false positives  
    idfn = 0  # ID false negatives
    
    # Track GT -> Pred mapping
    gt_to_pred_mapping = {}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is None or gt_id is None:
                continue
            
            # Check if this is the correct ID assignment
            if gt_id in gt_to_pred_mapping:
                if gt_to_pred_mapping[gt_id] == pred_id:
                    idtp += 1
                else:
                    idfp += 1
                    idfn += 1
            else:
                gt_to_pred_mapping[gt_id] = pred_id
                idtp += 1
        
        # Unmatched predictions with IDs are false positives
        for pred_idx in unmatched_preds:
            if preds[pred_idx].get('track_id') is not None:
                idfp += 1
        
        # Unmatched GTs with IDs are false negatives
        for gt_idx in unmatched_gts:
            if gts[gt_idx].get('track_id') is not None:
                idfn += 1
    
    # Calculate metrics
    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0  # ID Precision
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0  # ID Recall
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0  # ID F1
    
    return {
        'idp': float(idp),
        'idr': float(idr),
        'idf1': float(idf1),
        'idtp': idtp,
        'idfp': idfp,
        'idfn': idfn
    }


def calculate_amota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    recall_thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Average MOTA (AMOTA) across multiple recall thresholds.
    
    AMOTA is the primary metric used in nuScenes tracking benchmark.
    It evaluates MOTA at different recall operating points to provide
    a more robust measure of tracking performance.
    
    Formula:
        AMOTA = (1/R) * sum(MOTA(r)) for r in recall_thresholds
        where MOTA(r) is computed on top-scoring predictions to achieve recall r
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        recall_thresholds: List of recall thresholds (default: [0.2, 0.3, ..., 0.9])
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - amota: Average MOTA across recall thresholds
            - motas: List of MOTA values at each recall threshold
            - recall_thresholds: The recall thresholds used
            - amotp: Average MOTP across recall thresholds
            - motps: List of MOTP values at each recall threshold
    
    Reference:
        nuScenes Tracking Challenge
        https://www.nuscenes.org/tracking
    """
    if recall_thresholds is None:
        recall_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    motas = []
    motps = []
    
    for recall_thresh in recall_thresholds:
        # Filter predictions to achieve target recall
        filtered_predictions = _filter_predictions_by_recall(
            frame_predictions, frame_ground_truth, recall_thresh, iou_threshold
        )
        
        # Calculate MOTA at this recall level
        result = calculate_multi_frame_mota(filtered_predictions, frame_ground_truth, iou_threshold)
        motas.append(result['mota'])
        motps.append(result['motp'])
    
    amota = float(np.mean(motas))
    amotp = float(np.mean(motps))
    
    return {
        'amota': amota,
        'motas': motas,
        'recall_thresholds': recall_thresholds,
        'amotp': amotp,
        'motps': motps
    }


def _filter_predictions_by_recall(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    target_recall: float,
    iou_threshold: float
) -> Dict[int, List[Dict]]:
    """
    Filter predictions to achieve a target recall level.
    
    Sorts predictions by confidence and keeps top-k to reach target recall.
    
    Args:
        frame_predictions: Frame-indexed predictions
        frame_ground_truth: Frame-indexed ground truth
        target_recall: Target recall (0-1)
        iou_threshold: IoU threshold for matching
    
    Returns:
        Filtered predictions achieving approximately target recall
    """
    # Collect all predictions with scores
    all_preds_with_frames = []
    for frame_id, preds in frame_predictions.items():
        for pred in preds:
            all_preds_with_frames.append((frame_id, pred))
    
    # Sort by confidence score (descending)
    all_preds_with_frames.sort(key=lambda x: x[1].get('score', 1.0), reverse=True)
    
    # Count total GT
    total_gt = sum(len(gts) for gts in frame_ground_truth.values())
    
    if total_gt == 0:
        return frame_predictions
    
    # Determine how many predictions needed for target recall
    target_tp = int(target_recall * total_gt)
    
    # Greedily select predictions until we reach target TP count
    selected_preds = {frame_id: [] for frame_id in frame_predictions.keys()}
    gt_matched_global = {frame_id: set() for frame_id in frame_ground_truth.keys()}
    tp_count = 0
    
    for frame_id, pred in all_preds_with_frames:
        if tp_count >= target_tp:
            break
        
        # Check if this prediction matches any unmatched GT
        if frame_id not in frame_ground_truth:
            selected_preds[frame_id].append(pred)
            continue
        
        gts = frame_ground_truth[frame_id]
        matched = False
        
        for gt_idx, gt in enumerate(gts):
            if gt_idx in gt_matched_global[frame_id]:
                continue
            
            if pred.get('class') != gt.get('class'):
                continue
            
            from admetrics.detection.iou import calculate_iou_3d
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if iou >= iou_threshold:
                gt_matched_global[frame_id].add(gt_idx)
                matched = True
                tp_count += 1
                break
        
        selected_preds[frame_id].append(pred)
    
    return selected_preds


def calculate_motar(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    recall_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate MOTA at a specific Recall threshold (MOTAR).
    
    This metric is used in nuScenes to evaluate tracking performance
    at a fixed recall operating point.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        recall_threshold: Target recall level (default 0.5)
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - motar: MOTA at the specified recall threshold
            - recall: Actual achieved recall
            - mota: MOTA value
            - motp: MOTP value
    """
    filtered_preds = _filter_predictions_by_recall(
        frame_predictions, frame_ground_truth, recall_threshold, iou_threshold
    )
    
    result = calculate_multi_frame_mota(filtered_preds, frame_ground_truth, iou_threshold)
    
    return {
        'motar': result['mota'],
        'target_recall': recall_threshold,
        'actual_recall': result['recall'],
        'mota': result['mota'],
        'motp': result['motp']
    }


def calculate_false_alarm_rate(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate False Alarms per Frame (FAF) and related metrics.
    
    FAF measures the average number of false positive detections per frame.
    This is a key metric in nuScenes tracking evaluation.
    
    Formula:
        FAF = total_false_positives / num_frames
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - faf: False alarms per frame
            - total_false_positives: Total FP count
            - num_frames: Number of frames
            - far: False alarm rate (FP / total predictions)
    """
    total_fp = 0
    total_preds = 0
    num_frames = len(frame_predictions)
    
    for frame_id in frame_predictions.keys():
        preds = frame_predictions.get(frame_id, [])
        gts = frame_ground_truth.get(frame_id, [])
        
        total_preds += len(preds)
        
        matches, unmatched_preds, _ = _match_frame(preds, gts, iou_threshold)
        total_fp += len(unmatched_preds)
    
    faf = total_fp / num_frames if num_frames > 0 else 0.0
    far = total_fp / total_preds if total_preds > 0 else 0.0
    
    return {
        'faf': float(faf),
        'total_false_positives': total_fp,
        'num_frames': num_frames,
        'far': float(far)
    }


def calculate_track_metrics(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate track-level metrics including track recall and track precision.
    
    Track recall: Ratio of GT tracks that are detected at least once
    Track precision: Ratio of predicted tracks that match at least one GT
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - track_recall: Ratio of matched GT tracks
            - track_precision: Ratio of matched predicted tracks
            - num_gt_tracks: Total number of GT tracks
            - num_pred_tracks: Total number of predicted tracks
            - num_matched_tracks: Number of tracks with at least one match
    """
    # Collect all unique track IDs
    gt_track_ids = set()
    pred_track_ids = set()
    
    for gts in frame_ground_truth.values():
        for gt in gts:
            if gt.get('track_id') is not None:
                gt_track_ids.add(gt.get('track_id'))
    
    for preds in frame_predictions.values():
        for pred in preds:
            if pred.get('track_id') is not None:
                pred_track_ids.add(pred.get('track_id'))
    
    # Track which GT and Pred tracks have been matched
    matched_gt_tracks = set()
    matched_pred_tracks = set()
    
    for frame_id in frame_ground_truth.keys():
        if frame_id not in frame_predictions:
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, _, _ = _match_frame(preds, gts, iou_threshold)
        
        for pred_idx, gt_idx in matches:
            gt_id = gts[gt_idx].get('track_id')
            pred_id = preds[pred_idx].get('track_id')
            
            if gt_id is not None:
                matched_gt_tracks.add(gt_id)
            if pred_id is not None:
                matched_pred_tracks.add(pred_id)
    
    num_gt_tracks = len(gt_track_ids)
    num_pred_tracks = len(pred_track_ids)
    num_matched_gt = len(matched_gt_tracks)
    num_matched_pred = len(matched_pred_tracks)
    
    track_recall = num_matched_gt / num_gt_tracks if num_gt_tracks > 0 else 0.0
    track_precision = num_matched_pred / num_pred_tracks if num_pred_tracks > 0 else 0.0
    
    return {
        'track_recall': float(track_recall),
        'track_precision': float(track_precision),
        'num_gt_tracks': num_gt_tracks,
        'num_pred_tracks': num_pred_tracks,
        'num_matched_gt_tracks': num_matched_gt,
        'num_matched_pred_tracks': num_matched_pred
    }


def calculate_moda(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate MODA (Multiple Object Detection Accuracy).
    
    MODA is MOTA without the ID switch penalty - it only considers
    detection errors (false positives and false negatives).
    
    Formula:
        MODA = 1 - (FN + FP) / GT
    
    This is useful for evaluating pure detection quality independent
    of tracking/ID consistency.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - moda: Multiple Object Detection Accuracy
            - num_false_positives: Total false positives
            - num_misses: Total false negatives
            - total_gt: Total ground truth objects
    """
    total_fp = 0
    total_fn = 0
    total_gt = 0
    
    for frame_id in frame_ground_truth.keys():
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        total_gt += len(gts)
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
    
    if total_gt == 0:
        moda = 0.0
    else:
        moda = 1.0 - (total_fn + total_fp) / total_gt
    
    return {
        'moda': float(moda),
        'num_false_positives': total_fp,
        'num_misses': total_fn,
        'total_gt': total_gt
    }


def calculate_hota_components(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detailed HOTA components including all sub-metrics.
    
    Extended HOTA calculation with full decomposition:
    - DetA: Detection Accuracy (DetRe * DetPr)
    - AssA: Association Accuracy (AssRe * AssPr)
    - LocA: Localization Accuracy
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for detection matching
    
    Returns:
        Dictionary with:
            - hota: Higher Order Tracking Accuracy
            - det_a: Detection Accuracy
            - det_re: Detection Recall
            - det_pr: Detection Precision
            - ass_a: Association Accuracy
            - ass_re: Association Recall
            - ass_pr: Association Precision
            - loc_a: Localization Accuracy (Average IoU of TP)
    
    Reference:
        "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"
        Luiten et al., IJCV 2021
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    
    # Track associations
    pred_track_to_gt = defaultdict(set)
    gt_track_to_pred = defaultdict(set)
    track_match_counts = defaultdict(int)  # (gt_id, pred_id) -> count
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            gts = frame_ground_truth[frame_id]
            total_fn += len(gts)
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
        
        # Calculate IoU for localization accuracy
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            total_iou += iou
            
            # Track associations
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is not None and gt_id is not None:
                pred_track_to_gt[pred_id].add(gt_id)
                gt_track_to_pred[gt_id].add(pred_id)
                track_match_counts[(gt_id, pred_id)] += 1
    
    # Calculate Detection metrics
    if total_tp + total_fp + total_fn == 0:
        det_re = 0.0
        det_pr = 0.0
        det_a = 0.0
    else:
        det_re = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        det_pr = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        det_a = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
    
    # Calculate Association metrics
    # TPA: True Positive Associations (correctly associated tracks)
    tpa = 0
    for gt_id in gt_track_to_pred.keys():
        # Find best matching predicted track
        best_match_count = 0
        for pred_id in gt_track_to_pred[gt_id]:
            match_count = track_match_counts[(gt_id, pred_id)]
            best_match_count = max(best_match_count, match_count)
        tpa += best_match_count
    
    # Calculate association precision and recall
    num_gt_tracks = len(gt_track_to_pred)
    num_pred_tracks = len(pred_track_to_gt)
    
    if num_gt_tracks == 0 or num_pred_tracks == 0:
        ass_re = 0.0
        ass_pr = 0.0
        ass_a = 0.0
    else:
        ass_re = tpa / total_tp if total_tp > 0 else 0.0
        ass_pr = tpa / total_tp if total_tp > 0 else 0.0
        ass_a = tpa / (total_tp + 0.5 * (num_gt_tracks - len(gt_track_to_pred)) + 
                       0.5 * (num_pred_tracks - len(pred_track_to_gt))) if total_tp > 0 else 0.0
    
    # Calculate Localization Accuracy
    loc_a = total_iou / total_tp if total_tp > 0 else 0.0
    
    # HOTA is geometric mean of DetA and AssA
    hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0.0
    
    return {
        'hota': float(hota),
        'det_a': float(det_a),
        'det_re': float(det_re),
        'det_pr': float(det_pr),
        'ass_a': float(ass_a),
        'ass_re': float(ass_re),
        'ass_pr': float(ass_pr),
        'loc_a': float(loc_a),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_trajectory_metrics(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detailed trajectory-level metrics.
    
    Computes MT/ML/PT ratios and detailed statistics about track coverage.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - mt_ratio: Mostly Tracked ratio (MT / total_tracks)
            - ml_ratio: Mostly Lost ratio (ML / total_tracks)
            - pt_ratio: Partially Tracked ratio (PT / total_tracks)
            - mt_count: Number of mostly tracked trajectories
            - ml_count: Number of mostly lost trajectories
            - pt_count: Number of partially tracked trajectories
            - total_tracks: Total number of GT tracks
            - avg_track_coverage: Average coverage ratio across all tracks
            - avg_track_length: Average length of GT tracks (frames)
    """
    # Track coverage for each GT trajectory
    gt_track_frames_detected = defaultdict(int)
    gt_track_total_frames = defaultdict(int)
    
    for frame_id in sorted(frame_ground_truth.keys()):
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        # Count total frames for each GT track
        for gt in gts:
            gt_id = gt.get('track_id')
            if gt_id is not None:
                gt_track_total_frames[gt_id] += 1
        
        # Match and count detected frames
        if len(preds) > 0:
            matches, _, _ = _match_frame(preds, gts, iou_threshold)
            
            for _, gt_idx in matches:
                gt_id = gts[gt_idx].get('track_id')
                if gt_id is not None:
                    gt_track_frames_detected[gt_id] += 1
    
    # Calculate trajectory classification
    mt_count = 0
    pt_count = 0
    ml_count = 0
    total_coverage = 0.0
    total_length = 0
    
    for gt_id, total_frames in gt_track_total_frames.items():
        detected_frames = gt_track_frames_detected.get(gt_id, 0)
        coverage_ratio = detected_frames / total_frames if total_frames > 0 else 0.0
        
        total_coverage += coverage_ratio
        total_length += total_frames
        
        if coverage_ratio >= 0.8:
            mt_count += 1
        elif coverage_ratio >= 0.2:
            pt_count += 1
        else:
            ml_count += 1
    
    total_tracks = len(gt_track_total_frames)
    
    return {
        'mt_ratio': mt_count / total_tracks if total_tracks > 0 else 0.0,
        'ml_ratio': ml_count / total_tracks if total_tracks > 0 else 0.0,
        'pt_ratio': pt_count / total_tracks if total_tracks > 0 else 0.0,
        'mt_count': mt_count,
        'ml_count': ml_count,
        'pt_count': pt_count,
        'total_tracks': total_tracks,
        'avg_track_coverage': total_coverage / total_tracks if total_tracks > 0 else 0.0,
        'avg_track_length': total_length / total_tracks if total_tracks > 0 else 0.0
    }


def calculate_detection_metrics(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate frame-level detection metrics (Precision, Recall, F1).
    
    These are pure detection metrics without tracking/ID considerations.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - precision: Detection precision (TP / (TP + FP))
            - recall: Detection recall (TP / (TP + FN))
            - f1: Detection F1 score
            - tp: Total true positives
            - fp: Total false positives
            - fn: Total false negatives
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for frame_id in frame_ground_truth.keys():
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_smota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5,
    use_soft_matching: bool = True
) -> Dict[str, float]:
    """
    Calculate soft MOTA (sMOTA) for segmentation-based tracking.
    
    sMOTA uses IoU as a continuous similarity measure instead of binary matching.
    Useful for instance segmentation tracking (MOTS).
    
    Formula:
        sMOTA = 1 - (sum(1 - IoU) + FP + IDSW) / GT
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for considering matches
        use_soft_matching: If True, uses IoU directly; if False, uses binary matching
    
    Returns:
        Dictionary with:
            - smota: Soft MOTA score
            - soft_tp_error: Sum of (1 - IoU) for matched pairs
            - num_matches: Number of matches
            - num_false_positives: False positives
            - num_switches: ID switches
            - total_gt: Total GT objects
    
    Reference:
        "MOTS: Multi-Object Tracking and Segmentation"
        Voigtlaender et al., CVPR 2019
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    total_soft_error = 0.0
    num_matches = 0
    num_fp = 0
    num_switches = 0
    total_gt = 0
    
    # Track GT -> Pred mapping for ID switch detection
    gt_to_pred_mapping = {}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        total_gt += len(gts)
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        # Calculate soft error for matches
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if use_soft_matching:
                # Soft error: 1 - IoU (0 for perfect match, 1 for no overlap)
                total_soft_error += (1.0 - iou)
            else:
                # Binary matching: count as 0 error if IoU >= threshold
                total_soft_error += 0.0
            
            num_matches += 1
            
            # Check for ID switches
            gt_id = gt.get('track_id')
            pred_id = pred.get('track_id')
            
            if gt_id is not None and pred_id is not None:
                if gt_id in gt_to_pred_mapping:
                    if gt_to_pred_mapping[gt_id] != pred_id:
                        num_switches += 1
                        gt_to_pred_mapping[gt_id] = pred_id
                else:
                    gt_to_pred_mapping[gt_id] = pred_id
        
        num_fp += len(unmatched_preds)
    
    # Calculate soft MOTA
    if total_gt == 0:
        smota = 0.0
    else:
        smota = 1.0 - (total_soft_error + num_fp + num_switches) / total_gt
    
    return {
        'smota': float(smota),
        'soft_tp_error': float(total_soft_error),
        'num_matches': num_matches,
        'num_false_positives': num_fp,
        'num_switches': num_switches,
        'total_gt': total_gt
    }


def calculate_completeness(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate tracking completeness metrics.
    
    Measures what fraction of the ground truth is successfully tracked.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - gt_covered_ratio: Ratio of GT objects detected at least once
            - avg_gt_coverage: Average detection rate per GT object
            - frame_coverage: Ratio of frames with at least one detection
            - detection_density: Average detections per frame
    """
    # Track which GT objects have been detected
    all_gt_objects = set()
    detected_gt_objects = set()
    gt_object_detection_counts = defaultdict(int)
    gt_object_total_appearances = defaultdict(int)
    
    frames_with_detections = 0
    total_detections = 0
    num_frames = len(frame_ground_truth)
    
    for frame_id in sorted(frame_ground_truth.keys()):
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        total_detections += len(preds)
        
        if len(preds) > 0:
            frames_with_detections += 1
        
        # Track GT objects
        for gt in gts:
            gt_id = gt.get('track_id')
            if gt_id is not None:
                all_gt_objects.add(gt_id)
                gt_object_total_appearances[gt_id] += 1
        
        # Match and track detections
        if len(preds) > 0:
            matches, _, _ = _match_frame(preds, gts, iou_threshold)
            
            for _, gt_idx in matches:
                gt_id = gts[gt_idx].get('track_id')
                if gt_id is not None:
                    detected_gt_objects.add(gt_id)
                    gt_object_detection_counts[gt_id] += 1
    
    # Calculate metrics
    gt_covered_ratio = len(detected_gt_objects) / len(all_gt_objects) if len(all_gt_objects) > 0 else 0.0
    
    avg_coverage = 0.0
    if len(all_gt_objects) > 0:
        total_coverage = sum(
            gt_object_detection_counts[gt_id] / gt_object_total_appearances[gt_id]
            for gt_id in all_gt_objects
        )
        avg_coverage = total_coverage / len(all_gt_objects)
    
    frame_coverage = frames_with_detections / num_frames if num_frames > 0 else 0.0
    detection_density = total_detections / num_frames if num_frames > 0 else 0.0
    
    return {
        'gt_covered_ratio': float(gt_covered_ratio),
        'avg_gt_coverage': float(avg_coverage),
        'frame_coverage': float(frame_coverage),
        'detection_density': float(detection_density),
        'num_gt_objects': len(all_gt_objects),
        'num_detected_objects': len(detected_gt_objects)
    }


def calculate_identity_metrics(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detailed identity preservation metrics.
    
    Provides comprehensive analysis of ID switches and identity consistency.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with:
            - id_switches: Total number of ID switches
            - id_switch_rate: ID switches per GT track
            - avg_track_purity: Average purity of predicted tracks
            - avg_track_completeness: Average completeness of GT tracks
            - num_fragmentations: Number of fragmentations
            - fragmentation_rate: Fragmentations per GT track
    """
    num_switches = 0
    gt_to_pred_mapping = {}
    
    # Track purity: for each pred track, what fraction maps to the same GT?
    pred_track_to_gt_counts = defaultdict(lambda: defaultdict(int))
    
    # Track completeness: for each GT track, what fraction is in the same pred track?
    gt_track_to_pred_counts = defaultdict(lambda: defaultdict(int))
    
    for frame_id in sorted(frame_ground_truth.keys()):
        gts = frame_ground_truth[frame_id]
        preds = frame_predictions.get(frame_id, [])
        
        matches, _, _ = _match_frame(preds, gts, iou_threshold)
        
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is not None and gt_id is not None:
                # Track purity
                pred_track_to_gt_counts[pred_id][gt_id] += 1
                gt_track_to_pred_counts[gt_id][pred_id] += 1
                
                # ID switches
                if gt_id in gt_to_pred_mapping:
                    if gt_to_pred_mapping[gt_id] != pred_id:
                        num_switches += 1
                        gt_to_pred_mapping[gt_id] = pred_id
                else:
                    gt_to_pred_mapping[gt_id] = pred_id
    
    # Calculate purity (for each pred track, fraction from dominant GT)
    purity_scores = []
    for pred_id, gt_counts in pred_track_to_gt_counts.items():
        total_detections = sum(gt_counts.values())
        max_count = max(gt_counts.values())
        purity = max_count / total_detections if total_detections > 0 else 0.0
        purity_scores.append(purity)
    
    # Calculate completeness (for each GT track, fraction in dominant pred)
    completeness_scores = []
    for gt_id, pred_counts in gt_track_to_pred_counts.items():
        total_detections = sum(pred_counts.values())
        max_count = max(pred_counts.values())
        completeness = max_count / total_detections if total_detections > 0 else 0.0
        completeness_scores.append(completeness)
    
    # Count fragmentations
    num_fragmentations = _count_fragmentations(frame_predictions, frame_ground_truth, iou_threshold)
    
    num_gt_tracks = len(gt_track_to_pred_counts)
    
    return {
        'id_switches': num_switches,
        'id_switch_rate': num_switches / num_gt_tracks if num_gt_tracks > 0 else 0.0,
        'avg_track_purity': float(np.mean(purity_scores)) if purity_scores else 0.0,
        'avg_track_completeness': float(np.mean(completeness_scores)) if completeness_scores else 0.0,
        'num_fragmentations': num_fragmentations,
        'fragmentation_rate': num_fragmentations / num_gt_tracks if num_gt_tracks > 0 else 0.0
    }


def calculate_tid_lgd(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Track Initialization Duration (TID) and Longest Gap Duration (LGD).
    
    TID: Average number of frames before a GT track is first detected.
    LGD: Average longest consecutive gap in tracking for each GT track.
    
    These metrics are used in the nuScenes tracking benchmark.
    
    Args:
        frame_predictions: Dict mapping frame_id -> list of predictions
        frame_ground_truth: Dict mapping frame_id -> list of ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary containing:
            - tid: Track Initialization Duration (lower is better)
            - lgd: Longest Gap Duration (lower is better)
            - avg_initialization_frames: Average frames until first detection
            - avg_longest_gap: Average longest gap in frames
            - num_tracks: Number of GT tracks evaluated
    """
    from admetrics.detection.iou import calculate_iou_3d
    from scipy.optimize import linear_sum_assignment
    
    # Track GT track lifetimes and detection status
    gt_track_frames = defaultdict(list)  # gt_id -> list of frame_ids
    gt_track_first_detection = {}  # gt_id -> first frame detected
    gt_track_detections = defaultdict(set)  # gt_id -> set of frames where detected
    
    sorted_frames = sorted(frame_ground_truth.keys())
    
    # First pass: record all GT track appearances
    for frame_id in sorted_frames:
        gts = frame_ground_truth.get(frame_id, [])
        for gt in gts:
            gt_id = gt.get('track_id')
            if gt_id is not None:
                gt_track_frames[gt_id].append(frame_id)
    
    # Second pass: match predictions to GT and track detections
    for frame_id in sorted_frames:
        preds = frame_predictions.get(frame_id, [])
        gts = frame_ground_truth.get(frame_id, [])
        
        if not preds or not gts:
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(preds), len(gts)))
        for i, pred in enumerate(preds):
            for j, gt in enumerate(gts):
                if pred.get('class') == gt.get('class'):
                    iou_matrix[i, j] = calculate_iou_3d(pred['box'], gt['box'])
        
        # Hungarian matching
        if iou_matrix.size > 0:
            pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
            
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                    gt = gts[gt_idx]
                    gt_id = gt.get('track_id')
                    
                    if gt_id is not None:
                        gt_track_detections[gt_id].add(frame_id)
                        
                        # Record first detection
                        if gt_id not in gt_track_first_detection:
                            gt_track_first_detection[gt_id] = frame_id
    
    # Calculate TID and LGD for each track
    tid_values = []
    lgd_values = []
    
    for gt_id, all_frames in gt_track_frames.items():
        if not all_frames:
            continue
        
        all_frames_sorted = sorted(all_frames)
        first_frame = all_frames_sorted[0]
        detected_frames = sorted(gt_track_detections.get(gt_id, []))
        
        # TID: frames until first detection
        if detected_frames:
            first_detection = detected_frames[0]
            initialization_duration = first_detection - first_frame
            tid_values.append(initialization_duration)
        else:
            # Never detected - count all frames
            tid_values.append(len(all_frames))
        
        # LGD: longest consecutive gap
        if detected_frames:
            gaps = []
            detected_set = set(detected_frames)
            
            current_gap = 0
            for frame in all_frames_sorted:
                if frame in detected_set:
                    if current_gap > 0:
                        gaps.append(current_gap)
                    current_gap = 0
                else:
                    current_gap += 1
            
            # Don't count final gap (track ended)
            longest_gap = max(gaps) if gaps else 0
            lgd_values.append(longest_gap)
        else:
            # Never detected
            lgd_values.append(len(all_frames))
    
    num_tracks = len(gt_track_frames)
    
    return {
        'tid': float(np.mean(tid_values)) if tid_values else 0.0,
        'lgd': float(np.mean(lgd_values)) if lgd_values else 0.0,
        'avg_initialization_frames': float(np.mean(tid_values)) if tid_values else 0.0,
        'avg_longest_gap': float(np.mean(lgd_values)) if lgd_values else 0.0,
        'num_tracks': num_tracks,
        'num_detected_tracks': len(gt_track_first_detection)
    }


def calculate_motal(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate MOTAL (MOTA with Logarithmic ID Switches).
    
    MOTAL uses a logarithmic penalty for ID switches instead of linear,
    reducing the impact of many ID switches.
    
    Formula:
        MOTAL = 1 - (FP + FN + log₁₀(IDSW + 1)) / GT
    
    Args:
        frame_predictions: Dict mapping frame_id -> list of predictions
        frame_ground_truth: Dict mapping frame_id -> list of ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary containing:
            - motal: MOTAL score
            - mota: Standard MOTA for comparison
            - tp, fp, fn: Detection counts
            - id_switches: Number of ID switches
            - log_id_switches: log₁₀(IDSW + 1)
            - num_gt: Total ground truth detections
    """
    # First calculate standard MOTA components
    mota_result = calculate_multi_frame_mota(
        frame_predictions,
        frame_ground_truth,
        iou_threshold
    )
    
    tp = mota_result['num_matches']
    fp = mota_result['num_false_positives']
    fn = mota_result['num_misses']
    id_switches = mota_result['num_switches']
    num_gt = mota_result['total_gt']
    
    # Calculate logarithmic ID switch penalty
    log_id_switches = np.log10(id_switches + 1) if id_switches > 0 else 0.0
    
    # MOTAL formula
    if num_gt == 0:
        motal = 0.0
    else:
        motal = 1.0 - (fp + fn + log_id_switches) / num_gt
    
    return {
        'motal': float(motal),
        'mota': mota_result['mota'],
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'id_switches': id_switches,
        'log_id_switches': float(log_id_switches),
        'num_gt': num_gt
    }


def calculate_clr_metrics(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate CLEAR MOT metrics including CLR_Re, CLR_Pr, and CLR_F1.
    
    These are the official CLEAR MOT precision, recall, and F1 metrics.
    
    Formulas:
        CLR_Re = TP / (TP + FN)  # Recall
        CLR_Pr = TP / (TP + FP)  # Precision
        CLR_F1 = TP / (TP + 0.5*FN + 0.5*FP)  # Harmonic mean variant
    
    Args:
        frame_predictions: Dict mapping frame_id -> list of predictions
        frame_ground_truth: Dict mapping frame_id -> list of ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary containing:
            - clr_re: CLEAR Recall
            - clr_pr: CLEAR Precision
            - clr_f1: CLEAR F1 Score
            - tp, fp, fn: Detection counts
            - num_frames: Number of frames evaluated
    """
    from admetrics.detection.iou import calculate_iou_3d
    from scipy.optimize import linear_sum_assignment
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_frames = 0
    
    for frame_id in sorted(frame_ground_truth.keys()):
        preds = frame_predictions.get(frame_id, [])
        gts = frame_ground_truth.get(frame_id, [])
        
        num_frames += 1
        
        if not preds and not gts:
            continue
        
        if not preds:
            total_fn += len(gts)
            continue
        
        if not gts:
            total_fp += len(preds)
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(preds), len(gts)))
        for i, pred in enumerate(preds):
            for j, gt in enumerate(gts):
                if pred.get('class') == gt.get('class'):
                    iou_matrix[i, j] = calculate_iou_3d(pred['box'], gt['box'])
        
        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        
        matched_preds = set()
        matched_gts = set()
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                total_tp += 1
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
        
        total_fp += len(preds) - len(matched_preds)
        total_fn += len(gts) - len(matched_gts)
    
    # Calculate metrics
    clr_re = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    clr_pr = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    clr_f1 = total_tp / (total_tp + 0.5 * total_fn + 0.5 * total_fp) if (total_tp + 0.5 * total_fn + 0.5 * total_fp) > 0 else 0.0
    
    return {
        'clr_re': float(clr_re),
        'clr_pr': float(clr_pr),
        'clr_f1': float(clr_f1),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'num_frames': num_frames
    }


def calculate_owta(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate OWTA (Open World Tracking Accuracy).
    
    OWTA is defined as:
        OWTA = √(DetRe × AssA)
    
    This metric is used for open-world tracking scenarios where the number
    of object categories may be unknown or growing.
    
    Args:
        frame_predictions: Dict mapping frame_id -> list of predictions
        frame_ground_truth: Dict mapping frame_id -> list of ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary containing:
            - owta: Open World Tracking Accuracy
            - det_re: Detection Recall
            - ass_a: Association Accuracy
    """
    # Calculate HOTA components to get DetRe and AssA
    hota_result = calculate_hota_components(
        frame_predictions,
        frame_ground_truth,
        iou_threshold
    )
    
    det_re = hota_result['det_re']
    ass_a = hota_result['ass_a']
    
    # OWTA = sqrt(DetRe × AssA)
    owta = np.sqrt(det_re * ass_a)
    
    return {
        'owta': float(owta),
        'det_re': float(det_re),
        'ass_a': float(ass_a)
    }
