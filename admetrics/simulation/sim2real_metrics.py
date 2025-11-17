"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_perception_sim2real_gap(
    sim_detections: List[Dict],
    real_detections: List[Dict],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Measure the sim-to-real gap for perception performance.
    
    Compares object detection/tracking performance between simulation and real world.
    
    Args:
        sim_detections: Detection results from simulation, list of dicts with:
            - 'predictions': predicted boxes
            - 'ground_truth': ground truth boxes
            - 'scores': confidence scores
        real_detections: Detection results from real-world data, same format
        metrics: List of metrics to compute ['ap', 'recall', 'precision', 'latency']
        
    Returns:
        Dictionary with sim2real gap metrics:
            - ap_gap: Difference in Average Precision
            - recall_gap: Difference in recall
            - precision_gap: Difference in precision
            - performance_drop: Overall performance degradation
            
    Example:
        >>> sim_results = [{'predictions': np.random.randn(5, 7), 'ground_truth': np.random.randn(5, 7)}]
        >>> real_results = [{'predictions': np.random.randn(4, 7), 'ground_truth': np.random.randn(5, 7)}]
        >>> gap = perception_sim2real_gap(sim_results, real_results)
    """
    results = {}
    
    if metrics is None:
        metrics = ['recall', 'precision']
    
    # Calculate metrics for sim and real
    sim_tp = sim_fp = sim_fn = 0
    real_tp = real_fp = real_fn = 0
    
    iou_threshold = 0.5
    
    for det_dict in sim_detections:
        preds = det_dict.get('predictions', np.array([]))
        gts = det_dict.get('ground_truth', np.array([]))
        
        if len(preds) > 0 and len(gts) > 0:
            # Simple matching based on distance
            pred_centers = preds[:, :3] if preds.shape[1] >= 3 else preds
            gt_centers = gts[:, :3] if gts.shape[1] >= 3 else gts
            
            dist_matrix = cdist(pred_centers, gt_centers)
            
            matched_gts = set()
            for pred_idx in range(len(preds)):
                min_dist = np.min(dist_matrix[pred_idx])
                min_idx = np.argmin(dist_matrix[pred_idx])
                
                if min_dist < 2.0 and min_idx not in matched_gts:  # 2m threshold
                    sim_tp += 1
                    matched_gts.add(min_idx)
                else:
                    sim_fp += 1
            
            sim_fn += len(gts) - len(matched_gts)
        elif len(preds) > 0:
            sim_fp += len(preds)
        elif len(gts) > 0:
            sim_fn += len(gts)
    
    for det_dict in real_detections:
        preds = det_dict.get('predictions', np.array([]))
        gts = det_dict.get('ground_truth', np.array([]))
        
        if len(preds) > 0 and len(gts) > 0:
            pred_centers = preds[:, :3] if preds.shape[1] >= 3 else preds
            gt_centers = gts[:, :3] if gts.shape[1] >= 3 else gts
            
            dist_matrix = cdist(pred_centers, gt_centers)
            
            matched_gts = set()
            for pred_idx in range(len(preds)):
                min_dist = np.min(dist_matrix[pred_idx])
                min_idx = np.argmin(dist_matrix[pred_idx])
                
                if min_dist < 2.0 and min_idx not in matched_gts:
                    real_tp += 1
                    matched_gts.add(min_idx)
                else:
                    real_fp += 1
            
            real_fn += len(gts) - len(matched_gts)
        elif len(preds) > 0:
            real_fp += len(preds)
        elif len(gts) > 0:
            real_fn += len(gts)
    
    # Calculate precision and recall
    if 'precision' in metrics:
        sim_precision = sim_tp / (sim_tp + sim_fp + 1e-6)
        real_precision = real_tp / (real_tp + real_fp + 1e-6)
        
        results['precision_sim'] = float(sim_precision)
        results['precision_real'] = float(real_precision)
        results['precision_gap'] = float(sim_precision - real_precision)
    
    if 'recall' in metrics:
        sim_recall = sim_tp / (sim_tp + sim_fn + 1e-6)
        real_recall = real_tp / (real_tp + real_fn + 1e-6)
        
        results['recall_sim'] = float(sim_recall)
        results['recall_real'] = float(real_recall)
        results['recall_gap'] = float(sim_recall - real_recall)
    
    # Overall performance drop (F1-based)
    if 'precision' in metrics and 'recall' in metrics:
        sim_f1 = 2 * (sim_precision * sim_recall) / (sim_precision + sim_recall + 1e-6)
        real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall + 1e-6)
        
        results['f1_sim'] = float(sim_f1)
        results['f1_real'] = float(real_f1)
        results['f1_gap'] = float(sim_f1 - real_f1)
        results['performance_drop_pct'] = float((sim_f1 - real_f1) / (sim_f1 + 1e-6) * 100)
    
    return results


