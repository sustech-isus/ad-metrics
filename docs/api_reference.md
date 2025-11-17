# API Reference

Complete API documentation for the admetrics library - a comprehensive metrics suite for autonomous driving evaluation.

## Quick Navigation

- [Detection Metrics (24)](#detection-metrics)
- [Tracking Metrics (6)](#tracking-metrics)
- [Trajectory Prediction Metrics (10)](#trajectory-prediction-metrics)
- [Localization Metrics (8)](#localization-metrics)
- [Occupancy Metrics (6)](#occupancy-metrics)
- [Planning Metrics (20)](#planning-metrics)
- [Vector Map Metrics (8)](#vector-map-metrics)
- [Simulation Quality Metrics (7)](#simulation-quality-metrics)
- [Utility Functions (9)](#utility-functions)

---

## Detection Metrics

---

### IoU Metrics (`admetrics.detection.iou`)

#### `calculate_iou_3d(box1, box2, box_format='xyzwhlr')`

Calculate 3D Intersection over Union between two 3D bounding boxes.

**Parameters:**
- `box1` (array-like): First 3D bounding box `[x, y, z, w, h, l, rotation]`
  - `x, y, z`: Center coordinates
  - `w`: Width, `h`: Height, `l`: Length
  - `rotation`: Yaw angle in radians
- `box2` (array-like): Second 3D bounding box in same format
- `box_format` (str): Format of boxes ('xyzwhlr' or 'xyzhwlr')

**Returns:**
- `float`: IoU value between 0 and 1

**Example:**
```python
from admetrics import calculate_iou_3d

box1 = [0, 0, 0, 4, 2, 1.5, 0]
box2 = [1, 0, 0, 4, 2, 1.5, 0]
iou = calculate_iou_3d(box1, box2)
print(f"3D IoU: {iou:.4f}")
```

---

#### `calculate_iou_bev(box1, box2, box_format='xyzwhlr')`

Calculate Bird's Eye View (BEV) IoU by projecting boxes to the x-y plane.

**Parameters:**
- `box1` (array-like): First 3D bounding box
- `box2` (array-like): Second 3D bounding box
- `box_format` (str): Format of boxes

**Returns:**
- `float`: BEV IoU value between 0 and 1

---

#### `calculate_iou_batch(boxes1, boxes2, box_format='xyzwhlr', mode='3d')`

Calculate IoU for batches of boxes efficiently.

**Parameters:**
- `boxes1` (np.ndarray): Shape (N, 7) array of N bounding boxes
- `boxes2` (np.ndarray): Shape (M, 7) array of M bounding boxes
- `box_format` (str): Format of boxes
- `mode` (str): '3d' for 3D IoU, 'bev' for BEV IoU

**Returns:**
- `np.ndarray`: Shape (N, M) array of IoU values

---

#### `calculate_giou_3d(box1, box2, box_format='xyzwhlr')`

Calculate Generalized IoU (GIoU) for 3D boxes.

GIoU = IoU - |C - (A ∪ B)| / |C|, where C is the smallest enclosing box.

**Parameters:**
- `box1` (array-like): First box
- `box2` (array-like): Second box
- `box_format` (str): Format of boxes

**Returns:**
- `float`: GIoU value between -1 and 1

---

### Average Precision (`admetrics.detection.ap`)

#### `calculate_ap(predictions, ground_truth, iou_threshold=0.5, num_recall_points=40, metric_type='3d')`

Calculate Average Precision for 3D object detection.

**Parameters:**
- `predictions` (List[Dict]): List of prediction dicts with keys:
  - `'box'`: 3D bounding box [x, y, z, w, h, l, r]
  - `'score'`: Confidence score
  - `'class'`: Class name
- `ground_truth` (List[Dict]): List of ground truth dicts with keys:
  - `'box'`: 3D bounding box
  - `'class'`: Class name
  - `'difficulty'`: (optional) Difficulty level
- `iou_threshold` (float): IoU threshold for matching
- `num_recall_points` (int): Number of recall points for interpolation
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Dict`: Dictionary containing:
  - `'ap'`: Average Precision value
  - `'precision'`: Precision values
  - `'recall'`: Recall values
  - `'scores'`: Confidence scores
  - `'num_tp'`, `'num_fp'`, `'num_gt'`: Counts

**Example:**
```python
from admetrics import calculate_ap

predictions = [
    {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
    {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
]
ground_truth = [
    {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
]
result = calculate_ap(predictions, ground_truth)
print(f"AP: {result['ap']:.4f}")
```

---

#### `calculate_map(predictions, ground_truth, class_names, iou_thresholds=0.5, num_recall_points=40, metric_type='3d')`

Calculate Mean Average Precision across multiple classes and IoU thresholds.

**Parameters:**
- `predictions` (List[Dict]): All predictions
- `ground_truth` (List[Dict]): All ground truth
- `class_names` (List[str]): List of class names to evaluate
- `iou_thresholds` (float or List[float]): Single threshold or list
- `num_recall_points` (int): Number of recall points
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Dict`: Dictionary containing:
  - `'mAP'`: Overall mean AP
  - `'AP_per_class'`: AP for each class
  - `'AP_per_threshold'`: AP for each IoU threshold
  - `'num_classes'`: Number of evaluated classes

---

#### `calculate_ap_coco_style(predictions, ground_truth, iou_thresholds=None)`

Calculate AP using COCO-style evaluation (average over IoU thresholds 0.5:0.95).

**Returns:**
- `Dict`: Dictionary with:
  - `'AP'`: Average over [0.5:0.95]
  - `'AP50'`: AP at IoU=0.5
  - `'AP75'`: AP at IoU=0.75
  - `'AP_per_threshold'`: AP for each threshold

---

#### `calculate_precision_recall_curve(predictions, ground_truth, iou_threshold=0.5)`

Calculate precision-recall curve data.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with:
  - `'precision'`: Precision array
  - `'recall'`: Recall array
  - `'scores'`: Confidence scores
  - `'tp'`: True positive array
  - `'fp'`: False positive array

---

### NuScenes Detection Score (`admetrics.detection.nds`)

#### `calculate_nds(predictions, ground_truth, class_names, iou_threshold=0.5, distance_thresholds=None)`

Calculate nuScenes Detection Score (NDS).

NDS combines mAP with error metrics: Translation, Scale, Orientation, Velocity, and Attribute errors.

**Parameters:**
- `predictions` (List[Dict]): Predictions with keys:
  - `'box'`: [x, y, z, w, h, l, yaw]
  - `'score'`: Confidence score
  - `'class'`: Class name
  - `'velocity'`: (optional) [vx, vy]
  - `'attributes'`: (optional) Attribute predictions
- `ground_truth` (List[Dict]): Ground truth annotations
- `class_names` (List[str]): List of class names
- `iou_threshold` (float): IoU threshold
- `distance_thresholds` (List[float], optional): Distance thresholds per class

**Returns:**
- `float`: NDS score (0-1)

---

#### `calculate_nds_detailed(predictions, ground_truth, class_names, iou_threshold=0.5)`

Calculate NDS with detailed breakdown.

**Returns:**
- `Dict`: Dictionary containing:
  - `'nds'`: Overall NDS score
  - `'mAP'`: Mean Average Precision
  - `'tp_metrics'`: Dict of TP error metrics (ATE, ASE, AOE, AVE, AAE)
  - `'per_class_nds'`: NDS for each class
  - `'AP_per_class'`: AP for each class

---

#### `calculate_tp_metrics(predictions, ground_truth, iou_threshold=0.5)`

Calculate True Positive error metrics (ATE, ASE, AOE, AVE, AAE).

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with:
  - `'ate'`: Average Translation Error
  - `'ase'`: Average Scale Error
  - `'aoe'`: Average Orientation Error
  - `'ave'`: Average Velocity Error
  - `'aae'`: Average Attribute Error

---

### Average Orientation Similarity (`admetrics.detection.aos`)

#### `calculate_aos(predictions, ground_truth, iou_threshold=0.7, num_recall_points=40)`

Calculate Average Orientation Similarity (AOS) for KITTI benchmark.

AOS combines detection accuracy with orientation estimation quality.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold (KITTI uses 0.7 for cars)
- `num_recall_points` (int): Number of recall points

**Returns:**
- `Dict`: Dictionary with:
  - `'aos'`: Average Orientation Similarity
  - `'ap'`: Average Precision
  - `'orientation_similarity'`: Mean orientation similarity for TPs
  - `'num_tp'`, `'num_fp'`, `'num_gt'`: Counts

---

#### `calculate_aos_per_difficulty(predictions, ground_truth, iou_threshold=0.7)`

Calculate AOS per difficulty level (easy, moderate, hard).

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth with 'difficulty' key
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with AOS for 'easy', 'moderate', 'hard'

---

#### `calculate_orientation_similarity(box1, box2)`

Calculate orientation similarity between two boxes.

**Parameters:**
- `box1`, `box2`: Boxes [x, y, z, w, h, l, yaw]

**Returns:**
- `float`: Orientation similarity [0, 1]

---

### Confusion Metrics (`admetrics.detection.confusion`)

#### `calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Calculate True Positives, False Positives, and False Negatives.

**Returns:**
- `Dict`: Dictionary with `'tp'`, `'fp'`, `'fn'` counts

---

#### `calculate_confusion_metrics(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Calculate comprehensive confusion matrix metrics.

**Returns:**
- `Dict`: Dictionary with:
  - `'precision'`: TP / (TP + FP)
  - `'recall'`: TP / (TP + FN)
  - `'f1_score'`: 2 * (precision * recall) / (precision + recall)
  - `'tp'`, `'fp'`, `'fn'`: Raw counts

---

#### `calculate_confusion_matrix_multiclass(predictions, ground_truth, class_names, iou_threshold=0.5, metric_type='3d')`

Calculate confusion matrix for multi-class detection.

**Returns:**
- `Dict`: Dictionary with:
  - `'confusion_matrix'`: (N, N) numpy array
  - `'class_names'`: List of class names
  - `'per_class_metrics'`: Metrics for each class

---

#### `calculate_specificity(predictions, ground_truth, iou_threshold=0.5)`

Calculate specificity (true negative rate).

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold

**Returns:**
- `float`: Specificity [0, 1]

---

### Distance Metrics (`admetrics.detection.distance`)

#### `calculate_center_distance(box1, box2, distance_type='euclidean')`

Calculate distance between centers of two 3D bounding boxes.

**Parameters:**
- `box1`, `box2`: Boxes [x, y, z, w, h, l, r]
- `distance_type` (str): 'euclidean' (3D), 'bev' (2D), or 'vertical' (z-axis)

**Returns:**
- `float`: Distance in meters

---

#### `calculate_orientation_error(box1, box2, error_type='absolute')`

Calculate orientation/heading error.

**Parameters:**
- `error_type` (str): 'absolute' (radians) or 'degrees'

**Returns:**
- `float`: Orientation error

---

#### `calculate_size_error(box1, box2, box_format='xyzwhlr', error_type='absolute')`

Calculate size/dimension errors between two boxes.

**Parameters:**
- `box1`, `box2`: Boxes [x, y, z, w, h, l, r]
- `box_format` (str): Box format
- `error_type` (str): 'absolute' or 'relative'

**Returns:**
- `Dict`: Dictionary with:
  - `'width_error'`: Width error
  - `'height_error'`: Height error
  - `'length_error'`: Length error
  - `'volume_error'`: Volume error

---

#### `calculate_velocity_error(pred_velocity, gt_velocity)`

Calculate velocity estimation error.

**Parameters:**
- `pred_velocity` (np.ndarray): [vx, vy] or [vx, vy, vz]
- `gt_velocity` (np.ndarray): [vx, vy] or [vx, vy, vz]

**Returns:**
- `Dict`: Dictionary with:
  - `'error'`: Absolute error
  - `'relative_error'`: Relative error
  - `'angle_error'`: Angle error

---

#### `calculate_average_distance_error(predictions, ground_truth, iou_threshold=0.5)`

Calculate average distance errors across multiple detections.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with 'mean', 'std', 'min', 'max'

---

#### `calculate_translation_error_bins(predictions, ground_truth, distance_bins=[0, 20, 40, 60, 80, 100], iou_threshold=0.5)`

Calculate translation errors binned by distance ranges.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `distance_bins` (List[float]): Distance bin edges
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Error for each distance bin (e.g., '0-20m', '20-40m')

---

## Tracking Metrics

### Multi-Object Tracking (`admetrics.tracking`)

#### `calculate_mota(predictions, ground_truth, iou_threshold=0.5)`

Calculate Multiple Object Tracking Accuracy.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with:
  - `'mota'`: MOTA score
  - `'tp'`, `'fp'`, `'fn'`: Counts
  - `'id_switches'`: ID switch count
  - `'num_gt'`: Total ground truth

---

#### `calculate_motp(predictions, ground_truth, iou_threshold=0.5, distance_type='euclidean')`

Calculate Multiple Object Tracking Precision.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold
- `distance_type` (str): Distance metric type

**Returns:**
- `Dict`: Dictionary with:
  - `'motp'`: MOTP score
  - `'mean_distance'`: Mean distance
  - `'num_tp'`: Number of TPs

---

#### `calculate_clearmot_metrics(predictions, ground_truth, iou_threshold=0.5)`

Calculate complete CLEAR MOT metrics suite.

**Returns:**
- `Dict`: Dictionary with MOTA, MOTP, precision, recall, and counts

---

#### `calculate_multi_frame_mota(frame_predictions, frame_ground_truth, iou_threshold=0.5)`

Calculate MOTA across multiple frames with ID tracking.

**Parameters:**
- `frame_predictions` (List[List[Dict]]): Predictions per frame
- `frame_ground_truth` (List[List[Dict]]): Ground truth per frame
- `iou_threshold` (float): IoU threshold

**Returns:**
- `Dict`: Dictionary with MOTA, MOTP, ID switches, fragmentations

---

#### `calculate_hota(frame_predictions, frame_ground_truth, iou_threshold=0.5)`

Calculate Higher Order Tracking Accuracy.

**Returns:**
- `Dict`: Dictionary with:
  - `'hota'`: HOTA score
  - `'deta'`: Detection accuracy
  - `'assa'`: Association accuracy
  - `'localization'`: Localization accuracy

---

#### `calculate_id_f1(frame_predictions, frame_ground_truth, iou_threshold=0.5)`

Calculate ID F1-Score (ID precision and recall).

**Returns:**
- `Dict`: Dictionary with:
  - `'idf1'`: IDF1 score
  - `'idp'`: ID precision
  - `'idr'`: ID recall

---

## Trajectory Prediction Metrics

### Motion Forecasting (`admetrics.prediction`)

#### `calculate_ade(predictions, ground_truth)`

Calculate Average Displacement Error.

**Parameters:**
- `predictions` (np.ndarray): Predicted trajectory (T, 2) or (T, 3)
- `ground_truth` (np.ndarray): Ground truth trajectory (T, 2) or (T, 3)

**Returns:**
- `float`: ADE in meters

**Example:**
```python
from admetrics import calculate_ade
import numpy as np

pred = np.array([[0, 0], [1, 1], [2, 2]])
gt = np.array([[0, 0], [1.1, 0.9], [2.2, 1.8]])
ade = calculate_ade(pred, gt)
print(f"ADE: {ade:.3f}m")
```

---

#### `calculate_fde(predictions, ground_truth)`

Calculate Final Displacement Error.

**Parameters:**
- `predictions` (np.ndarray): Predicted trajectory (T, 2) or (T, 3)
- `ground_truth` (np.ndarray): Ground truth trajectory (T, 2) or (T, 3)

**Returns:**
- `float`: FDE in meters

---

#### `calculate_miss_rate(predictions, ground_truth, threshold=2.0)`

Calculate Miss Rate (fraction exceeding threshold).

**Parameters:**
- `predictions` (np.ndarray): Predicted trajectory
- `ground_truth` (np.ndarray): Ground truth trajectory
- `threshold` (float): Distance threshold for miss

**Returns:**
- `Dict`: Dictionary with:
  - `'miss_rate'`: Miss rate [0, 1]
  - `'is_miss'`: Boolean if prediction is miss
  - `'fde'`: Final displacement error

---

#### `calculate_multimodal_ade(predictions, ground_truth, k=1)`

Calculate minimum ADE across multiple trajectory modes.

**Parameters:**
- `predictions` (List[np.ndarray] or np.ndarray): Multiple trajectory predictions (K, T, 2)
- `ground_truth` (np.ndarray): Ground truth trajectory (T, 2)
- `k` (int): Top-k modes to consider

**Returns:**
- `Dict`: Dictionary with:
  - `'min_ade'`: Minimum ADE
  - `'top_k_ade'`: Top-k average ADE
  - `'best_mode_idx'`: Index of best mode

---

#### `calculate_multimodal_fde(predictions, ground_truth, k=1)`

Calculate minimum FDE across multiple trajectory modes.

**Parameters:**
- `predictions` (List[np.ndarray] or np.ndarray): Multiple trajectory predictions (K, T, 2)
- `ground_truth` (np.ndarray): Ground truth trajectory (T, 2)
- `k` (int): Top-k modes to consider

**Returns:**
- `Dict`: Dictionary with:
  - `'min_fde'`: Minimum FDE
  - `'top_k_fde'`: Top-k average FDE
  - `'best_mode_idx'`: Index of best mode

---

#### `calculate_brier_fde(predictions, ground_truth, probabilities)`

Calculate Brier-FDE (probabilistic metric).

**Parameters:**
- `predictions` (List[np.ndarray] or np.ndarray): Multiple predictions (K, T, 2)
- `ground_truth` (np.ndarray): Ground truth (T, 2)
- `probabilities` (np.ndarray): Mode probabilities (K,)

**Returns:**
- `Dict`: Dictionary with:
  - `'brier_fde'`: Brier-FDE score
  - `'fde_values'`: FDE per mode
  - `'weighted_fde'`: Probability-weighted FDE

---

#### `calculate_nll(ground_truth, predicted_means, predicted_covariances)`

Calculate Negative Log-Likelihood for probabilistic predictions.

**Parameters:**
- `ground_truth` (np.ndarray): Ground truth (T, 2)
- `predicted_means` (np.ndarray): Predicted means (T, 2)
- `predicted_covariances` (np.ndarray): Predicted covariances (T, 2, 2)

**Returns:**
- `float`: NLL value

---

#### `calculate_trajectory_metrics(predictions, ground_truth, metrics=None)`

Calculate comprehensive trajectory metrics.

**Parameters:**
- `predictions` (np.ndarray): Predicted trajectory
- `ground_truth` (np.ndarray): Ground truth trajectory
- `metrics` (List[str], optional): Metrics to compute

**Returns:**
- `Dict`: Dictionary with ADE, FDE, Miss Rate, and more

---

#### `calculate_collision_rate(trajectory, obstacles, vehicle_size=(4.5, 2.0), safety_margin=0.0)`

Calculate collision rate with obstacles.

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory (T, 2)
- `obstacles` (List[np.ndarray]): Obstacle trajectories
- `vehicle_size` (Tuple): (length, width) in meters
- `safety_margin` (float): Additional safety buffer

**Returns:**
- `Dict`: Dictionary with:
  - `'collision_rate'`: Fraction of timesteps with collision
  - `'num_collisions'`: Total collisions
  - `'collision_timesteps'`: List of collision timesteps
  - `'first_collision'`: First collision timestep

---

#### `calculate_drivable_area_compliance(trajectory, drivable_area, vehicle_size=(4.5, 2.0))`

Calculate compliance with drivable area.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory (T, 2)
- `drivable_area` (np.ndarray or polygon): Drivable area
- `vehicle_size` (Tuple): Vehicle dimensions

**Returns:**
- `Dict`: Dictionary with:
  - `'compliance_rate'`: Compliance rate
  - `'violation_timesteps'`: Violation timesteps
  - `'num_violations'`: Total violations

---

## Localization Metrics

### Ego Pose Estimation (`admetrics.localization`)

#### `calculate_ate(predicted_poses, ground_truth_poses, align=False)`

Calculate Absolute Trajectory Error.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 3) or (N, 7)
- `ground_truth_poses` (np.ndarray): Ground truth poses (N, 3) or (N, 7)
- `align` (bool): Apply Umeyama alignment

**Returns:**
- `Dict`: Dictionary with 'mean', 'std', 'min', 'max', 'rmse', 'median'

---

#### `calculate_rte(predicted_poses, ground_truth_poses, distances=[100, 200, 300, 400, 500, 600, 700, 800])`

Calculate Relative Trajectory Error (drift over distance).

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses
- `ground_truth_poses` (np.ndarray): Ground truth poses
- `distances` (List[float]): Distance thresholds in meters

**Returns:**
- `Dict`: RTE for each distance (e.g., 'rte_100', 'rte_200')

---

#### `calculate_are(predicted_poses, ground_truth_poses)`

Calculate Absolute Rotation Error.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 7) with quaternions
- `ground_truth_poses` (np.ndarray): Ground truth poses (N, 7)

**Returns:**
- `Dict`: Dictionary with 'mean', 'std', 'min', 'max', 'rmse'

---

#### `calculate_lateral_error(predicted_poses, reference_path)`

Calculate lateral deviation from reference path.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 2) or (N, 3)
- `reference_path` (np.ndarray): Reference path (M, 2) or (M, 3)

**Returns:**
- `Dict`: Dictionary with 'mean', 'std', 'max', 'percentile_95'

---

#### `calculate_longitudinal_error(predicted_poses, reference_path)`

Calculate longitudinal error along path direction.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses
- `reference_path` (np.ndarray): Reference path

**Returns:**
- `Dict`: Dictionary with 'mean', 'std', 'max'

---

#### `calculate_convergence_rate(predicted_poses, ground_truth_poses, threshold=0.5)`

Calculate convergence rate to ground truth.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 3)
- `ground_truth_poses` (np.ndarray): Ground truth poses (N, 3)
- `threshold` (float): Convergence threshold in meters

**Returns:**
- `Dict`: Dictionary with:
  - `'convergence_rate'`: Convergence rate
  - `'convergence_time'`: Time to converge
  - `'converged'`: Boolean

---

#### `calculate_localization_metrics(predicted_poses, ground_truth_poses, metrics=None)`

Calculate comprehensive localization metrics.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 7)
- `ground_truth_poses` (np.ndarray): Ground truth poses (N, 7)
- `metrics` (List[str], optional): Metrics to compute

**Returns:**
- `Dict`: Dictionary with ATE, RTE, ARE, and more

---

#### `calculate_map_alignment_score(predicted_poses, map_lanes, max_distance=2.0)`

Calculate alignment score with HD map.

**Parameters:**
- `predicted_poses` (np.ndarray): Predicted poses (N, 3)
- `map_lanes` (List[np.ndarray]): HD map lane polylines
- `max_distance` (float): Maximum matching distance

**Returns:**
- `Dict`: Dictionary with:
  - `'alignment_score'`: Alignment score
  - `'mean_distance'`: Mean distance to lanes
  - `'aligned_percentage'`: Percentage of aligned poses

---

## Occupancy Metrics

### 3D Occupancy Prediction (`admetrics.occupancy`)

#### `calculate_occupancy_iou(pred_occupancy, gt_occupancy, class_id=None, ignore_index=255)`

Calculate IoU for 3D occupancy predictions.

**Parameters:**
- `pred_occupancy` (np.ndarray): Predicted occupancy grid (X, Y, Z)
- `gt_occupancy` (np.ndarray): Ground truth occupancy grid (X, Y, Z)
- `class_id` (int, optional): Specific class ID
- `ignore_index` (int): Label value to ignore

**Returns:**
- `float`: IoU [0, 1]

---

#### `calculate_mean_iou(pred_occupancy, gt_occupancy, num_classes, ignore_index=255, ignore_classes=None)`

Calculate mean IoU across all classes.

**Parameters:**
- `pred_occupancy` (np.ndarray): Predicted occupancy grid
- `gt_occupancy` (np.ndarray): Ground truth occupancy grid
- `num_classes` (int): Total number of classes
- `ignore_index` (int): Label to ignore
- `ignore_classes` (List[int], optional): Classes to exclude

**Returns:**
- `Dict`: Dictionary with:
  - `'mIoU'`: Mean IoU
  - `'class_iou'`: Per-class IoU dict
  - `'valid_classes'`: Number of valid classes

---

#### `calculate_occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=None, ignore_index=255)`

Calculate precision and recall for occupancy.

**Parameters:**
- `pred_occupancy` (np.ndarray): Predicted occupancy
- `gt_occupancy` (np.ndarray): Ground truth occupancy
- `class_id` (int, optional): Specific class ID
- `ignore_index` (int): Label to ignore

**Returns:**
- `Dict`: Dictionary with 'precision', 'recall', 'f1'

---

#### `calculate_scene_completion(pred_occupancy, gt_occupancy, num_classes)`

Calculate scene completion metrics.

**Parameters:**
- `pred_occupancy` (np.ndarray): Predicted occupancy
- `gt_occupancy` (np.ndarray): Ground truth occupancy
- `num_classes` (int): Number of classes

**Returns:**
- `Dict`: Dictionary with:
  - `'sc_iou'`: Scene completion IoU
  - `'sc_precision'`: Scene completion precision
  - `'sc_recall'`: Scene completion recall
  - `'completion_ratio'`: Completion ratio

---

#### `calculate_chamfer_distance(pred_points, gt_points, bidirectional=True)`

Calculate Chamfer distance between point clouds.

**Parameters:**
- `pred_points` (np.ndarray): Predicted points (N, 3)
- `gt_points` (np.ndarray): Ground truth points (M, 3)
- `bidirectional` (bool): Use bidirectional distance

**Returns:**
- `Dict`: Dictionary with:
  - `'chamfer_distance'`: Chamfer distance
  - `'pred_to_gt'`: Pred->GT distance
  - `'gt_to_pred'`: GT->Pred distance

---

#### `calculate_surface_distance(pred_occupancy, gt_occupancy, voxel_size=0.2)`

Calculate surface distance metrics.

**Parameters:**
- `pred_occupancy` (np.ndarray): Predicted occupancy
- `gt_occupancy` (np.ndarray): Ground truth occupancy
- `voxel_size` (float): Voxel size in meters

**Returns:**
- `Dict`: Dictionary with:
  - `'mean_distance'`: Mean surface distance
  - `'median_distance'`: Median surface distance
  - `'hausdorff_distance'`: Hausdorff distance

---

## Planning Metrics

### End-to-End Planning (`admetrics.planning`)

#### `calculate_l2_distance(predicted_trajectory, expert_trajectory, weights=None)`

Calculate L2 distance between predicted and expert trajectories.

**Parameters:**
- `predicted_trajectory` (np.ndarray): Predicted trajectory (T, 2) or (T, 3)
- `expert_trajectory` (np.ndarray): Expert trajectory (T, 2) or (T, 3)
- `weights` (np.ndarray, optional): Temporal weights (T,)

**Returns:**
- `float`: Average L2 distance

**Example:**
```python
from admetrics.planning import calculate_l2_distance

pred = np.array([[0, 0], [1, 0], [2, 0]])
expert = np.array([[0, 0], [1, 0.5], [2, 0.5]])
dist = calculate_l2_distance(pred, expert)
```

---

#### `calculate_collision_rate(trajectory, obstacles, vehicle_size=(4.5, 2.0), obstacle_sizes=None, safety_margin=0.0)`

Calculate collision rate for planned trajectory.

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory (T, 2)
- `obstacles` (List[np.ndarray]): List of obstacle trajectories
- `vehicle_size` (Tuple): Ego vehicle (length, width) in meters
- `obstacle_sizes` (List[Tuple], optional): Obstacle sizes
- `safety_margin` (float): Safety buffer in meters

**Returns:**
- `Dict`: Dictionary containing:
  - `'collision_rate'`: Fraction of timesteps with collisions
  - `'num_collisions'`: Total number of collisions
  - `'first_collision'`: Index of first collision (or None)

---

#### `calculate_collision_with_fault_classification(ego_trajectory, ego_velocities, ego_headings, other_vehicles, vehicle_size=(4.5, 2.0), collision_threshold=0.5)`

**NEW**: Calculate collisions with at-fault classification (nuPlan/tuplan_garage style).

**Parameters:**
- `ego_trajectory` (np.ndarray): Ego trajectory positions (T, 2)
- `ego_velocities` (np.ndarray): Ego velocities (T,) in m/s
- `ego_headings` (np.ndarray): Ego heading angles (T,) in radians
- `other_vehicles` (List[np.ndarray]): Other vehicle trajectories
- `vehicle_size` (Tuple): Vehicle (length, width)
- `collision_threshold` (float): Collision distance threshold

**Returns:**
- `Dict`: Dictionary containing:
  - `'total_collisions'`: Total collision count
  - `'at_fault_collisions'`: Number of at-fault collisions
  - `'collision_rate'`: Fraction of timesteps with collisions
  - `'collision_types'`: Counter of collision types
    - `'active_front'`: Ego rear-ends another vehicle (at-fault)
    - `'stopped_track'`: Ego hits stopped/slow object (at-fault)
    - `'active_lateral'`: Ego sideswipes another vehicle (at-fault)
    - `'active_rear'`: Ego is rear-ended (not at-fault)
    - `'passive'`: Other passive collisions (not at-fault)

**Example:**
```python
result = calculate_collision_with_fault_classification(
    ego_traj, ego_vels, ego_heads, other_vehicles
)
print(f"At-fault: {result['at_fault_collisions']}")
```

---

#### `calculate_progress_score(trajectory, route, distance_threshold=2.0)`

Calculate progress along planned route.

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory (T, 2)
- `route` (np.ndarray): Route waypoints (N, 2)
- `distance_threshold` (float): Threshold for progress detection

**Returns:**
- `Dict`: Dictionary with:
  - `'progress_score'`: Normalized progress score [0, 1]
  - `'distance_traveled'`: Total distance traveled
  - `'progress_ratio'`: Ratio of route completed

---

#### `calculate_route_completion(trajectory, route_waypoints, completion_threshold=5.0)`

Calculate route completion percentage.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory (T, 2)
- `route_waypoints` (List[np.ndarray]): Ordered route waypoints
- `completion_threshold` (float): Distance threshold for waypoint completion

**Returns:**
- `Dict`: Dictionary with:
  - `'completion_rate'`: Completion rate [0, 1]
  - `'completed_waypoints'`: Number of completed waypoints
  - `'total_waypoints'`: Total number of waypoints

---

#### `average_displacement_error_planning(predicted_trajectory, expert_trajectory, horizons=None)`

Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE) for planning.

**Parameters:**
- `predicted_trajectory` (np.ndarray): Predicted trajectory (T, 2)
- `expert_trajectory` (np.ndarray): Expert trajectory (T, 2)
- `horizons` (List[int], optional): Specific time horizons for multi-horizon evaluation

**Returns:**
- `Dict`: Dictionary with:
  - `'ADE'`: Average displacement error over full trajectory
  - `'FDE'`: Final displacement error
  - `'ADE_H'`: ADE at horizon H (if horizons specified)
  - `'FDE_H'`: FDE at horizon H (if horizons specified)

**Example:**
```python
# Multi-horizon evaluation (nuPlan/Waymo style)
result = average_displacement_error_planning(
    pred, expert, horizons=[10, 30, 50, 80]  # 1s, 3s, 5s, 8s at 10Hz
)
print(f"1s ADE: {result['ADE_10']:.3f}m")
print(f"8s FDE: {result['FDE_80']:.3f}m")
```

---

#### `calculate_lateral_deviation(trajectory, reference_path)`

Calculate lateral deviation from reference path.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory (T, 2)
- `reference_path` (np.ndarray): Reference path (N, 2)

**Returns:**
- `Dict`: Dictionary with:
  - `'mean_deviation'`: Mean lateral deviation
  - `'std_deviation'`: Standard deviation
  - `'max_deviation'`: Maximum deviation

---

#### `calculate_heading_error(predicted_headings, reference_headings)`

Calculate heading angle error with proper angle wrapping.

**Parameters:**
- `predicted_headings` (np.ndarray): Predicted headings in radians (T,)
- `reference_headings` (np.ndarray): Reference headings in radians (T,)

**Returns:**
- `Dict`: Dictionary with:
  - `'mean_error'`: Mean heading error
  - `'std_error'`: Standard deviation
  - `'max_error'`: Maximum error

---

#### `calculate_velocity_error(predicted_velocities, reference_velocities)`

Calculate velocity error metrics.

**Parameters:**
- `predicted_velocities` (np.ndarray): Predicted velocities (T,)
- `reference_velocities` (np.ndarray): Reference velocities (T,)

**Returns:**
- `Dict`: Dictionary with:
  - `'mean_error'`: Mean velocity error
  - `'std_error'`: Standard deviation
  - `'max_error'`: Maximum error
  - `'rmse'`: Root mean square error

---

#### `calculate_comfort_metrics(trajectory, timestamps, max_longitudinal_accel=4.0, max_lateral_accel=4.0, max_jerk=4.0, max_yaw_rate=1.0, max_yaw_accel=1.0, include_lateral=True, use_smoothing=False, smoothing_window=15, smoothing_order=2)`

**UPDATED**: Calculate comprehensive comfort metrics with optional Savitzky-Golay smoothing (nuPlan/tuplan_garage style).

**Parameters:**
- `trajectory` (np.ndarray): Trajectory positions (T, 2) for [x, y]
- `timestamps` (np.ndarray): Time at each position (T,)
- `max_longitudinal_accel` (float): Comfort threshold for longitudinal acceleration (m/s²), default 4.0
- `max_lateral_accel` (float): Comfort threshold for lateral acceleration (m/s²), default 4.0
- `max_jerk` (float): Comfort threshold for jerk (m/s³), default 4.0
- `max_yaw_rate` (float): Comfort threshold for yaw rate (rad/s), default 1.0
- `max_yaw_accel` (float): Comfort threshold for yaw acceleration (rad/s²), default 1.0
- `include_lateral` (bool): Whether to compute lateral metrics, default True
- `use_smoothing` (bool): Whether to use Savitzky-Golay filter for smoother derivatives, default False
- `smoothing_window` (int): Window length for filter (must be odd), default 15
- `smoothing_order` (int): Polynomial order for filter, default 2

**Returns:**
- `Dict`: Dictionary containing:
  - `'mean_longitudinal_accel'`: Average longitudinal acceleration magnitude
  - `'max_longitudinal_accel'`: Maximum longitudinal acceleration
  - `'mean_lateral_accel'`: Average lateral acceleration magnitude (if include_lateral)
  - `'max_lateral_accel'`: Maximum lateral acceleration (if include_lateral)
  - `'mean_jerk'`: Average jerk magnitude
  - `'max_jerk'`: Maximum jerk
  - `'mean_yaw_rate'`: Average yaw rate (if include_lateral)
  - `'max_yaw_rate'`: Maximum yaw rate (if include_lateral)
  - `'mean_yaw_accel'`: Average yaw acceleration (if include_lateral)
  - `'max_yaw_accel'`: Maximum yaw acceleration (if include_lateral)
  - `'comfort_violations'`: Number of timesteps exceeding thresholds
  - `'comfort_rate'`: Fraction of time within comfort limits [0, 1]

**Example:**
```python
# Standard comfort metrics
result = calculate_comfort_metrics(traj, timestamps)
print(f"Comfort rate: {result['comfort_rate']*100:.1f}%")

# With smoothing for noisy GPS data
result = calculate_comfort_metrics(
    traj, timestamps, 
    use_smoothing=True,
    smoothing_window=15
)
```

---

#### `calculate_driving_score(predicted_trajectory, expert_trajectory, timestamps, obstacles=None, route=None, weights=None, mode='default')`

**UPDATED**: Calculate composite driving score with optional nuPlan mode.

**Parameters:**
- `predicted_trajectory` (np.ndarray): Predicted trajectory (T, 2)
- `expert_trajectory` (np.ndarray): Expert trajectory (T, 2)
- `timestamps` (np.ndarray): Timestamps (T,)
- `obstacles` (List[np.ndarray], optional): Obstacles for safety scoring
- `route` (np.ndarray, optional): Route waypoints for progress scoring
- `weights` (Dict, optional): Component weights (default: equal weights)
- `mode` (str): Scoring mode - 'default' or 'nuplan'

**Returns:**
- `Dict`: Dictionary with:
  - `'driving_score'`: Overall composite score [0, 100]
  - `'planning_accuracy'`: Trajectory following accuracy
  - `'safety_score'`: Collision avoidance score
  - `'progress_score'`: Route progress score
  - `'comfort_score'`: Comfort/smoothness score
  - Additional nuPlan-specific scores if mode='nuplan'

**Example:**
```python
# nuPlan evaluation mode
result = calculate_driving_score(
    pred_traj, expert_traj, timestamps,
    obstacles=obstacles,
    mode='nuplan'
)
```

---

#### `calculate_planning_kl_divergence(predicted_distribution, expert_distribution)`

Calculate KL divergence between trajectory distributions.

**Parameters:**
- `predicted_distribution` (np.ndarray): Predicted distribution
- `expert_distribution` (np.ndarray): Expert distribution

**Returns:**
- `float`: KL divergence value

---

#### `calculate_time_to_collision(ego_trajectory, ego_velocity, obstacles, vehicle_size=(4.5, 2.0))`

Calculate basic Time-to-Collision (TTC) metric.

**Parameters:**
- `ego_trajectory` (np.ndarray): Ego trajectory (T, 2)
- `ego_velocity` (np.ndarray): Ego velocities (T,)
- `obstacles` (List[np.ndarray]): Obstacle trajectories
- `vehicle_size` (Tuple): Vehicle dimensions

**Returns:**
- `Dict`: Dictionary with:
  - `'min_ttc'`: Minimum TTC across all timesteps
  - `'ttc_violations'`: Count of dangerous TTC situations

---

#### `calculate_time_to_collision_enhanced(ego_trajectory, ego_velocities, ego_headings, timestamps, other_vehicles, projection_horizon=1.0, projection_dt=0.3, ttc_threshold=3.0, vehicle_size=(4.5, 2.0), stopped_velocity_threshold=0.005)`

**NEW**: Calculate enhanced TTC with forward trajectory projection (nuPlan style).

**Parameters:**
- `ego_trajectory` (np.ndarray): Ego positions (T, 2)
- `ego_velocities` (np.ndarray): Ego velocities (T,) in m/s
- `ego_headings` (np.ndarray): Ego heading angles (T,) in radians
- `timestamps` (np.ndarray): Timestamps (T,)
- `other_vehicles` (List[np.ndarray]): Other vehicle trajectories
- `projection_horizon` (float): Projection time horizon in seconds, default 1.0
- `projection_dt` (float): Projection time step in seconds, default 0.3
- `ttc_threshold` (float): TTC threshold for violations in seconds, default 3.0
- `vehicle_size` (Tuple): Vehicle (length, width)
- `stopped_velocity_threshold` (float): Velocity threshold for stopped vehicles

**Returns:**
- `Dict`: Dictionary with:
  - `'min_ttc'`: Minimum TTC across all timesteps and projections
  - `'mean_ttc'`: Mean TTC value
  - `'ttc_violations'`: Number of timesteps with TTC < threshold
  - `'ttc_profile'`: Full TTC time series

**Example:**
```python
result = calculate_time_to_collision_enhanced(
    ego_traj, ego_vels, ego_heads, timestamps, other_vehicles,
    projection_horizon=1.0,
    projection_dt=0.3
)
```

---

#### `calculate_lane_invasion_rate(trajectory, lane_boundaries, vehicle_width=2.0)`

Calculate lane invasion/departure rate.

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory (T, 2)
- `lane_boundaries` (Tuple[np.ndarray, np.ndarray]): Left and right lane boundaries
- `vehicle_width` (float): Vehicle width

**Returns:**
- `Dict`: Dictionary with:
  - `'invasion_rate'`: Fraction of time outside lane
  - `'num_invasions'`: Count of invasions
  - `'max_invasion_distance'`: Maximum invasion distance

---

#### `calculate_collision_severity(trajectory, obstacles, vehicle_size=(4.5, 2.0), velocities=None)`

Calculate collision severity based on impact dynamics.

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory (T, 2)
- `obstacles` (List[np.ndarray]): Obstacle trajectories
- `vehicle_size` (Tuple): Vehicle dimensions
- `velocities` (np.ndarray, optional): Ego velocities for impact severity

**Returns:**
- `Dict`: Dictionary with:
  - `'max_severity'`: Maximum collision severity
  - `'severities'`: List of all collision severities

---

#### `check_kinematic_feasibility(trajectory, timestamps, max_velocity=15.0, max_acceleration=4.0, max_lateral_accel=4.0, max_yaw_rate=0.5)`

Check if trajectory is kinematically feasible.

**Parameters:**
- `trajectory` (np.ndarray): Trajectory (T, 2)
- `timestamps` (np.ndarray): Timestamps (T,)
- `max_velocity` (float): Maximum velocity (m/s)
- `max_acceleration` (float): Maximum acceleration (m/s²)
- `max_lateral_accel` (float): Maximum lateral acceleration (m/s²)
- `max_yaw_rate` (float): Maximum yaw rate (rad/s)

**Returns:**
- `Dict`: Dictionary with:
  - `'feasible'`: Boolean feasibility status
  - `'max_velocity'`: Maximum velocity encountered
  - `'max_acceleration'`: Maximum acceleration
  - `'max_lateral_accel'`: Maximum lateral acceleration
  - `'max_yaw_rate'`: Maximum yaw rate
  - `'violations'`: List of constraint violations

---

#### `calculate_distance_to_road_edge(trajectory, drivable_area=None, lane_centerline=None, lane_width=None, vehicle_width=2.0)`

**NEW**: Calculate signed distance to drivable area boundaries (Waymo Sim Agents style).

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory positions (T, 2)
- `drivable_area` (shapely.Polygon, optional): Drivable area polygon (preferred method)
- `lane_centerline` (np.ndarray, optional): Lane centerline (N, 2) - fallback method
- `lane_width` (float, optional): Lane width for centerline method
- `vehicle_width` (float): Vehicle width

**Returns:**
- `Dict`: Dictionary containing:
  - `'mean_distance'`: Mean signed distance (negative = inside, positive = outside)
  - `'min_distance'`: Minimum distance to edge
  - `'max_violation'`: Maximum drivable area violation distance
  - `'violation_rate'`: Fraction of time outside drivable area
  - `'distances'`: Distance at each timestep

**Example:**
```python
# With Shapely polygon (preferred)
from shapely.geometry import Polygon
drivable = Polygon([...])
result = calculate_distance_to_road_edge(traj, drivable_area=drivable)

# With lane centerline (fallback)
result = calculate_distance_to_road_edge(
    traj, lane_centerline=centerline, lane_width=3.5
)
```

---

#### `calculate_driving_direction_compliance(trajectory, headings, lane_centerline, angle_threshold=1.5708)`

**NEW**: Calculate driving direction compliance (wrong-way detection, nuPlan style).

**Parameters:**
- `trajectory` (np.ndarray): Ego trajectory positions (T, 2)
- `headings` (np.ndarray): Ego heading angles (T,) in radians
- `lane_centerline` (np.ndarray): Lane centerline points (N, 2)
- `angle_threshold` (float): Angle threshold for wrong-way (radians), default π/2

**Returns:**
- `Dict`: Dictionary containing:
  - `'compliance_score'`: Overall compliance score [0, 1]
  - `'wrong_way_distance'`: Total distance driven wrong way
  - `'wrong_way_rate'`: Fraction of time driving wrong way
  - `'heading_errors'`: Heading errors at each timestep

**Example:**
```python
result = calculate_driving_direction_compliance(
    trajectory, headings, lane_centerline
)
if result['compliance_score'] < 0.5:
    print("⚠️ Wrong-way driving detected!")
```

---

#### `calculate_interaction_metrics(ego_trajectory, other_objects, vehicle_size=(4.5, 2.0), close_distance_threshold=5.0)`

**NEW**: Calculate interaction/proximity metrics for multi-agent scenarios (Waymo Sim Agents style).

**Parameters:**
- `ego_trajectory` (np.ndarray): Ego trajectory positions (T, 2)
- `other_objects` (List[np.ndarray]): Other object trajectories (static or dynamic)
- `vehicle_size` (Tuple): Vehicle (length, width)
- `close_distance_threshold` (float): Distance threshold for close interactions, default 5.0m

**Returns:**
- `Dict`: Dictionary containing:
  - `'min_distance'`: Minimum distance to any object across all timesteps
  - `'mean_distance_to_nearest'`: Mean distance to nearest object
  - `'distance_to_nearest_per_timestep'`: Distance to nearest object at each timestep
  - `'closest_object_id'`: ID of closest object
  - `'closest_approach_timestep'`: Timestep of closest approach
  - `'num_close_interactions'`: Count of timesteps with distance < threshold

**Example:**
```python
result = calculate_interaction_metrics(
    ego_traj, other_vehicles, close_distance_threshold=5.0
)
print(f"Min distance: {result['min_distance']:.2f}m")
print(f"Close interactions: {result['num_close_interactions']}")
```

---

## Vector Map Metrics

### HD Map Evaluation (`admetrics.vectormap`)

#### `calculate_chamfer_distance_polyline(pred_polyline, gt_polyline, max_distance=None)`

Calculate Chamfer Distance between polylines.

**Parameters:**
- `pred_polyline` (np.ndarray): Predicted polyline (N, 2) or (N, 3)
- `gt_polyline` (np.ndarray): Ground truth polyline (M, 2) or (M, 3)
- `max_distance` (float, optional): Max distance for P/R

**Returns:**
- `Dict`: Dictionary with:
  - `'chamfer_distance'`: Chamfer distance
  - `'chamfer_pred_to_gt'`: Pred->GT distance
  - `'chamfer_gt_to_pred'`: GT->Pred distance
  - `'precision'`, `'recall'`: If max_distance provided

---

#### `calculate_frechet_distance(pred_polyline, gt_polyline)`

Calculate Fréchet Distance between polylines.

**Parameters:**
- `pred_polyline` (np.ndarray): Predicted polyline (N, 2) or (N, 3)
- `gt_polyline` (np.ndarray): Ground truth polyline (M, 2) or (M, 3)

**Returns:**
- `float`: Fréchet distance

---

#### `calculate_polyline_iou(pred_polyline, gt_polyline, buffer_distance=0.5)`

Calculate IoU for polylines (buffered overlap).

**Parameters:**
- `pred_polyline` (np.ndarray): Predicted polyline (N, 2)
- `gt_polyline` (np.ndarray): Ground truth polyline (M, 2)
- `buffer_distance` (float): Buffer distance for overlap

**Returns:**
- `float`: Polyline IoU [0, 1]

---

#### `calculate_lane_detection_metrics(pred_lanes, gt_lanes, distance_threshold=1.0)`

Calculate lane detection metrics.

**Parameters:**
- `pred_lanes` (List[np.ndarray]): Predicted lanes
- `gt_lanes` (List[np.ndarray]): Ground truth lanes
- `distance_threshold` (float): Matching threshold

**Returns:**
- `Dict`: Dictionary with 'precision', 'recall', 'f1', 'chamfer'

---

#### `calculate_topology_metrics(pred_topology, gt_topology)`

Calculate topology estimation metrics.

**Parameters:**
- `pred_topology` (dict or graph): Predicted topology
- `gt_topology` (dict or graph): Ground truth topology

**Returns:**
- `Dict`: Dictionary with:
  - `'topology_accuracy'`: Topology accuracy
  - `'connectivity_score'`: Connectivity score
  - `'edge_accuracy'`: Edge accuracy

---

#### `calculate_endpoint_error(pred_polyline, gt_polyline)`

Calculate endpoint error for polylines.

**Parameters:**
- `pred_polyline` (np.ndarray): Predicted polyline (N, 2)
- `gt_polyline` (np.ndarray): Ground truth polyline (M, 2)

**Returns:**
- `Dict`: Dictionary with:
  - `'start_error'`: Start point error
  - `'end_error'`: End point error
  - `'mean_endpoint_error'`: Mean endpoint error

---

#### `calculate_direction_accuracy(pred_polylines, gt_polylines, angle_threshold=15.0)`

Calculate direction/heading accuracy for lanes.

**Parameters:**
- `pred_polylines` (List[np.ndarray]): Predicted polylines
- `gt_polylines` (List[np.ndarray]): Ground truth polylines
- `angle_threshold` (float): Angle threshold in degrees

**Returns:**
- `Dict`: Dictionary with:
  - `'direction_accuracy'`: Direction accuracy
  - `'mean_angle_error'`: Mean angle error

---

#### `calculate_vectormap_ap(pred_lanes, gt_lanes, distance_thresholds=[0.5, 1.0, 1.5])`

Calculate Average Precision for vector map detection.

**Parameters:**
- `pred_lanes` (List[Dict]): Predicted lanes
- `gt_lanes` (List[Dict]): Ground truth lanes
- `distance_thresholds` (List[float]): Distance thresholds

**Returns:**
- `Dict`: Dictionary with:
  - `'ap'`: Mean AP
  - `'ap_0.5'`, `'ap_1.0'`, `'ap_1.5'`: AP at thresholds

---

## Simulation Quality Metrics

### Sim-to-Real Validation (`admetrics.simulation`)

#### `calculate_camera_image_quality(sim_images, real_images, metrics=None)`

Evaluate camera image simulation quality.

**Parameters:**
- `sim_images` (np.ndarray): Simulated images (N, H, W, C)
- `real_images` (np.ndarray): Real-world images (N, H, W, C)
- `metrics` (List[str], optional): Metrics to compute ['psnr', 'ssim', 'color_distribution', 'brightness', 'contrast']

**Returns:**
- `Dict`: Dictionary with requested metrics (PSNR, SSIM, color KL divergence, etc.)

---

#### `calculate_lidar_point_cloud_quality(sim_pointcloud, real_pointcloud, metrics=None)`

Evaluate LiDAR point cloud simulation quality.

**Parameters:**
- `sim_pointcloud` (np.ndarray): Simulated point cloud (N, 3) or (N, 4)
- `real_pointcloud` (np.ndarray): Real point cloud (M, 3) or (M, 4)
- `metrics` (List[str], optional): Metrics to compute ['chamfer', 'density', 'intensity', 'range']

**Returns:**
- `Dict`: Dictionary with Chamfer distance, density diff, intensity KL, range distribution

---

#### `calculate_radar_quality(sim_detections, real_detections, metrics=None)`

Evaluate radar simulation quality.

**Parameters:**
- `sim_detections` (List[Dict]): Simulated radar detections
- `real_detections` (List[Dict]): Real radar detections
- `metrics` (List[str], optional): Metrics ['velocity', 'rcs', 'position']

**Returns:**
- `Dict`: Dictionary with velocity error, RCS error, position error

---

#### `calculate_sensor_noise_characteristics(sim_sensor_data, real_sensor_data, sensor_type='lidar')`

Evaluate sensor noise characteristics.

**Parameters:**
- `sim_sensor_data` (np.ndarray): Simulated sensor data
- `real_sensor_data` (np.ndarray): Real sensor data
- `sensor_type` (str): 'lidar', 'camera', or 'radar'

**Returns:**
- `Dict`: Dictionary with:
  - `'noise_std_ratio'`: Noise std ratio
  - `'noise_distribution_kl'`: KL divergence of noise
  - `'snr_ratio'`: SNR ratio

---

#### `calculate_multimodal_sensor_alignment(sim_sensors, real_sensors, calibration=None)`

Evaluate alignment between multiple sensor modalities.

**Parameters:**
- `sim_sensors` (dict): Simulated sensors {'camera', 'lidar', 'radar'}
- `real_sensors` (dict): Real sensors with same structure
- `calibration` (dict, optional): Calibration parameters

**Returns:**
- `Dict`: Dictionary with:
  - `'temporal_alignment'`: Temporal alignment score
  - `'spatial_alignment'`: Spatial alignment score
  - `'calibration_error'`: Calibration error

---

#### `calculate_temporal_consistency(sim_sequence, real_sequence, fps=10.0)`

Evaluate temporal consistency of sensor data.

**Parameters:**
- `sim_sequence` (List[np.ndarray]): Simulated sequence
- `real_sequence` (List[np.ndarray]): Real sequence
- `fps` (float): Frames per second

**Returns:**
- `Dict`: Dictionary with:
  - `'temporal_smoothness'`: Smoothness score
  - `'motion_consistency'`: Motion consistency
  - `'frame_difference_kl'`: Frame difference KL

---

#### `calculate_perception_sim2real_gap(model_predictions_sim, model_predictions_real, ground_truth)`

Evaluate perception model sim-to-real gap.

**Parameters:**
- `model_predictions_sim` (List[Dict]): Model predictions on sim data
- `model_predictions_real` (List[Dict]): Model predictions on real data
- `ground_truth` (List[Dict]): Ground truth annotations

**Returns:**
- `Dict`: Dictionary with:
  - `'sim_accuracy'`: Accuracy on sim data
  - `'real_accuracy'`: Accuracy on real data
  - `'performance_gap'`: Performance gap
  - `'transferability_score'`: Transferability score

---

## Utility Functions

---

## Utility Functions

### Box Transforms (`admetrics.utils.transforms`)

#### `transform_box(box, translation=None, rotation=None, scale=None)`

Apply transformation to a 3D bounding box.

**Parameters:**
- `box` (np.ndarray): 3D box [x, y, z, w, h, l, yaw]
- `translation` (np.ndarray, optional): Translation vector [tx, ty, tz]
- `rotation` (float, optional): Rotation angle in radians
- `scale` (float, optional): Scaling factor

**Returns:**
- `np.ndarray`: Transformed box

---

#### `rotate_box(box, rotation, origin=None)`

Rotate a 3D bounding box around a point.

**Parameters:**
- `box` (np.ndarray): 3D box [x, y, z, w, h, l, yaw]
- `rotation` (float): Rotation angle in radians
- `origin` (np.ndarray, optional): Point to rotate around [x, y, z]

**Returns:**
- `np.ndarray`: Rotated box

---

#### `translate_box(box, translation)`

Translate a 3D bounding box.

**Parameters:**
- `box` (np.ndarray): 3D box [x, y, z, w, h, l, yaw]
- `translation` (np.ndarray): Translation vector [tx, ty, tz]

**Returns:**
- `np.ndarray`: Translated box

---

#### `convert_box_format(box, src_format, dst_format)`

Convert between different box format representations.

**Parameters:**
- `box` (np.ndarray): Box in source format
- `src_format` (str): Source format ('xyzwhlr', 'xyzhwlr', etc.)
- `dst_format` (str): Destination format

**Returns:**
- `np.ndarray`: Converted box

---

### Detection Matching (`admetrics.utils.matching`)

#### `match_detections(predictions, ground_truth, iou_threshold=0.5, method='greedy', metric_type='3d')`

Match predictions to ground truth boxes.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): Minimum IoU for match
- `method` (str): 'greedy' or 'hungarian'
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Tuple`: (matches, unmatched_preds, unmatched_gts)
  - `matches`: List of (pred_idx, gt_idx) tuples
  - `unmatched_preds`: List of unmatched prediction indices
  - `unmatched_gts`: List of unmatched GT indices

---

#### `greedy_matching(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Greedy matching algorithm (highest score first).

**Parameters:**
- `predictions` (List[Dict]): Predictions sorted by score
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): Minimum IoU threshold
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Tuple`: (matches, unmatched_preds, unmatched_gts)

---

#### `hungarian_matching(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Hungarian (optimal) matching algorithm.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): Minimum IoU threshold
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Tuple`: (matches, unmatched_preds, unmatched_gts)

---

### Non-Maximum Suppression (`admetrics.utils.nms`)

#### `nms_3d(boxes, scores=None, iou_threshold=0.5, score_threshold=0.0)`

3D Non-Maximum Suppression.

**Parameters:**
- `boxes` (List[Dict] or np.ndarray): Boxes with scores or (N, 7) array
- `scores` (np.ndarray, optional): Scores if boxes is array
- `iou_threshold` (float): IoU threshold for suppression
- `score_threshold` (float): Minimum score threshold

**Returns:**
- `List[int]`: Indices of boxes to keep

**Example:**
```python
from admetrics import nms_3d

boxes = [
    {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
    {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8}
]
keep_indices = nms_3d(boxes, iou_threshold=0.5)
```

---

#### `nms_bev(boxes, scores=None, iou_threshold=0.5, score_threshold=0.0)`

Bird's Eye View Non-Maximum Suppression.

**Parameters:**
- `boxes` (List[Dict] or np.ndarray): Boxes
- `scores` (np.ndarray, optional): Scores
- `iou_threshold` (float): IoU threshold
- `score_threshold` (float): Minimum score

**Returns:**
- `List[int]`: Indices of boxes to keep

---

## Complete Function List

### All 80 Exported Functions

**Detection (24):**
1. `calculate_iou_3d` - 3D IoU
2. `calculate_iou_bev` - BEV IoU
3. `calculate_iou_batch` - Batch IoU
4. `calculate_giou_3d` - Generalized IoU
5. `calculate_ap` - Average Precision
6. `calculate_map` - Mean Average Precision
7. `calculate_ap_coco_style` - COCO-style mAP
8. `calculate_precision_recall_curve` - PR Curve
9. `calculate_nds` - NuScenes Detection Score
10. `calculate_nds_detailed` - Detailed NDS
11. `calculate_tp_metrics` - TP Error Metrics
12. `calculate_aos` - Average Orientation Similarity
13. `calculate_aos_per_difficulty` - AOS per Difficulty
14. `calculate_orientation_similarity` - Orientation Similarity
15. `calculate_tp_fp_fn` - TP/FP/FN Counts
16. `calculate_confusion_metrics` - Confusion Metrics
17. `calculate_confusion_matrix_multiclass` - Multi-class Confusion Matrix
18. `calculate_specificity` - Specificity
19. `calculate_center_distance` - Center Distance
20. `calculate_orientation_error` - Orientation Error
21. `calculate_size_error` - Size Error
22. `calculate_velocity_error` - Velocity Error
23. `calculate_average_distance_error` - Average Distance Error
24. `calculate_translation_error_bins` - Translation Error by Distance

**Tracking (6):**
<ol start="25">
<li><code>calculate_mota</code> - MOTA</li>
<li><code>calculate_motp</code> - MOTP</li>
<li><code>calculate_clearmot_metrics</code> - CLEAR MOT Metrics</li>
<li><code>calculate_multi_frame_mota</code> - Multi-frame MOTA</li>
<li><code>calculate_hota</code> - HOTA</li>
<li><code>calculate_id_f1</code> - ID F1-Score</li>
</ol>

**Trajectory Prediction (10):**
<ol start="31">
<li><code>calculate_ade</code> - Average Displacement Error</li>
<li><code>calculate_fde</code> - Final Displacement Error</li>
<li><code>calculate_miss_rate</code> - Miss Rate</li>
<li><code>calculate_multimodal_ade</code> - Multimodal ADE</li>
<li><code>calculate_multimodal_fde</code> - Multimodal FDE</li>
<li><code>calculate_brier_fde</code> - Brier-FDE</li>
<li><code>calculate_nll</code> - Negative Log-Likelihood</li>
<li><code>calculate_trajectory_metrics</code> - Comprehensive Trajectory Metrics</li>
<li><code>calculate_collision_rate</code> - Collision Rate</li>
<li><code>calculate_drivable_area_compliance</code> - Drivable Area Compliance</li>
</ol>

**Localization (8):**
<ol start="41">
<li><code>calculate_ate</code> - Absolute Trajectory Error</li>
<li><code>calculate_rte</code> - Relative Trajectory Error</li>
<li><code>calculate_are</code> - Absolute Rotation Error</li>
<li><code>calculate_lateral_error</code> - Lateral Error</li>
<li><code>calculate_longitudinal_error</code> - Longitudinal Error</li>
<li><code>calculate_convergence_rate</code> - Convergence Rate</li>
<li><code>calculate_localization_metrics</code> - Comprehensive Localization Metrics</li>
<li><code>calculate_map_alignment_score</code> - Map Alignment Score</li>
</ol>

**Occupancy (6):**
<ol start="49">
<li><code>calculate_occupancy_iou</code> - Occupancy IoU</li>
<li><code>calculate_mean_iou</code> - Mean IoU</li>
<li><code>calculate_occupancy_precision_recall</code> - Occupancy Precision/Recall</li>
<li><code>calculate_scene_completion</code> - Scene Completion</li>
<li><code>calculate_chamfer_distance</code> - Chamfer Distance</li>
<li><code>calculate_surface_distance</code> - Surface Distance</li>
</ol>

**Planning (11):**
<ol start="55">
<li><code>calculate_l2_distance</code> - L2 Distance</li>
<li><code>calculate_collision_rate</code> - Collision Rate</li>
<li><code>calculate_progress_score</code> - Progress Score</li>
<li><code>calculate_route_completion</code> - Route Completion</li>
<li><code>average_displacement_error_planning</code> - Planning ADE</li>
<li><code>calculate_lateral_deviation</code> - Lateral Deviation</li>
<li><code>calculate_heading_error</code> - Heading Error</li>
<li><code>calculate_velocity_error</code> - Velocity Error</li>
<li><code>calculate_comfort_metrics</code> - Comfort Metrics</li>
<li><code>calculate_driving_score</code> - Driving Score</li>
<li><code>calculate_planning_kl_divergence</code> - KL Divergence</li>
</ol>

**Vector Map (8):**
<ol start="66">
<li><code>calculate_chamfer_distance_polyline</code> - Chamfer Distance for Polylines</li>
<li><code>calculate_frechet_distance</code> - Fréchet Distance</li>
<li><code>calculate_polyline_iou</code> - Polyline IoU</li>
<li><code>calculate_lane_detection_metrics</code> - Lane Detection Metrics</li>
<li><code>calculate_topology_metrics</code> - Topology Metrics</li>
<li><code>calculate_endpoint_error</code> - Endpoint Error</li>
<li><code>calculate_direction_accuracy</code> - Direction Accuracy</li>
<li><code>calculate_vectormap_ap</code> - Vector Map AP</li>
</ol>

**Simulation Quality (7):**
<ol start="74">
<li><code>calculate_camera_image_quality</code> - Camera Image Quality</li>
<li><code>calculate_lidar_point_cloud_quality</code> - LiDAR Quality</li>
<li><code>calculate_radar_quality</code> - Radar Quality</li>
<li><code>calculate_sensor_noise_characteristics</code> - Sensor Noise</li>
<li><code>calculate_multimodal_sensor_alignment</code> - Multimodal Alignment</li>
<li><code>calculate_temporal_consistency</code> - Temporal Consistency</li>
<li><code>calculate_perception_sim2real_gap</code> - Sim2Real Gap</li>
</ol>

**Utilities (9):**
<ol start="81">
<li><code>transform_box</code> - Transform Box</li>
<li><code>rotate_box</code> - Rotate Box</li>
<li><code>translate_box</code> - Translate Box</li>
<li><code>convert_box_format</code> - Convert Box Format</li>
<li><code>match_detections</code> - Match Detections</li>
<li><code>greedy_matching</code> - Greedy Matching</li>
<li><code>hungarian_matching</code> - Hungarian Matching</li>
<li><code>nms_3d</code> - 3D NMS</li>
<li><code>nms_bev</code> - BEV NMS</li>
</ol>
