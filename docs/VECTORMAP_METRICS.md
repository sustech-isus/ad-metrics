# Vector Map Detection Metrics

Comprehensive metrics for evaluating HD map vector detection, focusing on lane lines, road boundaries, and topology.

## Table of Contents

- [Vector Map Detection Metrics](#vector-map-detection-metrics)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Challenges](#key-challenges)
    - [Evaluation Philosophy](#evaluation-philosophy)
  - [Metrics](#metrics)
    - [Chamfer Distance (Polyline)](#chamfer-distance-polyline)
    - [Fréchet Distance](#fréchet-distance)
    - [Polyline IoU](#polyline-iou)
    - [Lane Detection Metrics](#lane-detection-metrics)
    - [Topology Metrics](#topology-metrics)
    - [Endpoint Error](#endpoint-error)
    - [Direction Accuracy](#direction-accuracy)
    - [Vector Map AP](#vector-map-ap)
  - [Advanced Metrics](#advanced-metrics)
    - [3D Chamfer Distance](#3d-chamfer-distance)
    - [3D Fréchet Distance](#3d-fréchet-distance)
    - [Online Lane Segment (OLS)](#online-lane-segment-ols)
    - [Per-Category Metrics](#per-category-metrics)
  - [Usage Examples](#usage-examples)
    - [Complete Evaluation Pipeline](#complete-evaluation-pipeline)
    - [Benchmark Comparison](#benchmark-comparison)
  - [Benchmark Datasets](#benchmark-datasets)
    - [1. nuScenes Map Expansion](#1-nuscenes-map-expansion)
    - [2. Argoverse 2 HD Map](#2-argoverse-2-hd-map)
    - [3. OpenLane-V2](#3-openlane-v2)
    - [4. Waymo Open Dataset (Maps)](#4-waymo-open-dataset-maps)
  - [Best Practices](#best-practices)
    - [1. Metric Selection by Use Case](#1-metric-selection-by-use-case)
    - [2. Threshold Selection](#2-threshold-selection)
    - [3. Evaluation Pipeline](#3-evaluation-pipeline)
  - [References](#references)
    - [Papers](#papers)
    - [Datasets](#datasets)
    - [See Also](#see-also)
  - [Summary](#summary)

---

## Overview

Vector map detection is critical for autonomous driving, providing structured representations of road geometry, lane connectivity, and traffic topology. Unlike raster-based HD maps, vector representations use polylines (sequences of points) to describe:

- **Lane centerlines**: Drivable paths for vehicles
- **Road boundaries**: Edges of the road surface
- **Lane boundaries**: Dividers between adjacent lanes
- **Stop lines**: Traffic control elements
- **Crosswalks**: Pedestrian areas

### Key Challenges

1. **Geometric Accuracy**: Precise polyline positioning (sub-meter accuracy)
2. **Topology Preservation**: Correct lane connectivity (successors, neighbors)
3. **Endpoint Matching**: Accurate lane merging/splitting points
4. **Direction Consistency**: Correct lane flow direction

### Evaluation Philosophy

Vector map metrics combine:
- **Geometric metrics**: Measure spatial accuracy (Chamfer, Fréchet, IoU)
- **Detection metrics**: Precision/recall for lane-level matching (AP)
- **Topology metrics**: Graph connectivity and relationships
- **Semantic metrics**: Direction, endpoints, and lane attributes

---

## Metrics

### Chamfer Distance (Polyline)

**Purpose**: Bidirectional nearest-neighbor distance between predicted and ground truth polylines.

**Formula**:
```
Chamfer(P, G) = 1/2 * [1/|P| Σ min ||p - g||₂ + 1/|G| Σ min ||g - p||₂]
                              g∈G                    p∈P
```

Where:
- P: Set of points in predicted polyline
- G: Set of points in ground truth polyline
- Forward distance: Average nearest GT point for each prediction
- Backward distance: Average nearest prediction for each GT point

**Interpretation**:
- **< 0.5m**: Excellent geometric accuracy
- **0.5-1.0m**: Good accuracy (acceptable for most applications)
- **1.0-2.0m**: Moderate accuracy (may need refinement)
- **> 2.0m**: Poor accuracy

**Returns**:
```python
{
    'chamfer_distance': float,      # Symmetric average
    'forward_distance': float,      # P → G
    'backward_distance': float,     # G → P
    'precision': float,             # Points within threshold (forward)
    'recall': float                 # Points within threshold (backward)
}
```

**Usage**:
```python
from admetrics.vectormap import chamfer_distance_polyline

predicted = np.array([[0, 0], [1, 0], [2, 0]])
ground_truth = np.array([[0, 0.1], [1, 0.1], [2, 0.1]])

result = calculate_chamfer_distance_polyline(predicted, ground_truth, threshold=0.5)
print(f"Chamfer Distance: {result['chamfer_distance']:.3f}m")
print(f"Precision: {result['precision']:.1%}")
print(f"Recall: {result['recall']:.1%}")
```

**Advantages**:
- Symmetric measure (treats P and G equally)
- Robust to slight misalignment
- Provides precision/recall at threshold

**Limitations**:
- Doesn't account for polyline ordering
- Sensitive to outliers
- Doesn't capture topology

---

### Fréchet Distance

**Purpose**: Measures similarity considering polyline ordering and continuity (optimal matching).

**Formula**:
```
Fréchet(P, G) = inf   max  ||P(α(t)) - G(β(t))||₂
               α,β  t∈[0,1]
```

Where:
- α, β: Continuous, non-decreasing reparameterizations
- Intuition: "Dog walking" distance - person on P, dog on G

**Interpretation**:
- **< 0.3m**: Excellent curve similarity
- **0.3-1.0m**: Good similarity
- **1.0-3.0m**: Moderate similarity
- **> 3.0m**: Poor similarity

**Advantages**:
- Considers point ordering
- Captures curve shape better than Chamfer
- Single value (easier comparison)

**Limitations**:
- Computationally more expensive (O(n²))
- Harder to interpret than Chamfer
- No precision/recall breakdown

**Usage**:
```python
from admetrics.vectormap import frechet_distance

curve1 = np.array([[0, 0], [5, 2], [10, 0]])
curve2 = np.array([[0, 0.5], [5, 2.5], [10, 0.5]])

distance = calculate_frechet_distance(curve1, curve2)
print(f"Fréchet Distance: {distance:.3f}m")
```

**When to Use**:
- Curved lanes (highways, ramps)
- Lane shape evaluation
- Comparing alternative representations

---

### Polyline IoU

**Purpose**: Overlap-based metric using buffered polylines (area intersection).

**Formula**:
```
IoU(P, G, w) = Area(Buffer(P, w) ∩ Buffer(G, w))
               ────────────────────────────────────
               Area(Buffer(P, w) ∪ Buffer(G, w))
```

Where:
- Buffer(P, w): Polyline expanded by width w
- Typical width: 1.5-2.0m (lane width)

**Interpretation**:
- **> 0.7**: Strong overlap (matched)
- **0.5-0.7**: Moderate overlap
- **0.3-0.5**: Weak overlap
- **< 0.3**: No meaningful overlap

**Advantages**:
- Intuitive (similar to bounding box IoU)
- Accounts for lane width
- Good for parallel lanes

**Limitations**:
- Requires width parameter
- Expensive for long polylines
- May miss thin misalignments

**Usage**:
```python
from admetrics.vectormap import polyline_iou

pred_line = np.array([[0, 0], [10, 0]])
gt_line = np.array([[0, 0.5], [10, 0.5]])

iou = calculate_polyline_iou(pred_line, gt_line, width=2.0)
print(f"Polyline IoU: {iou:.3f}")
```

---

### Lane Detection Metrics

**Purpose**: Precision, recall, and F1 for lane-level detection (matching polylines).

**Matching Strategy**:
1. Compute distance matrix (Chamfer, Fréchet, or IoU)
2. Greedy matching: pair lanes below threshold
3. Count TP, FP, FN

**Metrics**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**Returns**:
```python
{
    'precision': float,
    'recall': float,
    'f1_score': float,
    'tp': int,              # Correctly detected lanes
    'fp': int,              # False positive lanes
    'fn': int,              # Missed lanes
    'matches': List[Tuple]  # (pred_idx, gt_idx) pairs
}
```

**Usage**:
```python
from admetrics.vectormap import lane_detection_metrics

predicted_lanes = [
    np.array([[0, 0], [10, 0]]),
    np.array([[0, 3], [10, 3]])
]
ground_truth_lanes = [
    np.array([[0, 0.2], [10, 0.2]]),
    np.array([[0, 3.1], [10, 3.1]]),
    np.array([[0, 6], [10, 6]])  # Missed lane
]

result = calculate_lane_detection_metrics(
    predicted_lanes, ground_truth_lanes,
    threshold=1.0, metric='chamfer'
)
print(f"Precision: {result['precision']:.1%}")
print(f"Recall: {result['recall']:.1%}")
print(f"F1: {result['f1_score']:.3f}")
```

**Thresholds by Metric**:
- Chamfer: 1.0-2.0m
- Fréchet: 2.0-3.0m
- IoU: 0.5-0.7

---

### Topology Metrics

**Purpose**: Evaluate lane connectivity graph (successors, left/right neighbors).

**Graph Structure**:
```python
lane = {
    'polyline': np.array,
    'successors': [idx1, idx2],    # Lanes ahead
    'predecessors': [idx],          # Lanes behind
    'left_neighbor': idx,           # Adjacent left lane
    'right_neighbor': idx           # Adjacent right lane
}
```

**Metrics**:
```
Successor Accuracy = Correct successor edges / Total GT successor edges
Neighbor Accuracy = Correct neighbor edges / Total GT neighbor edges
Topology F1 = 2PR / (P + R)  [for edge-level precision/recall]
```

**Returns**:
```python
{
    'successor_accuracy': float,
    'neighbor_accuracy': float,
    'topology_f1': float,
    'correct_edges': int,
    'missing_edges': int,
    'false_edges': int
}
```

**Usage**:
```python
from admetrics.vectormap import topology_metrics

# Each lane has topology information
predicted_lanes = [
    {'polyline': lane1, 'successors': [1], 'left_neighbor': -1, 'right_neighbor': 2},
    {'polyline': lane2, 'successors': [], 'left_neighbor': 0, 'right_neighbor': -1},
    # ...
]

result = calculate_topology_metrics(predicted_lanes, ground_truth_lanes, matching_threshold=1.0)
print(f"Successor Accuracy: {result['successor_accuracy']:.1%}")
print(f"Neighbor Accuracy: {result['neighbor_accuracy']:.1%}")
```

**Importance**:
- Critical for path planning
- Enables lane change decisions
- Validates routing logic

---

### Endpoint Error

**Purpose**: Measures accuracy of lane start/end points (critical for merging/splitting).

**Formula**:
```
Endpoint Error = 1/2 * [||P_start - G_start||₂ + ||P_end - G_end||₂]
```

**Interpretation**:
- **< 0.5m**: Excellent endpoint accuracy
- **0.5-1.5m**: Good accuracy
- **1.5-3.0m**: Moderate accuracy
- **> 3.0m**: Poor accuracy

**Returns**:
```python
{
    'start_error': float,       # Start point distance
    'end_error': float,         # End point distance
    'mean_error': float,        # Average of both
    'max_error': float          # Worst of both
}
```

**Usage**:
```python
from admetrics.vectormap import endpoint_error

pred_polyline = np.array([[0, 0], [5, 1], [10, 0]])
gt_polyline = np.array([[0.2, 0.1], [5, 1], [9.8, 0.1]])

result = calculate_endpoint_error(pred_polyline, gt_polyline)
print(f"Start Error: {result['start_error']:.3f}m")
print(f"End Error: {result['end_error']:.3f}m")
```

**Critical Scenarios**:
- Highway merges/splits
- Intersection entries/exits
- Lane termination points

---

### Direction Accuracy

**Purpose**: Measures tangent vector alignment (lane flow direction).

**Formula**:
```
Direction(P, G) = 1/N Σ |cos⁻¹(T_p · T_g)|
```

Where:
- T_p, T_g: Unit tangent vectors at sampled points
- Angle in radians

**Interpretation**:
- **< 5° (0.087 rad)**: Excellent direction alignment
- **5-15° (0.26 rad)**: Good alignment
- **15-30° (0.52 rad)**: Moderate alignment
- **> 30° (0.52 rad)**: Poor alignment

**Returns**:
```python
{
    'mean_angle_error': float,      # Average (radians)
    'mean_angle_error_deg': float,  # Average (degrees)
    'max_angle_error': float,       # Worst case
    'direction_accuracy': float      # 1 - normalized_error
}
```

**Usage**:
```python
from admetrics.vectormap import direction_accuracy

pred_polyline = np.array([[0, 0], [10, 0], [20, 5]])
gt_polyline = np.array([[0, 0], [10, 0.5], [20, 5]])

result = calculate_direction_accuracy(pred_polyline, gt_polyline)
print(f"Mean Angle Error: {result['mean_angle_error_deg']:.2f}°")
print(f"Direction Accuracy: {result['direction_accuracy']:.1%}")
```

**Applications**:
- Validate lane flow (no backward lanes)
- Roundabout direction
- One-way street detection

---

### Vector Map AP

**Purpose**: Average Precision at multiple distance thresholds (similar to object detection AP).

**Formula**:
```
AP@d = Precision-Recall AUC at distance threshold d
mAP = mean(AP@d₁, AP@d₂, ..., AP@dₙ)
```

**Common Thresholds**:
- **Strict**: 0.5m, 1.0m, 1.5m
- **Moderate**: 1.0m, 2.0m, 3.0m
- **Relaxed**: 2.0m, 3.0m, 5.0m

**Returns**:
```python
{
    'mAP': float,                       # Mean over thresholds
    'AP_per_threshold': {
        0.5: float,
        1.0: float,
        2.0: float
    },
    'precision_recall_curves': Dict     # For plotting
}
```

**Usage**:
```python
from admetrics.vectormap import vectormap_ap

result = calculate_vectormap_ap(
    predicted_lanes, ground_truth_lanes,
    thresholds=[0.5, 1.0, 2.0],
    metric='chamfer'
)
print(f"mAP: {result['mAP']:.3f}")
print(f"AP@0.5m: {result['AP_per_threshold'][0.5]:.3f}")
print(f"AP@1.0m: {result['AP_per_threshold'][1.0]:.3f}")
```

**Comparison with Object Detection**:
- Object: IoU thresholds (0.5, 0.75)
- Vector Map: Distance thresholds (0.5m, 1.0m, 2.0m)

---

## Advanced Metrics

### 3D Chamfer Distance

**Purpose**: Evaluate 3D lane polylines with elevation information.

**Formula**:
```
Chamfer_3D(P, G) = 1/2 * [1/|P| Σ min ||p - g||₃D + 1/|G| Σ min ||g - p||₃D]
                                  g∈G                    p∈P
where ||·||₃D includes weighted z-axis: √(Δx² + Δy² + (w_z·Δz)²)
```

**Use Cases**:
- OpenLane-V2 benchmark (3D lane detection)
- Elevation-aware HD mapping
- Multi-level road structures (overpasses, tunnels)
- Highway ramps with significant grade changes

**Returns**:
```python
{
    'chamfer_distance_3d': float,      # Full 3D distance
    'chamfer_distance_xy': float,      # Horizontal distance (reference)
    'elevation_error': float,          # Mean absolute Z error
    'chamfer_pred_to_gt': float,       # Forward distance
    'chamfer_gt_to_pred': float        # Backward distance
}
```

**Usage**:
```python
from admetrics.vectormap import calculate_chamfer_distance_3d

# 3D lane with elevation
pred_lane_3d = np.array([[0, 0, 0], [10, 0, 0.5], [20, 0, 1.0]])
gt_lane_3d = np.array([[0, 0, 0], [10, 0, 0.4], [20, 0, 0.9]])

result = calculate_chamfer_distance_3d(pred_lane_3d, gt_lane_3d, weight_z=1.0)
print(f"3D Chamfer: {result['chamfer_distance_3d']:.3f}m")
print(f"2D Chamfer: {result['chamfer_distance_xy']:.3f}m")
print(f"Elevation Error: {result['elevation_error']:.3f}m")
```

**Parameters**:
- `weight_z`: Weight for elevation differences (default 1.0). Use higher values (e.g., 2.0) to emphasize vertical accuracy.

**Interpretation**:
- **< 0.3m**: Excellent 3D accuracy
- **0.3-1.0m**: Good 3D accuracy
- **> 1.0m**: Poor 3D accuracy
- Compare `chamfer_distance_3d` vs `chamfer_distance_xy` to assess elevation contribution

---

### 3D Fréchet Distance

**Purpose**: Curve similarity for 3D lanes considering ordering and continuity.

**Use Cases**:
- Curved ramps with elevation changes
- Complex 3D road geometry
- Topology-aware 3D lane evaluation

**Returns**:
```python
{
    'frechet_distance_3d': float,      # 3D Fréchet distance
    'frechet_distance_xy': float       # 2D reference
}
```

**Usage**:
```python
from admetrics.vectormap import calculate_frechet_distance_3d

curved_3d = np.array([[0, 0, 0], [5, 2, 0.5], [10, 0, 1.0]])
gt_3d = np.array([[0, 0, 0.1], [5, 2, 0.6], [10, 0, 1.1]])

result = calculate_frechet_distance_3d(curved_3d, gt_3d, weight_z=1.5)
print(f"3D Fréchet: {result['frechet_distance_3d']:.3f}m")
```

---

### Online Lane Segment (OLS)

**Purpose**: Evaluate temporal consistency and tracking quality for online HD map construction.

**Formula**:
```
OLS = (1 - w_c) × Detection_Score + w_c × Consistency_Score

Detection_Score = Average F1 across frames
Consistency_Score = Correct_Tracks / Total_Tracks
```

**Use Cases**:
- OpenLane-V2 online evaluation
- Streaming map construction
- Temporal lane tracking
- Multi-frame aggregation assessment

**Metrics**:
- **Detection Score**: Average per-frame F1 score
- **Consistency Score**: Fraction of correctly maintained lane IDs across frames
- **ID Switches**: Number of incorrect lane identity changes
- **OLS**: Combined score (default weight: 30% consistency, 70% detection)

**Returns**:
```python
{
    'ols': float,                      # Overall Online Lane Segment score
    'detection_score': float,          # Avg F1 across frames
    'consistency_score': float,        # Temporal consistency
    'avg_precision': float,            # Avg precision
    'avg_recall': float,               # Avg recall
    'id_switches': int                 # Number of identity switches
}
```

**Usage**:
```python
from admetrics.vectormap import calculate_online_lane_segment_metric

# Sequence of frames (each frame is a list of lane polylines)
pred_sequence = [
    [lane1_frame1, lane2_frame1, lane3_frame1],  # Frame 1
    [lane1_frame2, lane2_frame2, lane3_frame2],  # Frame 2
    [lane1_frame3, lane2_frame3, lane3_frame3],  # Frame 3
]

gt_sequence = [
    [gt_lane1_f1, gt_lane2_f1, gt_lane3_f1],
    [gt_lane1_f2, gt_lane2_f2, gt_lane3_f2],
    [gt_lane1_f3, gt_lane2_f3, gt_lane3_f3],
]

result = calculate_online_lane_segment_metric(
    pred_sequence, gt_sequence,
    distance_threshold=1.0,
    consistency_weight=0.3
)

print(f"OLS Score: {result['ols']:.3f}")
print(f"Detection: {result['detection_score']:.3f}")
print(f"Consistency: {result['consistency_score']:.3f}")
print(f"ID Switches: {result['id_switches']}")
```

**Interpretation**:
- **OLS > 0.8**: Excellent online performance
- **OLS 0.6-0.8**: Good online performance
- **OLS < 0.6**: Poor temporal consistency or detection
- **ID Switches = 0**: Perfect tracking
- **Low Detection + High Consistency**: Stable but incomplete detection
- **High Detection + Low Consistency**: Good per-frame but unstable tracking

---

### Per-Category Metrics

**Purpose**: Separate evaluation for different map element types.

**Supported Categories**:
- `lane_divider`: Lane boundary markings (dashed, solid, double)
- `road_edge`: Road boundaries and curbs
- `crosswalk`: Pedestrian crossing areas
- `stop_line`: Traffic control lines
- `centerline`: Lane centerlines
- Custom categories

**Use Cases**:
- Category-specific performance analysis
- Imbalanced dataset evaluation
- Focused improvement on specific element types
- Benchmark reporting by category

**Returns**:
```python
{
    'lane_divider': {
        'precision': float,
        'recall': float,
        'f1_score': float,
        'ap': float,
        'num_pred': int,
        'num_gt': int
    },
    'road_edge': {...},
    'crosswalk': {...},
    ...
    'overall': {
        'precision': float,      # Macro-average
        'recall': float,
        'f1_score': float,
        'map': float,
        'num_categories': int
    }
}
```

**Usage**:
```python
from admetrics.vectormap import calculate_per_category_metrics

# Predictions with category labels
pred_lanes = [
    {'polyline': np.array([[0, 0], [10, 0]]), 'category': 'lane_divider', 'score': 0.95},
    {'polyline': np.array([[0, 3], [10, 3]]), 'category': 'road_edge', 'score': 0.88},
    {'polyline': np.array([[0, 10], [10, 10]]), 'category': 'crosswalk', 'score': 0.75}
]

gt_lanes = [
    {'polyline': np.array([[0, 0.1], [10, 0.1]]), 'category': 'lane_divider'},
    {'polyline': np.array([[0, 3.1], [10, 3.1]]), 'category': 'road_edge'},
    {'polyline': np.array([[0, 10.1], [10, 10.1]]), 'category': 'crosswalk'}
]

result = calculate_per_category_metrics(
    pred_lanes, gt_lanes,
    categories=['lane_divider', 'road_edge', 'crosswalk'],
    distance_threshold=1.0
)

# Per-category results
for category in ['lane_divider', 'road_edge', 'crosswalk']:
    cat_result = result[category]
    print(f"\n{category}:")
    print(f"  Precision: {cat_result['precision']:.1%}")
    print(f"  Recall: {cat_result['recall']:.1%}")
    print(f"  F1: {cat_result['f1_score']:.3f}")
    print(f"  AP: {cat_result['ap']:.3f}")
    print(f"  Count: {cat_result['num_gt']} GT, {cat_result['num_pred']} pred")

# Overall (macro-average)
print(f"\nOverall mAP: {result['overall']['map']:.3f}")
```

**Applications**:
- **Benchmark Reporting**: Standard format for competition leaderboards
- **Ablation Studies**: Identify which categories need improvement
- **Class Imbalance**: Prevent dominant categories from masking poor performance on rare elements
- **Multi-Task Analysis**: Separate performance for different prediction heads

---

## Usage Examples

### Complete Evaluation Pipeline

```python
import numpy as np
from admetrics.vectormap import (
    calculate_chamfer_distance_polyline,
    calculate_frechet_distance,
    calculate_polyline_iou,
    calculate_lane_detection_metrics,
    calculate_topology_metrics,
    calculate_endpoint_error,
    calculate_direction_accuracy,
    calculate_vectormap_ap,
    calculate_chamfer_distance_3d,
    calculate_online_lane_segment_metric,
    calculate_per_category_metrics
)

# Load predictions and ground truth
predicted_lanes = load_predicted_lanes()  # List of polylines
ground_truth_lanes = load_ground_truth_lanes()

# 1. Geometric accuracy per lane
for pred, gt in zip(predicted_lanes, ground_truth_lanes):
    chamfer = calculate_chamfer_distance_polyline(pred, gt, threshold=1.0)
    frechet = calculate_frechet_distance(pred, gt)
    iou = calculate_polyline_iou(pred, gt, width=2.0)
    
    print(f"Chamfer: {chamfer['chamfer_distance']:.2f}m")
    print(f"Fréchet: {frechet:.2f}m")
    print(f"IoU: {iou:.2f}")

# 2. Detection metrics (precision/recall)
detection = calculate_lane_detection_metrics(
    predicted_lanes, ground_truth_lanes,
    threshold=1.0, metric='chamfer'
)
print(f"\nDetection Metrics:")
print(f"  Precision: {detection['precision']:.1%}")
print(f"  Recall: {detection['recall']:.1%}")
print(f"  F1: {detection['f1_score']:.3f}")

# 3. Topology evaluation
topology = calculate_topology_metrics(predicted_lanes, ground_truth_lanes)
print(f"\nTopology Metrics:")
print(f"  Successor Accuracy: {topology['successor_accuracy']:.1%}")
print(f"  Neighbor Accuracy: {topology['neighbor_accuracy']:.1%}")

# 4. Endpoint and direction accuracy
for pred, gt in zip(predicted_lanes, ground_truth_lanes):
    endpoint = calculate_endpoint_error(pred, gt)
    direction = calculate_direction_accuracy(pred, gt)
    
    print(f"\nEndpoint Error: {endpoint['mean_error']:.2f}m")
    print(f"Direction Error: {direction['mean_angle_error_deg']:.1f}°")

# 5. Overall AP metric
ap_result = calculate_vectormap_ap(
    predicted_lanes, ground_truth_lanes,
    thresholds=[0.5, 1.0, 2.0],
    metric='chamfer'
)
print(f"\nVector Map AP:")
print(f"  mAP: {ap_result['mAP']:.3f}")
for thresh, ap in ap_result['AP_per_threshold'].items():
    print(f"  AP@{thresh}m: {ap:.3f}")
```

### Benchmark Comparison

```python
# Compare model performance across datasets
datasets = ['nuScenes', 'Argoverse2', 'OpenLane-V2']
models = ['VectorNet', 'HDMapNet', 'MapTR']

results = {}
for dataset in datasets:
    for model in models:
        preds = load_predictions(dataset, model)
        gts = load_ground_truth(dataset)
        
        metrics = {
            'chamfer': calculate_chamfer_distance_polyline(preds, gts, threshold=1.0),
            'detection': calculate_lane_detection_metrics(preds, gts, threshold=1.0),
            'topology': calculate_topology_metrics(preds, gts),
            'ap': calculate_vectormap_ap(preds, gts, thresholds=[0.5, 1.0, 2.0])
        }
        
        results[(dataset, model)] = metrics

# Print comparison table
print(f"{'Dataset':<15} {'Model':<15} {'Chamfer':<10} {'F1':<10} {'mAP':<10}")
for (dataset, model), metrics in results.items():
    print(f"{dataset:<15} {model:<15} "
          f"{metrics['chamfer']['chamfer_distance']:<10.2f} "
          f"{metrics['detection']['f1_score']:<10.3f} "
          f"{metrics['ap']['mAP']:<10.3f}")
```

---

## Benchmark Datasets

### 1. nuScenes Map Expansion

**Description**: Urban driving dataset with vectorized HD maps for Singapore and Boston.

**Characteristics**:
- **Scenes**: 1,000 scenes, 850 annotated maps
- **Coverage**: 11.3 km² area
- **Elements**: Lane centerlines, boundaries, crosswalks
- **Topology**: Lane connectivity graph
- **Format**: Polylines with attributes

**Metrics Used**:
- Chamfer Distance (primary)
- Fréchet Distance
- Topology Accuracy
- Direction Accuracy

**Benchmark Performance** (2023):
- **VectorNet**: Chamfer 1.2m, Topology 85%
- **HDMapNet**: Chamfer 1.5m, Topology 78%
- **MapTR**: Chamfer 0.9m, Topology 89%

**Citation**:
```
@inproceedings{caesar2020nuscenes,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and others},
  booktitle={CVPR},
  year={2020}
}
```

---

### 2. Argoverse 2 HD Map

**Description**: Large-scale dataset with detailed vector maps for 6 cities.

**Characteristics**:
- **Scenes**: 20,000+ scenarios
- **Coverage**: 1,000+ km of roadway
- **Elements**: Lane segments, crosswalks, pedestrian crossings
- **Topology**: Rich connectivity (left/right neighbors, successors)
- **Format**: Polygon-based lanes with centerlines

**Metrics Used**:
- Chamfer Distance
- Polyline IoU
- Lane Detection Precision/Recall
- Topology F1

**Benchmark Performance** (2023):
- **VectorMapNet**: F1 0.72, Chamfer 1.1m
- **MapTR-V2**: F1 0.78, Chamfer 0.85m

**Citation**:
```
@inproceedings{wilson2023argoverse,
  title={Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting},
  author={Wilson, Benjamin and others},
  booktitle={NeurIPS},
  year={2023}
}
```

---

### 3. OpenLane-V2

**Description**: Large-scale topology-focused dataset with 3D lane detection.

**Characteristics**:
- **Scenes**: 2,000 scenes from 5 datasets
- **Elements**: 3D lane centerlines with elevation
- **Topology**: Explicit successor/neighbor relationships
- **Challenges**: Complex intersections, multi-level roads
- **Format**: 3D polylines + topology graph

**Metrics Used**:
- Fréchet Distance (3D)
- Topology Metrics (primary focus)
- Endpoint Error
- Vector Map AP

**Benchmark Performance** (2024):
- **TopoNet**: Topology F1 0.68, Fréchet 1.8m
- **LaneGAP**: Topology F1 0.73, Fréchet 1.5m

**Citation**:
```
@article{wang2023openlane,
  title={OpenLane-V2: A Topology Reasoning Benchmark for Autonomous Driving},
  author={Wang, Huijie and others},
  journal={NeurIPS},
  year={2023}
}
```

---

### 4. Waymo Open Dataset (Maps)

**Description**: High-quality vector maps from Waymo's autonomous fleet.

**Characteristics**:
- **Coverage**: Phoenix, San Francisco, others
- **Elements**: Lane boundaries, road edges, crosswalks, stop lines
- **Quality**: Sub-decimeter accuracy
- **Format**: Polylines with semantic labels

**Metrics Used**:
- Chamfer Distance (strict thresholds: 0.3m, 0.5m)
- Direction Accuracy
- Endpoint Error

---

## Best Practices

### 1. Metric Selection by Use Case

**High-Speed Highway**:
- Primary: Fréchet Distance (captures curve shape)
- Secondary: Direction Accuracy
- Threshold: 1.5-2.0m

**Urban Intersections**:
- Primary: Topology Metrics
- Secondary: Endpoint Error
- Threshold: 0.5-1.0m

**General Evaluation**:
- Primary: Chamfer Distance + F1
- Secondary: Vector Map AP
- Threshold: 1.0m

### 2. Threshold Selection

**Conservative (Safety-Critical)**:
- Chamfer: 0.5m
- Fréchet: 1.0m
- IoU: 0.7

**Standard (Production)**:
- Chamfer: 1.0m
- Fréchet: 2.0m
- IoU: 0.5

**Relaxed (Research)**:
- Chamfer: 2.0m
- Fréchet: 3.0m
- IoU: 0.3

### 3. Evaluation Pipeline

```python
def evaluate_vectormap_model(predictions, ground_truth):
    """Complete evaluation pipeline."""
    
    # 1. Geometric Accuracy
    geometric = {
        'chamfer': calculate_chamfer_distance_polyline(predictions, ground_truth, threshold=1.0),
        'frechet': np.mean([calculate_frechet_distance(p, g) for p, g in zip(predictions, ground_truth)]),
        'iou': np.mean([calculate_polyline_iou(p, g, width=2.0) for p, g in zip(predictions, ground_truth)])
    }
    
    # 2. Detection Performance
    detection = calculate_lane_detection_metrics(predictions, ground_truth, threshold=1.0)
    
    # 3. Topology (if available)
    topology = calculate_topology_metrics(predictions, ground_truth, matching_threshold=1.0)
    
    # 4. Semantic Accuracy
    endpoints = np.mean([calculate_endpoint_error(p, g)['mean_error'] for p, g in zip(predictions, ground_truth)])
    directions = np.mean([calculate_direction_accuracy(p, g)['mean_angle_error_deg'] for p, g in zip(predictions, ground_truth)])
    
    # 5. Overall AP
    ap = calculate_vectormap_ap(predictions, ground_truth, thresholds=[0.5, 1.0, 2.0])
    
    return {
        'geometric': geometric,
        'detection': detection,
        'topology': topology,
        'semantic': {'endpoint_error': endpoints, 'direction_error': directions},
        'ap': ap
    }
```

---

## References

### Papers

1. **VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation**
   - Gao, J., Sun, C., Zhao, H., Shen, Y., Anguelov, D., Li, C., & Schmid, C. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/2005.04259
   - https://doi.org/10.48550/arXiv.2005.04259
   - Hierarchical graph neural network for vectorized HD maps
   - Introduced polyline representation for lanes and trajectory prediction

2. **HDMapNet: An Online HD Map Construction and Evaluation Framework**
   - Li, Q., Wang, Y., Wang, Y., & Zhao, H. (2022)
   - International Conference on Robotics and Automation (ICRA) 2022
   - https://arxiv.org/abs/2107.06307
   - https://doi.org/10.48550/arXiv.2107.06307
   - Online HD map construction from multi-view camera and LiDAR
   - Evaluated using Chamfer Distance and IoU metrics

3. **MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction**
   - Liao, B., Chen, S., Wang, X., Cheng, T., Zhang, Q., Liu, W., & Huang, C. (2023)
   - ICLR 2023 (Spotlight)
   - https://arxiv.org/abs/2208.14437
   - https://doi.org/10.48550/arXiv.2208.14437
   - https://github.com/hustvl/MapTR
   - Transformer-based end-to-end vectorized map construction
   - State-of-the-art performance on nuScenes and Argoverse 2 datasets
   - Unified permutation-equivalent modeling for map elements

4. **OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping**
   - Wang, H., Li, T., Li, Y., Chen, L., et al. (2023)
   - NeurIPS 2023 Track Datasets and Benchmarks
   - https://arxiv.org/abs/2304.10440
   - https://github.com/OpenDriveLab/OpenLane-V2
   - First perception and reasoning benchmark for scene structure
   - Introduces 3D lane detection and topology reasoning tasks

### Datasets

5. **nuScenes: A multimodal dataset for autonomous driving**
   - Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1903.11027
   - https://www.nuscenes.org/
   - 1000 scenes with 1.4M camera images, includes HD map annotations
   - Standard benchmark for online HD map construction

6. **Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting**
   - Wilson, B., Qi, W., Agarwal, T., Lambert, J., Singh, J., et al. (2023)
   - NeurIPS 2023 Track Datasets and Benchmarks
   - https://arxiv.org/abs/2301.00493
   - https://www.argoverse.org/av2.html
   - https://argoverse.github.io/user-guide/
   - High-quality HD maps across 6 cities with detailed lane geometry

7. **Waymo Open Dataset: An autonomous driving dataset**
   - Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1912.04838
   - https://waymo.com/open
   - Large-scale dataset with 1,150 scenes, includes map features

### See Also

- [DETECTION_METRICS.md](DETECTION_METRICS.md) - 3D object detection metrics
- [LOCALIZATION_METRICS.md](LOCALIZATION_METRICS.md) - Ego-pose accuracy metrics
- [TRACKING_METRICS.md](TRACKING_METRICS.md) - Multi-object tracking metrics
- [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) - Motion forecasting metrics
- [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) - 3D occupancy prediction metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics

---

## Summary

Vector map metrics provide comprehensive evaluation for HD map detection:

| Metric | Purpose | Key Advantage | Typical Threshold |
|--------|---------|---------------|-------------------|
| **Chamfer Distance** | Point-to-point accuracy | Symmetric, interpretable | 1.0m |
| **Fréchet Distance** | Curve similarity | Considers ordering | 2.0m |
| **Polyline IoU** | Overlap measurement | Intuitive, width-aware | 0.5 |
| **Lane Detection** | Precision/Recall | Detection quality | 1.0m |
| **Topology** | Connectivity | Planning-critical | Edge-level |
| **Endpoint Error** | Merge/split accuracy | Lane transitions | 1.0m |
| **Direction** | Flow alignment | Semantic correctness | 10° |
| **Vector Map AP** | Overall performance | Multi-threshold | 0.5-2.0m |
| **3D Chamfer** | 3D lane accuracy | Elevation-aware | 1.0m (3D) |
| **3D Fréchet** | 3D curve similarity | 3D shape matching | 2.0m (3D) |
| **OLS** | Temporal consistency | Online evaluation | 0.7+ score |
| **Per-Category** | Element-specific | Category analysis | Varies |

**Recommended Workflow**:

**Basic Evaluation**:
1. Start with **Chamfer Distance** (quick, interpretable)
2. Add **Lane Detection Metrics** (precision/recall)
3. Evaluate **Topology** (if applicable)
4. Compute **Vector Map AP** (comprehensive)
5. Check **Endpoint/Direction** (semantic validation)

**Advanced Evaluation**:
6. Use **3D Chamfer/Fréchet** for elevation-aware datasets (OpenLane-V2)
7. Apply **OLS** for online/streaming scenarios
8. Compute **Per-Category Metrics** for detailed analysis

**By Use Case**:
- **2D Lanes (nuScenes)**: Chamfer, Detection, Topology, AP
- **3D Lanes (OpenLane-V2)**: 3D Chamfer, 3D Fréchet, Topology, OLS
- **Multi-Category (Argoverse 2)**: Per-Category, Detection, AP
- **Online Construction**: OLS, Detection, Consistency

For questions or issues, see the [examples](../examples/vectormap_evaluation.py) or consult the [API reference](api_reference.md).

