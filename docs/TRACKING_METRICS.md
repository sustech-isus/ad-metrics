# Multi-Object Tracking Metrics

This document provides a comprehensive reference for multi-object tracking (MOT) metrics implemented in `admetrics`, covering all major benchmarks including MOTChallenge, nuScenes, KITTI, Waymo, and Argoverse.

## Table of Contents

- [Multi-Object Tracking Metrics](#multi-object-tracking-metrics)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Implemented Metrics](#implemented-metrics)
    - [1. CLEAR MOT Metrics](#1-clear-mot-metrics)
      - [MOTA (Multiple Object Tracking Accuracy)](#mota-multiple-object-tracking-accuracy)
      - [MOTP (Multiple Object Tracking Precision)](#motp-multiple-object-tracking-precision)
      - [MODA (Multiple Object Detection Accuracy)](#moda-multiple-object-detection-accuracy)
    - [2. HOTA (Higher Order Tracking Accuracy)](#2-hota-higher-order-tracking-accuracy)
    - [3. IDF1 (ID F1 Score)](#3-idf1-id-f1-score)
    - [4. nuScenes Metrics](#4-nuscenes-metrics)
      - [AMOTA (Average MOTA)](#amota-average-mota)
      - [MOTAR (MOTA at Recall)](#motar-mota-at-recall)
      - [FAF (False Alarms per Frame)](#faf-false-alarms-per-frame)
    - [5. Track-Level Metrics](#5-track-level-metrics)
    - [6. Trajectory-Level Metrics](#6-trajectory-level-metrics)
    - [7. Detection-Only Metrics](#7-detection-only-metrics)
    - [8. Soft MOTA (sMOTA)](#8-soft-mota-smota)
    - [9. Completeness Metrics](#9-completeness-metrics)
    - [10. Identity Preservation Metrics](#10-identity-preservation-metrics)
  - [Comprehensive Metrics Reference](#comprehensive-metrics-reference)
    - [Error Type Metrics](#error-type-metrics)
    - [Temporal Metrics](#temporal-metrics)
    - [VACE Metrics](#vace-metrics)
    - [Advanced Metrics](#advanced-metrics)
    - [Auxiliary Metrics](#auxiliary-metrics)
  - [Metric Comparison](#metric-comparison)
  - [Benchmark-Specific Metrics](#benchmark-specific-metrics)
  - [Choosing the Right Metric](#choosing-the-right-metric)
  - [Common Patterns](#common-patterns)
    - [Perfect Tracking](#perfect-tracking)
    - [ID Switch Detection](#id-switch-detection)
    - [Fragmentation Detection](#fragmentation-detection)
  - [Implementation Details](#implementation-details)
    - [Matching Algorithm](#matching-algorithm)
    - [ID Switch Detection](#id-switch-detection-1)
    - [Fragmentation Counting](#fragmentation-counting)
  - [Complete Metric Taxonomy](#complete-metric-taxonomy)
  - [Examples](#examples)
  - [Testing](#testing)
  - [References](#references)

## Overview

The tracking module (`admetrics.tracking`) provides comprehensive metrics for evaluating 3D multi-object tracking systems. These metrics go beyond single-frame detection to measure temporal consistency, identity preservation, and trajectory quality.

This document covers:
- **Implemented metrics**: Ready-to-use functions in `admetrics.tracking`
- **Comprehensive reference**: All metrics used across major benchmarks
- **Usage examples**: Practical code snippets for each metric
- **Selection guide**: How to choose the right metrics for your evaluation

## Implemented Metrics

The following metrics are currently implemented and available in `admetrics.tracking`:

### 1. CLEAR MOT Metrics

The CLEAR MOT (Multiple Object Tracking) framework provides two fundamental metrics:

#### MOTA (Multiple Object Tracking Accuracy)

**Formula:**
```
MOTA = 1 - (FN + FP + IDSW) / GT
```

Where:
- FN: False Negatives (missed detections)
- FP: False Positives (spurious detections)
- IDSW: ID Switches (identity switches)
- GT: Total number of ground truth objects

**Characteristics:**
- Ranges from -∞ to 1.0 (can be negative with many errors)
- Penalizes detection errors AND tracking errors (ID switches)
- Higher is better
- Most commonly used tracking accuracy metric

**Usage:**
```python
from admetrics.tracking import calculate_multi_frame_mota

# predictions: Dict[frame_id -> List[detection]]
# ground_truth: Dict[frame_id -> List[detection]]
# Each detection: {'box': [...], 'track_id': int, 'class': str}

result = calculate_multi_frame_mota(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"MOTA: {result['mota']:.4f}")
print(f"ID Switches: {result['num_switches']}")
print(f"Fragmentations: {result['num_fragmentations']}")
```

#### MOTP (Multiple Object Tracking Precision)

**Formula:**
```
MOTP = sum(distance_i) / TP
```

**Characteristics:**
- Average localization error for all true positive detections
- Measured in same units as coordinates (typically meters)
- Lower is better
- Measures how precisely objects are localized, not tracking consistency

**Usage:**
```python
from admetrics.tracking import calculate_motp

result = calculate_motp(
    predictions,
    ground_truth,
    iou_threshold=0.5,
    distance_type="euclidean"  # or "bev", "vertical"
)

print(f"MOTP: {result['motp']:.4f} meters")
```

### 2. HOTA (Higher Order Tracking Accuracy)

HOTA is a more recent metric that provides a balanced view of detection and association performance.

**Formula:**
```
HOTA = sqrt(DetA × AssA)
```

Where:
- DetA: Detection Accuracy (how well objects are detected)
- AssA: Association Accuracy (how well identities are maintained)

**Characteristics:**
- Ranges from 0 to 1
- Geometric mean balances detection and association equally
- More intuitive than MOTA for comparing trackers
- Less sensitive to class imbalance
- Better for understanding tracker behavior

**Usage:**
```python
from admetrics.tracking import calculate_hota

result = calculate_hota(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"HOTA: {result['hota']:.4f}")
print(f"DetA: {result['det_a']:.4f}")
print(f"AssA: {result['ass_a']:.4f}")
```

### 3. IDF1 (ID F1 Score)

IDF1 measures identity preservation using an F1-based approach.

**Formula:**
```
IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
IDP = IDTP / (IDTP + IDFP)  # ID Precision
IDR = IDTP / (IDTP + IDFN)  # ID Recall
```

Where:
- IDTP: Correctly identified detections
- IDFP: Incorrectly identified detections
- IDFN: Missed identifications

**Characteristics:**
- Ranges from 0 to 1
- Directly measures how well identities are preserved
- F1 score balances precision and recall of ID assignments
- More sensitive to ID switches than MOTA

**Usage:**
```python
from admetrics.tracking import calculate_id_f1

result = calculate_id_f1(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"IDF1: {result['idf1']:.4f}")
print(f"ID Precision: {result['idp']:.4f}")
print(f"ID Recall: {result['idr']:.4f}")
```

### 4. nuScenes Metrics

Metrics specifically designed for the nuScenes tracking benchmark, focusing on robustness across different recall operating points.

#### AMOTA (Average MOTA)

AMOTA is the primary metric used in nuScenes tracking benchmark. It evaluates tracking performance across multiple recall thresholds to provide a more robust measure.

**Formula:**
```
AMOTA = (1/R) × Σ MOTA(r) for r in recall_thresholds
```

Where MOTA(r) is computed on top-scoring predictions filtered to achieve recall r.

**Characteristics:**
- Default recall thresholds: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- More robust than single-threshold MOTA
- Penalizes systems that only work well at specific operating points
- Ranges from -∞ to 1.0

**Usage:**
```python
from admetrics.tracking import calculate_amota

result = calculate_amota(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"AMOTA: {result['amota']:.4f}")
print(f"AMOTP: {result['amotp']:.4f}")
print(f"MOTA values: {result['motas']}")
```

#### MOTAR (MOTA at Recall)

MOTAR evaluates tracking performance at a specific recall operating point.

**Formula:**
```
MOTAR = MOTA computed at target recall threshold
```

**Characteristics:**
- Default recall threshold: 0.5
- Useful for comparing systems at matched operating points
- Part of nuScenes tracking evaluation

**Usage:**
```python
from admetrics.tracking import calculate_motar

result = calculate_motar(
    predictions,
    ground_truth,
    recall_threshold=0.5,
    iou_threshold=0.5
)

print(f"MOTAR: {result['motar']:.4f}")
print(f"Actual recall: {result['actual_recall']:.4f}")
```

#### FAF (False Alarms per Frame)

FAF measures the average number of false positive detections per frame.

**Formula:**
```
FAF = total_false_positives / num_frames
```

**Characteristics:**
- Lower is better
- Key metric in nuScenes evaluation
- Measures detector noise/spurious detections
- Complementary to MOTA

**Usage:**
```python
from admetrics.tracking import calculate_false_alarm_rate

result = calculate_false_alarm_rate(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"FAF: {result['faf']:.4f}")
print(f"Total FP: {result['total_false_positives']}")
print(f"FAR: {result['far']:.4f}")  # False alarm rate
```

### 5. Track-Level Metrics

Metrics that evaluate the quality of entire tracks rather than frame-by-frame detections.

**Track Recall:** Ratio of ground truth tracks that are detected at least once  
**Track Precision:** Ratio of predicted tracks that match at least one ground truth

**Usage:**
```python
from admetrics.tracking import calculate_track_metrics

result = calculate_track_metrics(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"Track Recall: {result['track_recall']:.4f}")
print(f"Track Precision: {result['track_precision']:.4f}")
print(f"Matched GT tracks: {result['num_matched_gt_tracks']}/{result['num_gt_tracks']}")
```

## Additional Metrics

### MODA (Multiple Object Detection Accuracy)

MODA is MOTA without the ID switch penalty - it only considers detection errors.

**Formula:**
```
MODA = 1 - (FN + FP) / GT
```

**Usage:**
```python
from admetrics.tracking import calculate_moda

result = calculate_moda(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"MODA: {result['moda']:.4f}")  # Pure detection quality
```

### Enhanced HOTA Components

Full HOTA decomposition with all sub-metrics.

**Usage:**
```python
from admetrics.tracking import calculate_hota_components

result = calculate_hota_components(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"HOTA: {result['hota']:.4f}")
print(f"DetA: {result['det_a']:.4f}")
print(f"  DetRe: {result['det_re']:.4f}")  # Detection Recall
print(f"  DetPr: {result['det_pr']:.4f}")  # Detection Precision
print(f"AssA: {result['ass_a']:.4f}")
print(f"  AssRe: {result['ass_re']:.4f}")  # Association Recall
print(f"  AssPr: {result['ass_pr']:.4f}")  # Association Precision
print(f"LocA: {result['loc_a']:.4f}")      # Localization Accuracy
```

## Additional Metrics

### 6. Trajectory-Level Metrics

Trajectory metrics classify tracks based on coverage and compute track-level statistics.

**Usage:**
```python
from admetrics.tracking import calculate_trajectory_metrics

result = calculate_trajectory_metrics(
    predictions,
    ground_truth,
    iou_threshold=0.5,
    mt_threshold=0.8,  # Mostly Tracked threshold
    ml_threshold=0.2   # Mostly Lost threshold
)

print(f"Mostly Tracked (MT): {result['mt_count']} ({result['mt_ratio']:.2%})")
print(f"Partially Tracked (PT): {result['pt_count']} ({result['pt_ratio']:.2%})")
print(f"Mostly Lost (ML): {result['ml_count']} ({result['ml_ratio']:.2%})")
print(f"Average Track Coverage: {result['avg_coverage']:.2%}")
print(f"Average Track Length: {result['avg_track_length']:.1f} frames")
```

**Track Classification:**
- **Mostly Tracked (MT)**: ≥80% of frames successfully tracked
- **Partially Tracked (PT)**: 20-80% of frames tracked
- **Mostly Lost (ML)**: <20% of frames tracked

### 7. Detection-Only Metrics

Frame-level detection metrics without tracking (no track_id consideration).

**Usage:**
```python
from admetrics.tracking import calculate_detection_metrics

result = calculate_detection_metrics(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"Precision: {result['precision']:.4f}")
print(f"Recall: {result['recall']:.4f}")
print(f"F1 Score: {result['f1']:.4f}")
print(f"True Positives: {result['tp']}")
print(f"False Positives: {result['fp']}")
print(f"False Negatives: {result['fn']}")
```

### 8. Soft MOTA (sMOTA)

sMOTA uses continuous IoU similarity instead of binary matching, particularly useful for segmentation tracking (MOTS).

**Formula:**
```
sMOTA = 1 - (soft_FN + soft_FP + IDSW) / GT
```
where soft errors use (1 - IoU) instead of binary counting.

**Usage:**
```python
from admetrics.tracking import calculate_smota

result = calculate_smota(
    predictions,
    ground_truth,
    iou_threshold=0.5,
    use_soft_matching=True  # Use continuous IoU
)

print(f"sMOTA: {result['smota']:.4f}")
print(f"Soft TP Error: {result['soft_tp_error']:.4f}")
print(f"Matches: {result['num_matches']}")
print(f"ID Switches: {result['num_switches']}")
```

### 9. Completeness Metrics

Measures how completely ground truth objects are detected across frames.

**Usage:**
```python
from admetrics.tracking import calculate_completeness

result = calculate_completeness(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"GT Coverage Ratio: {result['gt_covered_ratio']:.2%}")  # % of GT objects detected at least once
print(f"Avg GT Coverage: {result['avg_gt_coverage']:.2%}")     # Avg % of frames each GT is detected
print(f"Frame Coverage: {result['frame_coverage']:.2%}")        # % of frames with detections
print(f"Detection Density: {result['detection_density']:.2f}") # Avg detections per frame
```

### 10. Identity Preservation Metrics

Detailed analysis of identity consistency and fragmentation.

**Usage:**
```python
from admetrics.tracking import calculate_identity_metrics

result = calculate_identity_metrics(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"ID Switches: {result['id_switches']}")
print(f"ID Switch Rate: {result['id_switch_rate']:.4f}")  # Per GT track
print(f"Avg Track Purity: {result['avg_track_purity']:.2%}")  # % from dominant GT
print(f"Avg Track Completeness: {result['avg_track_completeness']:.2%}")  # % of GT in dominant pred
print(f"Fragmentations: {result['num_fragmentations']}")
print(f"Fragmentation Rate: {result['fragmentation_rate']:.4f}")  # Per GT track
```

**Key Metrics:**
- **Track Purity**: Fraction of predicted track from its dominant ground truth object
- **Track Completeness**: Fraction of ground truth object in its dominant predicted track
- **Fragmentation Rate**: Average number of fragmentations per ground truth track

---

## Comprehensive Metrics Reference

This section provides a complete reference of ALL tracking metrics used across major benchmarks, including those not yet implemented in `admetrics`.

### Error Type Metrics

#### FP (False Positives)
**Definition:** Number of predicted boxes not matched to any GT.
```
FP = total_predictions - TP
```
**Status:** ✅ Implemented (returned by all metrics)

#### FN (False Negatives)
**Definition:** Number of GT boxes not matched to any prediction (missed detections).
```
FN = total_GT - TP
```
**Status:** ✅ Implemented (returned by all metrics)

#### TP (True Positives)
**Definition:** Number of correctly matched prediction-GT pairs.
```
TP = number of matches with IoU ≥ threshold
```
**Status:** ✅ Implemented (returned by all metrics)

#### CLR_TP, CLR_FP, CLR_FN
**Definition:** CLEAR MOT specific counts for true positives, false positives, false negatives.  
**Status:** ⚪ Not implemented (equivalent to standard TP/FP/FN)  
**Used in:** TrackEval implementation

### Temporal Metrics

#### TID (Track Initialization Duration)
**Formula:**
```
TID = (1 / num_tracks) × Σ(frames_until_first_detection)
```
**Interpretation:** Average number of frames before a GT track is first detected. Lower is better.  
**Status:** ✅ Implemented (via `calculate_tid_lgd`)  
**Used in:** nuScenes

**Usage:**
```python
from admetrics.tracking import calculate_tid_lgd

result = calculate_tid_lgd(predictions, ground_truth, iou_threshold=0.5)
print(f"TID: {result['tid']:.2f} frames")
print(f"LGD: {result['lgd']:.2f} frames")
print(f"Detected Tracks: {result['num_detected_tracks']}/{result['num_tracks']}")
```

#### LGD (Longest Gap Duration)
**Formula:**
```
LGD = (1 / num_tracks) × Σ(max_consecutive_missed_frames)
```
**Interpretation:** Average longest gap in tracking. Lower is better.  
**Status:** ✅ Implemented (via `calculate_tid_lgd`)  
**Used in:** nuScenes

#### Track Lifetime / Duration
**Definition:** Average or median duration of tracked trajectories.  
**Status:** ✅ Implemented (via `calculate_trajectory_metrics` - `avg_track_length`)  
**Used in:** Various research papers

### VACE Metrics

From "A system for video surveillance and monitoring" (VSAM).

#### ATA (Average Tracking Accuracy)
**Formula:**
```
ATA = STDA / (0.5 × (GT_IDs + Pred_IDs))
where STDA = Spatio-Temporal Detection Accuracy
```
**Status:** ⚪ Not implemented  
**Used in:** VACE benchmark

#### SFDA (Sequence Frame Detection Accuracy)
**Formula:**
```
SFDA = FDA / num_non_empty_timesteps
```
**Status:** ⚪ Not implemented  
**Used in:** VACE benchmark

### Advanced Metrics

#### Track mAP (Mean Average Precision for Tracks)
**Definition:** Average Precision metric extended to track-level evaluation.  
**Status:** ⚪ Not implemented  
**Used in:** TAO dataset, BURST benchmark  
**Reference:** Dave et al., ECCV 2020

#### J&F (Jaccard and F-measure)
**Used for:** Segmentation-based tracking (MOTS)  
**Metrics:**
- **J**: Jaccard Index (IoU) for segmentation masks
- **F**: F-measure for boundary accuracy
- **J&F**: Combined metric = (J + F) / 2

**Status:** ⚪ Not implemented  
**Used in:** DAVIS, MOTS Challenge, YouTube-VIS

#### MOTAL (MOTA with Logarithmic ID Switches)
**Formula:**
```
MOTAL = 1 - (FP + FN + log₁₀(IDSW + 1)) / GT
```
**Interpretation:** Logarithmic penalty for ID switches instead of linear.  
**Status:** ✅ Implemented  
**Used in:** Some MOTChallenge variants

**Usage:**
```python
from admetrics.tracking import calculate_motal

result = calculate_motal(predictions, ground_truth, iou_threshold=0.5)
print(f"MOTAL: {result['motal']:.4f}")
print(f"MOTA: {result['mota']:.4f}")
print(f"ID Switches: {result['id_switches']} (log: {result['log_id_switches']:.2f})")
```

#### CLR_F1 (CLEAR F1)
**Formula:**
```
CLR_F1 = TP / (TP + 0.5×FN + 0.5×FP)
CLR_Re = TP / (TP + FN)  # CLEAR Recall
CLR_Pr = TP / (TP + FP)  # CLEAR Precision
```
**Interpretation:** F1 score for detection quality.  
**Status:** ✅ Implemented (via `calculate_clr_metrics`)

**Usage:**
```python
from admetrics.tracking import calculate_clr_metrics

result = calculate_clr_metrics(predictions, ground_truth, iou_threshold=0.5)
print(f"CLR Recall: {result['clr_re']:.4f}")
print(f"CLR Precision: {result['clr_pr']:.4f}")
print(f"CLR F1: {result['clr_f1']:.4f}")
```

#### OWTA (Open World Tracking Accuracy)
**Formula:**
```
OWTA = √(DetRe × AssA)
```
**Status:** ✅ Implemented  
**Used in:** Open World Tracking benchmarks

**Usage:**
```python
from admetrics.tracking import calculate_owta

result = calculate_owta(predictions, ground_truth, iou_threshold=0.5)
print(f"OWTA: {result['owta']:.4f}")
print(f"Detection Recall: {result['det_re']:.4f}")
print(f"Association Accuracy: {result['ass_a']:.4f}")
```

### Auxiliary Metrics

#### GT (Ground Truth Count)
**Definition:** Total number of ground truth objects/boxes.  
**Status:** ✅ Implemented (returned by most metrics)

#### Dets (Detections Count)
**Definition:** Total number of predicted objects/boxes.  
**Status:** ✅ Implemented (returned by most metrics)

#### IDs (Unique Identities)
**Definition:** Number of unique track IDs in predictions or GT.  
**Status:** ✅ Implemented (tracked internally)

#### FPS (Frames Per Second)
**Definition:** Processing speed of the tracker.  
**Status:** ⚪ Not a metric function (benchmark separately)  
**Used in:** All benchmarks for runtime evaluation

---

## Metric Comparison

### Implemented Metrics Summary

| Metric | Range | Best Value | Measures | Use Case |
|--------|-------|------------|----------|----------|
| **MOTA** | -∞ to 1 | Higher | Overall tracking + detection | Standard benchmark |
| **MOTP** | 0 to ∞ | Lower | Localization precision | Complementary to MOTA |
| **MODA** | -∞ to 1 | Higher | Detection-only accuracy | Isolate detection errors |
| **HOTA** | 0 to 1 | Higher | Balanced det + assoc | Comparing trackers |
| **IDF1** | 0 to 1 | Higher | Identity consistency | ID switch analysis |
| **AMOTA** | -∞ to 1 | Higher | Robust multi-threshold | nuScenes benchmark |
| **MOTAR** | -∞ to 1 | Higher | Fixed recall point | Matched comparison |
| **FAF** | 0 to ∞ | Lower | False alarms per frame | Noise measurement |
| **Track Recall** | 0 to 1 | Higher | Track detection rate | Track-level quality |
| **MT/ML/PT** | 0 to 1 | Higher MT | Trajectory coverage | Track lifespan analysis |
| **Detection F1** | 0 to 1 | Higher | Pure detection quality | Frame-level detection |
| **sMOTA** | -∞ to 1 | Higher | Soft matching quality | Segmentation tracking |
| **Completeness** | 0 to 1 | Higher | GT coverage | Detection consistency |
| **Purity** | 0 to 1 | Higher | Identity preservation | Track contamination |

---

## Benchmark-Specific Metrics

### Primary Metrics by Benchmark

| Benchmark | Primary Metric | Secondary Metrics | Status |
|-----------|---------------|-------------------|--------|
| **MOTChallenge** | HOTA, MOTA | IDF1, MT, ML, FP, FN, IDSW, Frag, MOTP | ✅ Full support |
| **nuScenes** | AMOTA | AMOTP, MOTA, MOTP, Recall, FAF, MT, ML, TID, LGD | ✅ Full support |
| **KITTI Tracking** | MOTA | MOTP, MT, ML, FP, FN, IDSW, Frag | ✅ Full support |
| **KITTI MOTS** | HOTA | sMOTSA, MOTSP, DetA, AssA | ✅ HOTA/sMOTA support |
| **Waymo** | MOTA/MOTP | Similar to KITTI | ✅ Full support |
| **Argoverse** | MOTA/MOTP | MT, ML, IDF1, IDSW | ✅ Full support |

---

## Choosing the Right Metric

- **For general benchmarking:** Use MOTA (most established, widely used)
- **For nuScenes benchmark:** Use AMOTA (primary metric)
- **For balanced evaluation:** Use HOTA (better interpretability)
- **For ID consistency:** Use IDF1 (directly measures ID preservation)
- **For localization quality:** Use MOTP (complementary to accuracy metrics)
- **For detection quality only:** Use MODA or Detection F1 (isolates detection from tracking)
- **For false positive analysis:** Use FAF (measures noise)
- **For track-level analysis:** Use Track Recall/Precision
- **For trajectory lifespan:** Use MT/ML/PT ratios (understand track coverage distribution)
- **For segmentation tracking:** Use sMOTA (handles continuous similarity)
- **For GT coverage analysis:** Use Completeness (detect systematic misses)
- **For identity purity:** Use Track Purity/Completeness (detect ID contamination)

## Common Patterns

### Perfect Tracking

```python
# All objects correctly detected with consistent IDs
result = calculate_multi_frame_mota(perfect_preds, ground_truth)
# MOTA = 1.0, num_switches = 0, num_fragmentations = 0
```

### ID Switch Detection

```python
# Track 1 switches to Track 2 for same object
result = calculate_multi_frame_mota(preds_with_switch, ground_truth)
# num_switches > 0, IDF1 decreases
```

### Fragmentation Detection

```python
# Track lost in middle frames then recovered
result = calculate_multi_frame_mota(preds_with_gaps, ground_truth)
# num_fragmentations > 0, affects trajectory classification
```

## Implementation Details

### Matching Algorithm

All metrics use Hungarian algorithm for optimal bipartite matching:
- Constructs IoU cost matrix for each frame
- Filters by class (only same-class matches)
- Uses `scipy.optimize.linear_sum_assignment`
- Threshold filtering after optimal assignment

### ID Switch Detection

ID switches are detected by tracking GT→Pred mapping across frames:
```python
if gt_id in prev_gt_to_pred:
    if prev_gt_to_pred[gt_id] != pred_id:
        # ID switch detected
```

### Fragmentation Counting

Fragmentations occur when a track is interrupted:
```python
# matched → unmatched → matched = 1 fragmentation
for track in gt_tracks:
    if has_gap_then_recovery(track):
        fragmentations += 1
```

---

## Complete Metric Taxonomy

### Detection Quality Metrics
- ✅ **TP, FP, FN** - True/False Positives, False Negatives
- ✅ **Precision, Recall** - Frame-level detection metrics
- ✅ **DetA, DetRe, DetPr** - HOTA detection components
- ✅ **MOTP, LocA** - Localization accuracy
- ⚪ **CLR_F1** - CLEAR F1 score

### Association Quality Metrics
- ✅ **IDSW** - ID Switches
- ✅ **IDF1, IDP, IDR** - ID F1, Precision, Recall
- ✅ **AssA, AssRe, AssPr** - HOTA association components
- ✅ **Frag** - Fragmentations
- ✅ **Track Purity/Completeness** - Identity preservation

### Combined Metrics
- ✅ **MOTA** - Multiple Object Tracking Accuracy
- ✅ **MODA** - Detection-only MOTA
- ✅ **sMOTA** - Soft MOTA for segmentation
- ✅ **HOTA** - Higher Order Tracking Accuracy
- ✅ **AMOTA, AMOTP** - Average MOTA/MOTP
- ✅ **MOTAR** - MOTA at Recall
- ⚪ **MOTAL** - MOTA with logarithmic ID switches

### Trajectory-Level Metrics
- ✅ **MT, ML, PT** - Mostly Tracked/Lost/Partially (counts and ratios)
- ✅ **Frag** - Fragmentations
- ⚪ **TID** - Track Initialization Duration
- ⚪ **LGD** - Longest Gap Duration
- ✅ **Track Lifetime** - Average track duration

### Frame-Level Metrics
- ✅ **FAF** - False Alarms per Frame
- ✅ **FP_per_frame** - False positives per frame
- ✅ **Detection Density** - Average detections per frame
- ✅ **Frame Coverage** - Percentage of frames with detections

### Specialized Metrics
- ⚪ **Track mAP** - Track-level mean Average Precision
- ⚪ **J&F** - Jaccard and F-measure for segmentation
- ⚪ **ATA, SFDA** - VACE metrics
- ⚪ **OWTA** - Open World Tracking Accuracy

### Formula Reference

#### Basic Counts per Frame
```python
For each frame t:
    TP_t = number of matched predictions
    FP_t = number of unmatched predictions  
    FN_t = number of unmatched GT boxes
    IDSW_t = number of ID switches in frame t

Aggregate:
    TP = Σ TP_t
    FP = Σ FP_t
    FN = Σ FN_t
    IDSW = Σ IDSW_t
    GT = Σ (number of GT boxes in frame t)
```

#### ID Switch Detection
```python
For each matched GT trajectory:
    if previous_pred_id ≠ current_pred_id:
        IDSW += 1
```

#### Fragmentation Counting
```python
For each GT track:
    gaps = 0
    for each frame in track:
        if was_matched and now_unmatched:
            gaps += 1
    Frag += max(0, gaps - 1)
```

#### Multi-Threshold Evaluation
- **AMOTA/AMOTP:** Average over 40 recall thresholds (typically)
- **HOTA:** Evaluated at multiple IoU thresholds (0.05 to 0.95)
- **Traditional MOTA:** Single threshold, typically 0.5 IoU

#### Common Threshold Values
- **2D Tracking:** IoU ≥ 0.5
- **3D Tracking:** IoU3D ≥ 0.25 or center distance ≤ 2.0m
- **Pedestrian:** Often stricter thresholds

---

## Examples

See `examples/tracking_evaluation.py` for comprehensive usage examples demonstrating:
- Multi-frame sequence tracking
- ID switch scenarios
- Fragmentation detection
- Trajectory classification
- Error analysis

## Testing

Comprehensive test suite in `tests/test_tracking.py`:
- **36 test cases** covering all implemented metrics
- Single-frame and multi-frame scenarios
- Perfect tracking and error cases
- ID switches, fragmentations, and trajectory classification
- **92% code coverage** on tracking.py

Run tests:
```bash
pytest tests/test_tracking.py -v
pytest tests/test_tracking.py --cov=admetrics/tracking --cov-report=term-missing
```

---

## References

### Core Papers

1. **Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics**
   - Bernardin, K., & Stiefelhagen, R. (2008)
   - EURASIP Journal on Image and Video Processing, 2008
   - https://link.springer.com/article/10.1155/2008/246309
   - https://doi.org/10.1155/2008/246309
   - **Introduced:** MOTA, MOTP, foundational tracking metrics

2. **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking**
   - Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2020)
   - International Journal of Computer Vision (IJCV), 2020
   - https://arxiv.org/abs/2009.07736
   - https://github.com/JonathonLuiten/TrackEval
   - https://doi.org/10.1007/s11263-020-01375-2
   - **Introduced:** HOTA, DetA, AssA, LocA - unified tracking evaluation

3. **Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking**
   - Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016)
   - ECCV 2016 Workshop on Benchmarking Multi-Target Tracking
   - https://arxiv.org/abs/1609.01775
   - https://github.com/ergysr/DeepCC
   - **Introduced:** IDF1 (ID F1 Score) for identity-preserving tracking

4. **A Baseline for 3D Multi-Object Tracking**
   - Weng, X., Wang, J., Held, D., & Kitani, K. (2019)
   - https://arxiv.org/abs/1907.03961
   - **Introduced:** AMOTA, AMOTP for nuScenes

5. **Global Data Association for Multi-Object Tracking Using Network Flows**
   - Nevatia, R., et al. (2008)
   - CVPR 2008
   - **Introduced:** MT/ML/PT trajectory classification

### Benchmarks and Datasets

6. **MOTChallenge**  
   - https://motchallenge.net/
   - Primary benchmark for multi-object tracking
   - Uses HOTA, MOTA, IDF1 as primary metrics

7. **nuScenes Tracking Benchmark**  
   - https://www.nuscenes.org/tracking
   - 3D tracking benchmark for autonomous driving
   - Uses AMOTA as primary metric

8. **KITTI Tracking Benchmark**  
   - http://www.cvlibs.net/datasets/kitti/eval_tracking.php
   - 3D tracking for autonomous driving
   - Uses MOTA, MOTP, MT/ML

9. **Waymo Open Dataset**  
   - https://waymo.com/open/
   - Large-scale autonomous driving dataset

10. **Argoverse**  
    - https://www.argoverse.org/
    - Autonomous driving with 3D tracking

### Implementation References

11. **TrackEval - Official Tracking Evaluation Code**  
    - https://github.com/JonathonLuiten/TrackEval
    - Reference implementation for HOTA, CLEAR MOT, IDF1

12. **py-motmetrics**  
    - https://github.com/cheind/py-motmetrics
    - Python implementation of MOTChallenge metrics

13. **nuScenes devkit**  
    - https://github.com/nutonomy/nuscenes-devkit
    - Official nuScenes evaluation tools

### Additional Papers

14. **TAO: A Large-Scale Benchmark for Tracking Any Object** (Dave et al., ECCV 2020)  
    - https://arxiv.org/abs/2005.10356
    - Introduced Track mAP

15. **MOTS: Multi-Object Tracking and Segmentation** (Voigtlaender et al., CVPR 2019)  
    - https://arxiv.org/abs/1902.03604
    - Tracking with pixel-level segmentation

---

## Implementation Status

### ✅ Fully Implemented (21 metrics)
- MOTA, MOTP, MODA, MOTAL
- HOTA, DetA, AssA, LocA (with all sub-components)
- IDF1, IDP, IDR
- AMOTA, AMOTP, MOTAR
- FAF, Track Recall/Precision
- MT/ML/PT ratios
- Detection Metrics (P/R/F1)
- sMOTA
- Completeness Metrics
- Identity Metrics (Purity, Completeness, Switches, Fragmentation)
- TID/LGD (nuScenes temporal metrics)
- CLR_Re, CLR_Pr, CLR_F1 (CLEAR MOT metrics)
- OWTA (Open World Tracking)

### ⚪ Not Yet Implemented
- Track mAP
- J&F (segmentation metrics)
- VACE metrics (ATA, SFDA)

### 📊 Coverage
- **MOTChallenge:** 100% of core metrics
- **nuScenes:** 100% (all primary and secondary metrics)
- **KITTI:** 100%
- **HOTA Framework:** 100%
- **Open World Tracking:** 100%

---

This comprehensive reference covers all major tracking metrics used across autonomous driving and computer vision benchmarks as of 2025.
