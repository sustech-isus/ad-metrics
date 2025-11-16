"""Detection metrics for 3D object detection evaluation."""

from admetrics.detection.iou import (
    calculate_iou_3d,
    calculate_iou_bev,
    calculate_iou_batch,
    calculate_giou_3d,
)
from admetrics.detection.ap import (
    calculate_ap,
    calculate_map,
    calculate_ap_coco_style,
    calculate_precision_recall_curve,
)
from admetrics.detection.nds import (
    calculate_nds,
    calculate_nds_detailed,
    calculate_tp_metrics,
)
from admetrics.detection.aos import (
    calculate_aos,
    calculate_aos_per_difficulty,
    calculate_orientation_similarity,
)
from admetrics.detection.confusion import (
    calculate_confusion_metrics,
    calculate_tp_fp_fn,
    calculate_confusion_matrix_multiclass,
    calculate_specificity,
)
from admetrics.detection.distance import (
    calculate_center_distance,
    calculate_orientation_error,
    calculate_size_error,
    calculate_velocity_error,
    calculate_average_distance_error,
    calculate_translation_error_bins,
)

__all__ = [
    # IoU metrics
    "calculate_iou_3d",
    "calculate_iou_bev",
    "calculate_iou_batch",
    "calculate_giou_3d",
    # AP metrics
    "calculate_ap",
    "calculate_map",
    "calculate_ap_coco_style",
    "calculate_precision_recall_curve",
    # NuScenes metrics
    "calculate_nds",
    "calculate_nds_detailed",
    "calculate_tp_metrics",
    # KITTI metrics
    "calculate_aos",
    "calculate_aos_per_difficulty",
    "calculate_orientation_similarity",
    # Confusion matrix metrics
    "calculate_confusion_metrics",
    "calculate_tp_fp_fn",
    "calculate_confusion_matrix_multiclass",
    "calculate_specificity",
    # Distance/error metrics
    "calculate_center_distance",
    "calculate_orientation_error",
    "calculate_size_error",
    "calculate_velocity_error",
    "calculate_average_distance_error",
    "calculate_translation_error_bins",
]
