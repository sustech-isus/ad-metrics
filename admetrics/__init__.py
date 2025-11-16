"""
Metrics for Autonomous Driving

This package provides comprehensive metrics for evaluating autonomous driving systems,
including 3D object detection, multi-object tracking, trajectory prediction, 
ego vehicle localization, 3D occupancy prediction, end-to-end planning,
simulation quality assessment, and HD map vector detection.
"""

__version__ = "0.1.0"
__author__ = "AD-Metrics Contributors"

# Detection metrics
from admetrics.detection import (
    calculate_iou_3d,
    calculate_iou_bev,
    calculate_iou_batch,
    calculate_giou_3d,
    calculate_ap,
    calculate_map,
    calculate_ap_coco_style,
    calculate_precision_recall_curve,
    calculate_nds,
    calculate_nds_detailed,
    calculate_tp_metrics,
    calculate_aos,
    calculate_aos_per_difficulty,
    calculate_orientation_similarity,
    calculate_confusion_metrics,
    calculate_tp_fp_fn,
    calculate_confusion_matrix_multiclass,
    calculate_specificity,
    calculate_center_distance,
    calculate_orientation_error,
    calculate_size_error,
    calculate_velocity_error,
    calculate_average_distance_error,
    calculate_translation_error_bins,
)

# Tracking metrics
from admetrics.tracking import (
    calculate_mota,
    calculate_motp,
    calculate_clearmot_metrics,
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_id_f1,
)

# Trajectory prediction metrics
from admetrics.prediction import (
    calculate_ade,
    calculate_fde,
    calculate_miss_rate,
    calculate_multimodal_ade,
    calculate_multimodal_fde,
    calculate_brier_fde,
    calculate_nll,
    calculate_collision_rate,
    calculate_drivable_area_compliance,
    calculate_trajectory_metrics,
)

# Localization metrics
from admetrics.localization import (
    calculate_ate,
    calculate_rte,
    calculate_are,
    calculate_lateral_error,
    calculate_longitudinal_error,
    calculate_convergence_rate,
    calculate_localization_metrics,
    calculate_map_alignment_score,
)

# Occupancy metrics
from admetrics.occupancy import (
    calculate_occupancy_iou,
    calculate_mean_iou,
    calculate_occupancy_precision_recall,
    calculate_scene_completion,
    calculate_chamfer_distance,
    calculate_surface_distance,
)

# End-to-end planning metrics
from admetrics.planning import (
    calculate_l2_distance,
    calculate_collision_rate,
    calculate_progress_score,
    calculate_route_completion,
    average_displacement_error_planning,
    calculate_lateral_deviation,
    calculate_heading_error,
    calculate_velocity_error,
    calculate_comfort_metrics,
    calculate_driving_score,
    calculate_planning_kl_divergence,
)

# Simulation quality metrics
from admetrics.simulation import (
    calculate_camera_image_quality,
    calculate_lidar_point_cloud_quality,
    calculate_radar_quality,
    calculate_sensor_noise_characteristics,
    calculate_multimodal_sensor_alignment,
    calculate_temporal_consistency,
    calculate_perception_sim2real_gap,
)

# Vector map detection metrics
from admetrics.vectormap import (
    calculate_chamfer_distance_polyline,
    calculate_frechet_distance,
    calculate_polyline_iou,
    calculate_lane_detection_metrics,
    calculate_topology_metrics,
    calculate_endpoint_error,
    calculate_direction_accuracy,
    calculate_vectormap_ap,
)

__all__ = [
    # Detection metrics - IoU
    "calculate_iou_3d",
    "calculate_iou_bev",
    "calculate_iou_batch",
    "calculate_giou_3d",
    # Detection metrics - AP
    "calculate_ap",
    "calculate_map",
    "calculate_ap_coco_style",
    "calculate_precision_recall_curve",
    # Detection metrics - NuScenes
    "calculate_nds",
    "calculate_nds_detailed",
    "calculate_tp_metrics",
    # Detection metrics - KITTI
    "calculate_aos",
    "calculate_aos_per_difficulty",
    "calculate_orientation_similarity",
    # Detection metrics - Confusion matrix
    "calculate_confusion_metrics",
    "calculate_tp_fp_fn",
    "calculate_confusion_matrix_multiclass",
    "calculate_specificity",
    # Detection metrics - Distance/Error
    "calculate_center_distance",
    "calculate_orientation_error",
    "calculate_size_error",
    "calculate_velocity_error",
    "calculate_average_distance_error",
    "calculate_translation_error_bins",
    # Tracking metrics
    "calculate_mota",
    "calculate_motp",
    "calculate_clearmot_metrics",
    "calculate_multi_frame_mota",
    "calculate_hota",
    "calculate_id_f1",
    # Trajectory prediction metrics
    "calculate_ade",
    "calculate_fde",
    "calculate_miss_rate",
    "calculate_multimodal_ade",
    "calculate_multimodal_fde",
    "calculate_brier_fde",
    "calculate_nll",
    "calculate_collision_rate",
    "calculate_drivable_area_compliance",
    "calculate_trajectory_metrics",
    # Localization metrics
    "calculate_ate",
    "calculate_rte",
    "calculate_are",
    "calculate_lateral_error",
    "calculate_longitudinal_error",
    "calculate_convergence_rate",
    "calculate_localization_metrics",
    "calculate_map_alignment_score",
    # Occupancy metrics
    "calculate_occupancy_iou",
    "calculate_mean_iou",
    "calculate_occupancy_precision_recall",
    "calculate_scene_completion",
    "calculate_chamfer_distance",
    "calculate_surface_distance",
    # End-to-end planning metrics
    "calculate_l2_distance",
    "calculate_collision_rate",
    "calculate_progress_score",
    "calculate_route_completion",
    "average_displacement_error_planning",
    "calculate_lateral_deviation",
    "calculate_heading_error",
    "calculate_velocity_error",
    "calculate_comfort_metrics",
    "calculate_driving_score",
    "calculate_planning_kl_divergence",
    # Simulation quality metrics
    "calculate_camera_image_quality",
    "calculate_lidar_point_cloud_quality",
    "calculate_radar_quality",
    "calculate_sensor_noise_characteristics",
    "calculate_multimodal_sensor_alignment",
    "calculate_temporal_consistency",
    "calculate_perception_sim2real_gap",
    # Vector map detection metrics
    "calculate_chamfer_distance_polyline",
    "calculate_frechet_distance",
    "calculate_polyline_iou",
    "calculate_lane_detection_metrics",
    "calculate_topology_metrics",
    "calculate_endpoint_error",
    "calculate_direction_accuracy",
    "calculate_vectormap_ap",
]
