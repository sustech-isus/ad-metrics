"""
Vector Map Detection Metrics for HD Mapping.

This module provides metrics for evaluating road vector map detection/extraction,
including lane line detection, road boundary detection, and topology estimation.
Used in HD mapping, online mapping, and map-based localization tasks.
"""

from .vectormap import (
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
    'calculate_chamfer_distance_polyline',
    'calculate_frechet_distance',
    'calculate_polyline_iou',
    'calculate_lane_detection_metrics',
    'calculate_topology_metrics',
    'calculate_endpoint_error',
    'calculate_direction_accuracy',
    'calculate_vectormap_ap',
]
