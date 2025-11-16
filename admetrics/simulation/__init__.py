"""
Simulation Quality Metrics for Autonomous Driving.

This module provides metrics for evaluating the quality and realism of simulated
sensor data in autonomous vehicle simulators (CARLA, LGSVL, AirSim, etc.).
"""

from .sensor_quality import (
    calculate_camera_image_quality,
    calculate_lidar_point_cloud_quality,
    calculate_radar_quality,
    calculate_sensor_noise_characteristics,
    calculate_multimodal_sensor_alignment,
    calculate_temporal_consistency,
    calculate_perception_sim2real_gap,
)

__all__ = [
    'calculate_camera_image_quality',
    'calculate_lidar_point_cloud_quality',
    'calculate_radar_quality',
    'calculate_sensor_noise_characteristics',
    'calculate_multimodal_sensor_alignment',
    'calculate_temporal_consistency',
    'calculate_perception_sim2real_gap',
]
