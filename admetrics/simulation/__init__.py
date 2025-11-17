"""
Simulation Quality Metrics for Autonomous Driving.

This module provides metrics for evaluating the quality and realism of simulated
sensor data in autonomous vehicle simulators (CARLA, LGSVL, AirSim, etc.).
"""

from .camera_metrics import calculate_camera_image_quality
from .lidar_metrics import calculate_lidar_point_cloud_quality
from .radar_metrics import calculate_radar_quality
from .noise_metrics import calculate_sensor_noise_characteristics
from .alignment_metrics import calculate_multimodal_sensor_alignment
from .temporal_metrics import calculate_temporal_consistency
from .sim2real_metrics import calculate_perception_sim2real_gap
from .weather_metrics import calculate_weather_simulation_quality
from .dynamics_metrics import calculate_vehicle_dynamics_quality
from .semantic_metrics import calculate_semantic_consistency
from .occlusion_metrics import calculate_occlusion_visibility_quality

__all__ = [
    'calculate_camera_image_quality',
    'calculate_lidar_point_cloud_quality',
    'calculate_radar_quality',
    'calculate_sensor_noise_characteristics',
    'calculate_multimodal_sensor_alignment',
    'calculate_temporal_consistency',
    'calculate_perception_sim2real_gap',
    'calculate_weather_simulation_quality',
    'calculate_vehicle_dynamics_quality',
    'calculate_semantic_consistency',
    'calculate_occlusion_visibility_quality',
]
