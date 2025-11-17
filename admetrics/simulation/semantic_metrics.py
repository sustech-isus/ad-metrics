"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def calculate_semantic_consistency(
    sim_scene_data: Dict[str, np.ndarray],
    real_scene_data: Dict[str, np.ndarray],
    scene_type: str = 'mixed',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Validate semantic consistency and realism of scene composition.
    
    Evaluates whether simulated scenes have realistic object distributions,
    scene layouts, and traffic patterns compared to real-world data.
    
    Args:
        sim_scene_data: Simulated scene annotations with:
            - 'object_classes': Array of object class IDs (N,) 
            - 'object_counts': Dict of class counts {'car': 10, 'pedestrian': 5, ...}
            - 'vehicle_speeds': Array of vehicle speeds (M,) in m/s
            - 'lane_positions': Array of lateral lane positions (M,)
            - 'inter_vehicle_distances': Array of distances between vehicles (K,) in meters
            - 'pedestrian_speeds': Array of pedestrian speeds (P,) in m/s
        real_scene_data: Real-world scene data, same format
        scene_type: Type of scene. Options:
            - 'highway': Highway/freeway scenarios
            - 'urban': City streets and intersections
            - 'suburban': Residential areas
            - 'mixed': Combined scenarios
        metrics: List of metrics to compute. Options:
            - 'object_distribution': Object class distribution matching
            - 'vehicle_behavior': Vehicle speed and spacing
            - 'pedestrian_behavior': Pedestrian motion patterns
            - 'traffic_density': Overall traffic density
            - 'all': Compute all metrics
            
    Returns:
        Dictionary with metrics:
            - object_distribution_kl: KL divergence of object class distributions
            - vehicle_count_ratio: Ratio of sim/real vehicle counts
            - pedestrian_count_ratio: Ratio of sim/real pedestrian counts
            - vehicle_speed_kl: KL divergence of vehicle speed distributions
            - vehicle_speed_mean_error: Mean speed difference (m/s)
            - inter_vehicle_distance_kl: KL divergence of spacing distributions
            - inter_vehicle_distance_mean: Mean spacing difference (m)
            - pedestrian_speed_kl: KL divergence of pedestrian speeds
            - lane_position_kl: KL divergence of lane usage
            - traffic_density_ratio: Overall density comparison
            - overall_semantic_score: Composite realism score (0-100)
    """
    results = {}
    
    if metrics is None or metrics == 'all':
        metrics = ['object_distribution', 'vehicle_behavior', 'pedestrian_behavior', 'traffic_density']
    
    # 1. Object Distribution Validation
    if 'object_distribution' in metrics:
        if 'object_classes' in sim_scene_data and 'object_classes' in real_scene_data:
            sim_classes = np.array(sim_scene_data['object_classes'])
            real_classes = np.array(real_scene_data['object_classes'])
            
            # Get unique classes
            all_classes = np.union1d(np.unique(sim_classes), np.unique(real_classes))
            
            if len(all_classes) > 0:
                # Create histograms
                sim_hist = np.histogram(sim_classes, bins=len(all_classes), 
                                       range=(all_classes.min(), all_classes.max()+1))[0]
                real_hist = np.histogram(real_classes, bins=len(all_classes),
                                        range=(all_classes.min(), all_classes.max()+1))[0]
                
                # Normalize
                sim_dist = (sim_hist + 1e-10) / (sim_hist.sum() + 1e-10 * len(sim_hist))
                real_dist = (real_hist + 1e-10) / (real_hist.sum() + 1e-10 * len(real_hist))
                
                # KL divergence
                kl_div = float(np.sum(real_dist * np.log((real_dist + 1e-10) / (sim_dist + 1e-10))))
                results['object_distribution_kl'] = kl_div
        
        # Object count ratios
        if 'object_counts' in sim_scene_data and 'object_counts' in real_scene_data:
            sim_counts = sim_scene_data['object_counts']
            real_counts = real_scene_data['object_counts']
            
            # Vehicle ratio
            sim_vehicles = sim_counts.get('car', 0) + sim_counts.get('truck', 0) + sim_counts.get('bus', 0)
            real_vehicles = real_counts.get('car', 0) + real_counts.get('truck', 0) + real_counts.get('bus', 0)
            if real_vehicles > 0:
                results['vehicle_count_ratio'] = float(sim_vehicles / real_vehicles)
            
            # Pedestrian ratio
            sim_peds = sim_counts.get('pedestrian', 0)
            real_peds = real_counts.get('pedestrian', 0)
            if real_peds > 0:
                results['pedestrian_count_ratio'] = float(sim_peds / real_peds)
    
    # 2. Vehicle Behavior Validation
    if 'vehicle_behavior' in metrics:
        # Speed distribution
        if 'vehicle_speeds' in sim_scene_data and 'vehicle_speeds' in real_scene_data:
            sim_speeds = np.array(sim_scene_data['vehicle_speeds'])
            real_speeds = np.array(real_scene_data['vehicle_speeds'])
            
            if len(sim_speeds) > 0 and len(real_speeds) > 0:
                # Speed distribution matching
                bins = np.linspace(0, max(sim_speeds.max(), real_speeds.max()), 20)
                sim_hist = np.histogram(sim_speeds, bins=bins)[0]
                real_hist = np.histogram(real_speeds, bins=bins)[0]
                
                sim_dist = (sim_hist + 1e-10) / (sim_hist.sum() + 1e-10 * len(sim_hist))
                real_dist = (real_hist + 1e-10) / (real_hist.sum() + 1e-10 * len(real_hist))
                
                kl_div = float(np.sum(real_dist * np.log((real_dist + 1e-10) / (sim_dist + 1e-10))))
                results['vehicle_speed_kl'] = kl_div
                results['vehicle_speed_mean_error'] = float(np.abs(sim_speeds.mean() - real_speeds.mean()))
        
        # Inter-vehicle distance
        if 'inter_vehicle_distances' in sim_scene_data and 'inter_vehicle_distances' in real_scene_data:
            sim_dists = np.array(sim_scene_data['inter_vehicle_distances'])
            real_dists = np.array(real_scene_data['inter_vehicle_distances'])
            
            if len(sim_dists) > 0 and len(real_dists) > 0:
                # Distance distribution
                max_dist = max(sim_dists.max(), real_dists.max())
                bins = np.linspace(0, max_dist, 15)
                sim_hist = np.histogram(sim_dists, bins=bins)[0]
                real_hist = np.histogram(real_dists, bins=bins)[0]
                
                sim_dist = (sim_hist + 1e-10) / (sim_hist.sum() + 1e-10 * len(sim_hist))
                real_dist = (real_hist + 1e-10) / (real_hist.sum() + 1e-10 * len(real_hist))
                
                kl_div = float(np.sum(real_dist * np.log((real_dist + 1e-10) / (sim_dist + 1e-10))))
                results['inter_vehicle_distance_kl'] = kl_div
                results['inter_vehicle_distance_mean'] = float(np.abs(sim_dists.mean() - real_dists.mean()))
        
        # Lane position distribution
        if 'lane_positions' in sim_scene_data and 'lane_positions' in real_scene_data:
            sim_lanes = np.array(sim_scene_data['lane_positions'])
            real_lanes = np.array(real_scene_data['lane_positions'])
            
            if len(sim_lanes) > 0 and len(real_lanes) > 0:
                bins = np.linspace(min(sim_lanes.min(), real_lanes.min()),
                                  max(sim_lanes.max(), real_lanes.max()), 10)
                sim_hist = np.histogram(sim_lanes, bins=bins)[0]
                real_hist = np.histogram(real_lanes, bins=bins)[0]
                
                sim_dist = (sim_hist + 1e-10) / (sim_hist.sum() + 1e-10 * len(sim_hist))
                real_dist = (real_hist + 1e-10) / (real_hist.sum() + 1e-10 * len(real_hist))
                
                kl_div = float(np.sum(real_dist * np.log((real_dist + 1e-10) / (sim_dist + 1e-10))))
                results['lane_position_kl'] = kl_div
    
    # 3. Pedestrian Behavior Validation
    if 'pedestrian_behavior' in metrics:
        if 'pedestrian_speeds' in sim_scene_data and 'pedestrian_speeds' in real_scene_data:
            sim_ped_speeds = np.array(sim_scene_data['pedestrian_speeds'])
            real_ped_speeds = np.array(real_scene_data['pedestrian_speeds'])
            
            if len(sim_ped_speeds) > 0 and len(real_ped_speeds) > 0:
                # Pedestrian speed distribution (typical range 0-3 m/s)
                bins = np.linspace(0, 3.5, 15)
                sim_hist = np.histogram(sim_ped_speeds, bins=bins)[0]
                real_hist = np.histogram(real_ped_speeds, bins=bins)[0]
                
                sim_dist = (sim_hist + 1e-10) / (sim_hist.sum() + 1e-10 * len(sim_hist))
                real_dist = (real_hist + 1e-10) / (real_hist.sum() + 1e-10 * len(real_hist))
                
                kl_div = float(np.sum(real_dist * np.log((real_dist + 1e-10) / (sim_dist + 1e-10))))
                results['pedestrian_speed_kl'] = kl_div
                results['pedestrian_speed_mean_error'] = float(np.abs(sim_ped_speeds.mean() - real_ped_speeds.mean()))
    
    # 4. Traffic Density
    if 'traffic_density' in metrics:
        if 'object_counts' in sim_scene_data and 'object_counts' in real_scene_data:
            sim_total = sum(sim_scene_data['object_counts'].values())
            real_total = sum(real_scene_data['object_counts'].values())
            
            if real_total > 0:
                results['traffic_density_ratio'] = float(sim_total / real_total)
    
    # 5. Overall Semantic Score
    score_components = []
    
    # Object distribution (lower KL = better)
    if 'object_distribution_kl' in results:
        # KL < 0.3 = excellent, 0.3-0.8 = good, > 1.0 = poor
        dist_score = 100 * (1 - np.clip(results['object_distribution_kl'] / 1.0, 0, 1))
        score_components.append(dist_score)
    
    # Count ratios (closer to 1.0 = better)
    if 'vehicle_count_ratio' in results:
        ratio_error = abs(results['vehicle_count_ratio'] - 1.0)
        ratio_score = 100 * (1 - np.clip(ratio_error / 0.5, 0, 1))
        score_components.append(ratio_score)
    
    # Speed distribution (lower KL = better)
    if 'vehicle_speed_kl' in results:
        speed_score = 100 * (1 - np.clip(results['vehicle_speed_kl'] / 0.5, 0, 1))
        score_components.append(speed_score)
    
    # Inter-vehicle distance (lower KL = better)
    if 'inter_vehicle_distance_kl' in results:
        spacing_score = 100 * (1 - np.clip(results['inter_vehicle_distance_kl'] / 0.5, 0, 1))
        score_components.append(spacing_score)
    
    # Pedestrian speeds (lower KL = better)
    if 'pedestrian_speed_kl' in results:
        ped_score = 100 * (1 - np.clip(results['pedestrian_speed_kl'] / 0.5, 0, 1))
        score_components.append(ped_score)
    
    if score_components:
        results['overall_semantic_score'] = float(np.mean(score_components))
    
    return results

