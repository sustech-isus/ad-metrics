# API Reference Update Summary

**Date**: November 17, 2025  
**File**: `/docs/api_reference.md`  
**Status**: ✅ Completed

## Overview

Updated the API reference documentation to reflect all new planning metrics functions and the cleaned API (without backward compatibility).

## Changes Made

### 1. Updated Planning Metrics Count
- **Before**: Planning Metrics (11)
- **After**: Planning Metrics (20)
- **Added**: 9 new functions

### 2. New Functions Documented

#### High-Priority Features (Previously Implemented):

1. **`calculate_collision_with_fault_classification()`**
   - At-fault collision classification (nuPlan/tuplan_garage style)
   - 5 collision types: active_front, stopped_track, active_lateral, active_rear, passive
   - Full parameter documentation with examples

2. **`calculate_time_to_collision_enhanced()`**
   - Enhanced TTC with forward projection (nuPlan style)
   - 1-second horizon, 0.3s intervals
   - Accounts for ego and obstacle motion
   - Full TTC profile return

3. **`calculate_distance_to_road_edge()`**
   - Signed distance to drivable area (Waymo Sim Agents style)
   - Supports Shapely polygons or lane centerline fallback
   - Violation detection and rate calculation

4. **`calculate_driving_direction_compliance()`**
   - Wrong-way driving detection (nuPlan style)
   - Compliance scoring with thresholds
   - nuPlan score mapping (1.0 = excellent, 0.5 = acceptable, 0.0 = failure)

5. **`calculate_interaction_metrics()`**
   - Multi-agent proximity analysis (Waymo Sim Agents style)
   - Min/mean distance tracking
   - Close interaction counting (<5m threshold)
   - Closest object identification

### 3. Updated Function Documentation

#### `calculate_comfort_metrics()` - Major Update:
- **New Parameters**:
  - `max_longitudinal_accel` (replaced `max_acceleration`)
  - `max_lateral_accel` - NEW
  - `max_yaw_rate` - NEW
  - `max_yaw_accel` - NEW
  - `include_lateral` - NEW (default: True)
  - `use_smoothing` - NEW (Savitzky-Golay filter)
  - `smoothing_window` - NEW (default: 15)
  - `smoothing_order` - NEW (default: 2)

- **New Return Fields**:
  - `mean_longitudinal_accel` (was `mean_acceleration`)
  - `max_longitudinal_accel` (was `max_acceleration`)
  - `mean_lateral_accel` - NEW
  - `max_lateral_accel` - NEW
  - `mean_yaw_rate` - NEW
  - `max_yaw_rate` - NEW
  - `mean_yaw_accel` - NEW
  - `max_yaw_accel` - NEW

- **Removed** (backward compatibility cleanup):
  - ❌ `mean_acceleration` field
  - ❌ `max_acceleration` field
  - ❌ `max_acceleration` parameter

#### `calculate_driving_score()` - Enhanced:
- **New Parameters**:
  - `timestamps` - Required for comfort metrics
  - `mode` - 'default' or 'nuplan'

- **nuPlan Mode Features**:
  - At-fault collision classification
  - Lateral comfort metrics
  - Drivable area compliance
  - Driving direction compliance
  - Industry-standard weights (0.25/0.40/0.20/0.15)

#### `average_displacement_error_planning()` - Enhanced:
- **New Parameter**:
  - `horizons` - List of timesteps for multi-horizon evaluation

- **New Return Fields**:
  - `ADE_H` - ADE at specific horizon H
  - `FDE_H` - FDE at specific horizon H

- **Example Added**: Multi-horizon evaluation (1s, 3s, 5s, 8s)

### 4. Documentation Quality Improvements

#### Added Examples for All New Functions:
```python
# Collision with fault classification
result = calculate_collision_with_fault_classification(
    ego_traj, ego_vels, ego_heads, other_vehicles
)

# Multi-horizon evaluation
result = average_displacement_error_planning(
    pred, expert, horizons=[10, 30, 50, 80]
)

# Comfort metrics with smoothing
result = calculate_comfort_metrics(
    traj, timestamps, use_smoothing=True
)

# Distance to road edge
result = calculate_distance_to_road_edge(
    traj, drivable_area=polygon
)

# Direction compliance
result = calculate_driving_direction_compliance(
    trajectory, headings, lane_centerline
)

# Interaction metrics
result = calculate_interaction_metrics(
    ego_traj, other_vehicles, close_distance_threshold=5.0
)
```

#### Enhanced Parameter Descriptions:
- All parameters now have units specified (m, m/s, rad, s)
- Default values clearly stated
- Expected array shapes documented
- Optional vs required parameters clarified

#### Enhanced Return Value Documentation:
- All dictionary keys explicitly listed
- Data types specified
- Value ranges indicated where applicable
- Semantic meaning explained

### 5. Industry Alignment Annotations

Added clear indicators for industry-standard implementations:
- **nuPlan style**: Collision classification, direction compliance, comfort thresholds
- **Waymo Sim Agents style**: Road edge distance, interaction metrics
- **tuplan_garage style**: Savitzky-Golay smoothing, collision types

## File Statistics

- **Total Lines**: 1937 (increased from ~1621)
- **New Content**: ~316 lines
- **Planning Metrics Section**: Now comprehensive with 20 functions
- **Code Examples**: Added/updated for all new functions

## Structure

### Planning Metrics Section Organization:

1. Basic Metrics (4 functions)
   - L2 distance
   - Collision rate
   - Progress score
   - Route completion

2. Trajectory Comparison (4 functions)
   - ADE/FDE (with multi-horizon)
   - Lateral deviation
   - Heading error
   - Velocity error

3. Comfort & Dynamics (2 functions)
   - Comfort metrics (comprehensive)
   - Driving score (with nuPlan mode)

4. Safety & Collision (4 functions)
   - Planning KL divergence
   - Basic TTC
   - Enhanced TTC
   - Lane invasion rate
   - Collision severity

5. Advanced Planning (6 functions - NEW)
   - Kinematic feasibility
   - Collision with fault classification ⭐
   - Distance to road edge ⭐
   - Driving direction compliance ⭐
   - Interaction metrics ⭐

⭐ = Newly documented function

## Validation

✅ All 20 planning functions documented  
✅ All parameters documented with types and defaults  
✅ All return values documented with descriptions  
✅ Code examples provided for complex functions  
✅ Industry standards referenced where applicable  
✅ Backward compatibility removals reflected  
✅ Clean, modern API throughout  

## Migration Notes

Users referring to the old API reference should note:
- `calculate_comfort_metrics()` now uses explicit parameter names (`max_longitudinal_accel` not `max_acceleration`)
- Return fields updated to match (no more `mean_acceleration` alias)
- New functions available for advanced use cases
- Multi-horizon evaluation now standard for ADE/FDE
- nuPlan mode available in `calculate_driving_score()`

## Next Steps

**Recommended**:
1. ✅ API reference updated
2. Update example scripts to reference new API docs
3. Consider adding quick reference table for planning metrics
4. Add version notes indicating breaking changes from previous API

**Optional**:
- Add diagrams for complex metrics (TTC projection, road edge distance)
- Create API comparison table (basic vs enhanced functions)
- Add benchmark performance notes

## Summary

The API reference now provides comprehensive, accurate documentation for all 20 planning metrics functions, reflecting:
- ✅ Clean API without backward compatibility
- ✅ Industry-standard implementations (nuPlan, Waymo, tuplan_garage)
- ✅ Complete parameter and return value documentation
- ✅ Practical code examples
- ✅ Clear semantic descriptions

**Status**: Production-ready documentation for v0.2.0 release.
