# Planning Metrics Test Coverage Report

**Date**: November 17, 2025  
**Test File**: `/tests/test_planning.py`  
**Total Functions**: 20  
**Functions Tested**: 15  
**Coverage**: 75%

## Test Coverage Summary

### ✅ Tested Functions (15/20)

| # | Function | Test Class | Test Count | Coverage |
|---|----------|------------|------------|----------|
| 1 | `calculate_l2_distance` | `TestL2Distance` | 6 tests | ✅ Comprehensive |
| 2 | `calculate_collision_rate` | `TestCollisionRate` | 6 tests | ✅ Comprehensive |
| 3 | `calculate_progress_score` | `TestProgressScore` | 4 tests | ✅ Good |
| 4 | `calculate_route_completion` | `TestRouteCompletion` | 5 tests | ✅ Comprehensive |
| 5 | `average_displacement_error_planning` | `TestAverageDisplacementErrorPlanning` | 4 tests | ✅ Good (includes multi-horizon) |
| 6 | `calculate_lateral_deviation` | `TestLateralDeviation` | 4 tests | ✅ Good |
| 7 | `calculate_heading_error` | `TestHeadingError` | 3 tests | ✅ Adequate |
| 8 | `calculate_velocity_error` | `TestVelocityError` | 3 tests | ✅ Adequate |
| 9 | `calculate_comfort_metrics` | `TestComfortMetrics` | 4 tests | ⚠️ Missing smoothing tests |
| 10 | `calculate_driving_score` | `TestDrivingScore` | 3 tests | ⚠️ Missing nuPlan mode tests |
| 11 | `calculate_planning_kl_divergence` | `TestPlanningKLDivergence` | 4 tests | ✅ Good |
| 12 | `calculate_time_to_collision` | `TestTimeToCollision` | 2 tests | ✅ Basic |
| 13 | `calculate_lane_invasion_rate` | `TestLaneInvasion` | 2 tests | ✅ Basic |
| 14 | `calculate_collision_severity` | `TestCollisionSeverity` | 2 tests | ✅ Basic |
| 15 | `check_kinematic_feasibility` | `TestKinematicFeasibility` | 2 tests | ✅ Basic |

**Total Tests**: ~52 tests across 15 functions

### ❌ Missing Tests (5/20)

| # | Function | Status | Priority |
|---|----------|--------|----------|
| 1 | `calculate_collision_with_fault_classification` | ❌ No tests | **HIGH** - Core new feature |
| 2 | `calculate_time_to_collision_enhanced` | ❌ No tests | **HIGH** - Core new feature |
| 3 | `calculate_distance_to_road_edge` | ❌ No tests | **HIGH** - Core new feature |
| 4 | `calculate_driving_direction_compliance` | ❌ No tests | **HIGH** - Core new feature |
| 5 | `calculate_interaction_metrics` | ❌ No tests | **HIGH** - Core new feature |

### ⚠️ Partially Tested (Missing New Features)

| Function | Missing Coverage | Priority |
|----------|------------------|----------|
| `calculate_comfort_metrics` | Smoothing parameters (`use_smoothing`, `smoothing_window`) | **MEDIUM** |
| `calculate_comfort_metrics` | Lateral metrics verification | **MEDIUM** |
| `calculate_driving_score` | nuPlan mode (`mode='nuplan'`) | **MEDIUM** |
| `average_displacement_error_planning` | ✅ Has multi-horizon test | OK |

## Detailed Gap Analysis

### High Priority Gaps

#### 1. `calculate_collision_with_fault_classification`
**Why Critical**: Core nuPlan feature, at-fault classification is key differentiator

**Needed Tests**:
- Test at-fault collision detection (active_front, stopped_track, active_lateral)
- Test not-at-fault collision detection (active_rear, passive)
- Test collision type counting
- Test with no collisions
- Test with multiple collision types

**Estimated Tests**: 5-6 tests

---

#### 2. `calculate_time_to_collision_enhanced`
**Why Critical**: Enhanced TTC with forward projection, nuPlan standard

**Needed Tests**:
- Test forward projection with moving obstacles
- Test projection horizon and dt parameters
- Test TTC violations counting
- Test with stopped vehicles (should be excluded)
- Test TTC profile generation

**Estimated Tests**: 5-6 tests

---

#### 3. `calculate_distance_to_road_edge`
**Why Critical**: Waymo Sim Agents standard, drivable area compliance

**Needed Tests**:
- Test with Shapely polygon (preferred method)
- Test with lane centerline fallback
- Test violation detection (positive distance = outside)
- Test signed distance calculation
- Test with trajectory entirely inside/outside

**Estimated Tests**: 5-6 tests

---

#### 4. `calculate_driving_direction_compliance`
**Why Critical**: nuPlan wrong-way detection

**Needed Tests**:
- Test correct driving direction (score = 1.0)
- Test wrong-way detection (score = 0.0)
- Test partial wrong-way (score = 0.5)
- Test angle threshold parameter
- Test heading error calculation

**Estimated Tests**: 4-5 tests

---

#### 5. `calculate_interaction_metrics`
**Why Critical**: Waymo Sim Agents multi-agent scenarios

**Needed Tests**:
- Test minimum distance calculation
- Test mean distance to nearest
- Test close interactions counting (<5m)
- Test closest object identification
- Test with static and dynamic obstacles

**Estimated Tests**: 5-6 tests

---

### Medium Priority Gaps

#### 6. Enhanced `calculate_comfort_metrics` Features
**Missing**: Smoothing and lateral metrics tests

**Needed Tests**:
- Test with `use_smoothing=True` vs `False`
- Test smoothing window and order parameters
- Test lateral acceleration calculation
- Test yaw rate and yaw acceleration
- Test comfort rate improvement with smoothing

**Estimated Tests**: 3-4 tests

---

#### 7. Enhanced `calculate_driving_score` Features
**Missing**: nuPlan mode tests

**Needed Tests**:
- Test `mode='nuplan'` vs `mode='default'`
- Test nuPlan-specific weights
- Test with at-fault collision penalty
- Test lateral comfort scoring in nuPlan mode

**Estimated Tests**: 3-4 tests

---

## Test File Statistics

**Current State**:
- Total test classes: 16
- Total test methods: ~52
- Lines of code: 573
- Coverage: 75% (15/20 functions)

**After Adding Missing Tests**:
- Expected test classes: 21 (+5)
- Expected test methods: ~82 (+30)
- Expected lines of code: ~900 (+327)
- Expected coverage: 100% (20/20 functions)

## Recommendations

### Immediate Actions (High Priority)

1. **Add 5 new test classes** for missing functions:
   ```python
   class TestCollisionWithFaultClassification:
       # 5-6 tests for at-fault collision detection
   
   class TestTimeToCollisionEnhanced:
       # 5-6 tests for enhanced TTC with projection
   
   class TestDistanceToRoadEdge:
       # 5-6 tests for drivable area compliance
   
   class TestDrivingDirectionCompliance:
       # 4-5 tests for wrong-way detection
   
   class TestInteractionMetrics:
       # 5-6 tests for multi-agent proximity
   ```

2. **Enhance existing test classes**:
   ```python
   class TestComfortMetrics:
       # Add test_with_smoothing()
       # Add test_lateral_metrics()
       # Add test_smoothing_window_validation()
   
   class TestDrivingScore:
       # Add test_nuplan_mode()
       # Add test_nuplan_weights()
       # Add test_nuplan_collision_penalty()
   ```

### Test Template Example

```python
class TestCollisionWithFaultClassification:
    """Test collision with fault classification."""
    
    def test_at_fault_rear_end(self):
        """Test at-fault rear-end collision (active_front)."""
        ego_traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ego_vels = np.array([10, 10, 10, 10])
        ego_heads = np.array([0, 0, 0, 0])
        # Slower vehicle ahead
        other = [np.array([[2, 0], [2.5, 0], [3, 0], [3.5, 0]])]
        
        result = calculate_collision_with_fault_classification(
            ego_traj, ego_vels, ego_heads, other
        )
        
        assert result['at_fault_collisions'] > 0
        assert 'active_front' in result['collision_types']
    
    def test_not_at_fault_rear_ended(self):
        """Test not-at-fault being rear-ended (active_rear)."""
        # Implementation...
    
    def test_no_collisions(self):
        """Test with safe trajectories."""
        # Implementation...
```

## Validation Checklist

After adding tests, verify:

- [ ] All 20 functions have test coverage
- [ ] New feature parameters tested (smoothing, nuPlan mode, multi-horizon)
- [ ] Edge cases covered (empty trajectories, single points, etc.)
- [ ] Industry standards verified (nuPlan thresholds, Waymo distances)
- [ ] All tests pass: `pytest tests/test_planning.py -v`
- [ ] Code coverage > 90%: `pytest tests/test_planning.py --cov=admetrics.planning`

## Summary

**Current Status**: 75% test coverage (15/20 functions)

**Gaps**:
- ❌ 5 new functions completely untested (HIGH priority)
- ⚠️ 2 functions missing new feature tests (MEDIUM priority)

**Effort Required**:
- Estimated ~30 new tests
- Estimated ~327 lines of test code
- Estimated 4-6 hours of work

**Impact**:
- Achieve 100% function coverage
- Verify all new nuPlan/Waymo features
- Production-ready test suite for v0.2.0 release
