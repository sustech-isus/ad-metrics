"""
Tests for utility functions.
"""

import pytest
import numpy as np
from admetrics.utils.transforms import (
    transform_box,
    rotate_box,
    translate_box,
    convert_box_format,
    center_to_corners,
    corners_to_center,
    normalize_angle,
    angle_difference,
    lidar_to_camera,
    camera_to_lidar,
    boxes_to_bev
)
from admetrics.utils.matching import (
    match_detections,
    greedy_matching,
    hungarian_matching,
    match_by_center_distance,
    filter_matches_by_class
)
from admetrics.utils.nms import (
    nms_3d,
    nms_bev,
    nms_per_class,
    soft_nms_3d,
    distance_based_nms
)


class TestTransforms:
    """Test transformation utilities."""
    
    def test_translate_box(self):
        """Test box translation."""
        box = np.array([0, 0, 0, 4, 2, 1.5, 0])
        translation = np.array([1, 2, 3])
        
        translated = translate_box(box, translation)
        
        assert np.allclose(translated[:3], [1, 2, 3])
        assert np.allclose(translated[3:], box[3:])
    
    def test_rotate_box(self):
        """Test box rotation."""
        box = np.array([1, 0, 0, 4, 2, 1.5, 0])
        rotation = np.pi / 2  # 90 degrees
        
        rotated = rotate_box(box, rotation)
        
        # After 90° rotation around origin, (1, 0) should become ~(0, 1)
        assert np.isclose(rotated[0], 0, atol=1e-10)
        assert np.isclose(rotated[1], 1, atol=1e-10)
        assert np.isclose(rotated[6], np.pi / 2)
    
    def test_center_to_corners(self):
        """Test converting center format to corners."""
        box = np.array([0, 0, 0, 4, 2, 2, 0])
        
        corners = center_to_corners(box)
        
        assert corners.shape == (8, 3)
        
        # Check that corners are at expected distances
        center = np.array([0, 0, 0])
        for corner in corners:
            dist = np.linalg.norm(corner[:2] - center[:2])
            # Distance should be sqrt((l/2)^2 + (w/2)^2) for BEV
            expected_dist = np.sqrt((2/2)**2 + (4/2)**2)
            assert np.isclose(dist, expected_dist, atol=0.1)
    
    def test_normalize_angle(self):
        """Test angle normalization."""
        angle1 = 3 * np.pi
        normalized = normalize_angle(angle1)
        
        assert -np.pi <= normalized <= np.pi
        # π and -π are equivalent, so check if it's close to either
        assert np.isclose(np.abs(normalized), np.pi, atol=1e-10)
    
    def test_convert_box_format(self):
        """Test box format conversion."""
        box = np.array([0, 0, 0, 4, 2, 1.5, 0])
        
        # Convert xyzwhlr to xyzhwlr and back
        converted = convert_box_format(box, 'xyzwhlr', 'xyzhwlr')
        back = convert_box_format(converted, 'xyzhwlr', 'xyzwhlr')
        
        assert np.allclose(box, back)
    
    def test_transform_box(self):
        """Test combined transformations."""
        box = np.array([0, 0, 0, 4, 2, 1.5, 0])
        translation = np.array([1, 2, 3])
        rotation = np.pi / 4
        scale = 2.0
        
        transformed = transform_box(box, translation=translation, rotation=rotation, scale=scale)
        
        # Check that transformations were applied
        assert not np.allclose(transformed, box)
        assert transformed.shape == box.shape
    
    def test_corners_to_center(self):
        """Test converting corners back to center format."""
        box = np.array([0, 0, 0, 4, 2, 2, 0])
        
        corners = center_to_corners(box)
        back = corners_to_center(corners)
        
        # Should reconstruct the original box (approximately)
        assert np.allclose(back[:3], box[:3], atol=0.1)
        assert np.allclose(back[3:6], box[3:6], atol=0.1)
    
    def test_angle_difference(self):
        """Test angle difference calculation."""
        angle1 = 0.1
        angle2 = -0.1
        
        diff = angle_difference(angle1, angle2)
        
        assert np.isclose(diff, 0.2, atol=1e-6)
        
        # Test wrap around - use absolute value since sign may vary
        diff2 = angle_difference(np.pi - 0.1, -np.pi + 0.1)
        assert np.isclose(abs(diff2), 0.2, atol=1e-6)
    
    def test_lidar_to_camera(self):
        """Test LiDAR to camera transformation."""
        box = np.array([1, 0, 0, 4, 2, 1.5, 0.1])
        
        # 4x4 calibration matrix
        calibration = np.eye(4)
        calibration[:3, 3] = [1, 2, 3]  # Translation
        
        transformed = lidar_to_camera(box, calibration)
        
        assert transformed.shape == box.shape
        # Position should be transformed
        assert not np.allclose(transformed[:3], box[:3])
    
    def test_camera_to_lidar(self):
        """Test camera to LiDAR transformation."""
        box = np.array([1, 0, 0, 4, 2, 1.5, 0.1])
        
        # 4x4 calibration matrix
        calibration = np.eye(4)
        calibration[:3, 3] = [1, 2, 3]
        
        # Transform to camera and back
        cam_box = lidar_to_camera(box, calibration)
        back = camera_to_lidar(cam_box, calibration)
        
        assert np.allclose(back, box, atol=1e-6)
    
    def test_boxes_to_bev(self):
        """Test BEV projection of boxes."""
        boxes = np.array([
            [0, 0, 0, 4, 2, 1.5, 0],
            [5, 5, 0, 3, 2, 1, np.pi/4]
        ])
        
        bev_grid = boxes_to_bev(boxes)
        
        # Should return a 2D grid
        assert bev_grid.ndim == 2
        assert bev_grid.shape[0] > 0
        assert bev_grid.shape[1] > 0


class TestMatching:
    """Test detection matching algorithms."""
    
    @pytest.fixture
    def sample_detections(self):
        """Sample detections for testing."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [5.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        return predictions, ground_truth
    
    def test_greedy_matching(self, sample_detections):
        """Test greedy matching algorithm."""
        predictions, ground_truth = sample_detections
        
        matches, unmatched_preds, unmatched_gts = greedy_matching(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_gts) == 0
    
    def test_hungarian_matching(self, sample_detections):
        """Test Hungarian matching algorithm."""
        predictions, ground_truth = sample_detections
        
        matches, unmatched_preds, unmatched_gts = hungarian_matching(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_gts) == 0
    
    def test_match_detections_wrapper(self, sample_detections):
        """Test match_detections wrapper function."""
        predictions, ground_truth = sample_detections
        
        # Test greedy
        matches_greedy, _, _ = match_detections(
            predictions, ground_truth, method="greedy"
        )
        
        # Test hungarian
        matches_hungarian, _, _ = match_detections(
            predictions, ground_truth, method="hungarian"
        )
        
        # Both should find 2 matches
        assert len(matches_greedy) == 2
        assert len(matches_hungarian) == 2
    
    def test_match_by_center_distance(self):
        """Test matching by center distance."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [10.5, 10, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        
        matches, unmatched_preds, unmatched_gts = match_by_center_distance(
            predictions, ground_truth, distance_threshold=1.0
        )
        
        assert len(matches) == 2
    
    def test_filter_matches_by_class(self):
        """Test filtering matches by class."""
        matches = [(0, 0), (1, 1), (2, 2)]
        predictions = [
            {'class': 'car'},
            {'class': 'car'},
            {'class': 'pedestrian'}
        ]
        ground_truth = [
            {'class': 'car'},
            {'class': 'pedestrian'},  # Mismatch
            {'class': 'pedestrian'}
        ]
        
        filtered = filter_matches_by_class(matches, predictions, ground_truth)
        
        # Only first and third match have same class
        assert len(filtered) == 2
        assert (0, 0) in filtered
        assert (2, 2) in filtered
    
    def test_matching_with_invalid_method(self, sample_detections):
        """Test that invalid method raises error."""
        predictions, ground_truth = sample_detections
        
        with pytest.raises(ValueError):
            match_detections(predictions, ground_truth, method="invalid")


class TestNMS:
    """Test Non-Maximum Suppression."""
    
    def test_nms_3d_basic(self):
        """Test basic 3D NMS."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8},  # High overlap
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.7},  # No overlap
        ]
        
        keep_indices = nms_3d(boxes, iou_threshold=0.5)
        
        # Should keep first and third boxes (highest scores with no overlap)
        assert 0 in keep_indices
        assert 2 in keep_indices
        assert 1 not in keep_indices
    
    def test_nms_bev(self):
        """Test BEV NMS."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [0.5, 0, 5, 4, 2, 1.5, 0], 'score': 0.8},  # High overlap in BEV
        ]
        
        keep_indices = nms_bev(boxes, iou_threshold=0.5)
        
        # Should keep only the highest score (boxes have high BEV overlap)
        assert len(keep_indices) == 1
        assert 0 in keep_indices
    
    def test_nms_per_class(self):
        """Test NMS applied per class."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
            {'box': [0, 0, 0, 2, 1, 1, 0], 'score': 0.7, 'class': 'pedestrian'},
        ]
        
        keep_indices = nms_per_class(boxes, iou_threshold=0.5)
        
        # Should keep highest car and the pedestrian
        assert 0 in keep_indices  # Highest car
        assert 2 in keep_indices  # Pedestrian (different class)
    
    def test_nms_with_score_threshold(self):
        """Test NMS with score threshold."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.3},  # Below threshold
        ]
        
        keep_indices = nms_3d(boxes, score_threshold=0.5)
        
        # Should only keep first box
        assert len(keep_indices) == 1
        assert 0 in keep_indices
    
    def test_soft_nms_3d(self):
        """Test Soft-NMS implementation."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8},  # High overlap
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.7},  # No overlap
        ]
        
        keep_indices, updated_scores = soft_nms_3d(boxes, iou_threshold=0.5)
        
        # soft_nms_3d returns (keep_indices, updated_scores)
        assert len(keep_indices) >= 2  # At least some boxes kept
        assert isinstance(keep_indices, (list, np.ndarray))
    
    def test_distance_based_nms(self):
        """Test distance-based NMS."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [1, 0, 0, 4, 2, 1.5, 0], 'score': 0.8},  # Close distance
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.7},  # Far away
        ]
        
        keep_indices = distance_based_nms(boxes, distance_threshold=2.0)
        
        # Should keep boxes 0 and 2 (boxes 0 and 1 are too close)
        assert 0 in keep_indices
        assert 2 in keep_indices
        assert 1 not in keep_indices
    
    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        boxes = []
        
        keep_indices = nms_3d(boxes)
        
        assert len(keep_indices) == 0
    
    def test_nms_single_box(self):
        """Test NMS with single box."""
        boxes = [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9}]
        
        keep_indices = nms_3d(boxes)
        
        assert len(keep_indices) == 1
        assert 0 in keep_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
