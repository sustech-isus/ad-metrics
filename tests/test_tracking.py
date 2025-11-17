"""
Tests for multi-object tracking metrics.
"""

import pytest
import numpy as np
from admetrics.tracking.tracking import (
    calculate_mota,
    calculate_motp,
    calculate_clearmot_metrics,
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_id_f1,
    calculate_amota,
    calculate_motar,
    calculate_false_alarm_rate,
    calculate_track_metrics,
    calculate_moda,
    calculate_hota_components,
    calculate_trajectory_metrics,
    calculate_detection_metrics,
    calculate_smota,
    calculate_completeness,
    calculate_identity_metrics,
    calculate_tid_lgd,
    calculate_motal,
    calculate_clr_metrics,
    calculate_owta,
)


class TestMOTA:
    """Test MOTA calculation for single frame."""
    
    def test_perfect_tracking(self):
        """Test MOTA with perfect detections."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['mota'] == 1.0
        assert result['tp'] == 1
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_with_false_positives(self):
        """Test MOTA with false positives."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # MOTA = 1 - (FN + FP + IDSW) / GT = 1 - (0 + 1 + 0) / 1 = 0.0
        assert result['mota'] == 0.0
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 0
    
    def test_with_false_negatives(self):
        """Test MOTA with missed detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # MOTA = 1 - (FN + FP + IDSW) / GT = 1 - (1 + 0 + 0) / 1 = 0.0
        assert result['mota'] == 0.0
        assert result['tp'] == 0
        assert result['fp'] == 0
        assert result['fn'] == 1


class TestMOTP:
    """Test MOTP calculation."""
    
    def test_perfect_localization(self):
        """Test MOTP with perfect localization."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_motp(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['motp'] == 0.0  # Perfect match = 0 distance
        assert result['num_tp'] == 1
    
    def test_with_distance(self):
        """Test MOTP with some distance error."""
        predictions = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_motp(predictions, ground_truth, iou_threshold=0.5)
        
        # Distance should be 0.5 (x offset)
        assert 0.4 < result['motp'] < 0.6
        assert result['num_tp'] == 1


class TestClearMOT:
    """Test CLEAR MOT metrics (combined MOTA and MOTP)."""
    
    def test_clearmot_perfect(self):
        """Test CLEAR MOT with perfect tracking."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        # Should have both MOTA and MOTP
        assert 'mota' in result
        assert 'motp' in result
        assert 'tp' in result
        assert 'fp' in result
        assert 'fn' in result
        assert 'num_tp' in result
        
        # Perfect tracking
        assert result['mota'] == 1.0
        assert result['motp'] == 0.0  # Perfect localization = 0 distance
        assert result['tp'] == 1
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_clearmot_with_errors(self):
        """Test CLEAR MOT with tracking errors."""
        predictions = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}  # FP
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}  # FN
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        # Should have reduced MOTA due to FP and FN
        assert result['mota'] < 1.0
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 1
        
        # MOTP should reflect localization error of TP
        assert result['motp'] > 0
        assert result['num_tp'] == 1
    
    def test_clearmot_no_detections(self):
        """Test CLEAR MOT with no detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth)
        
        assert result['mota'] == 0.0
        assert result['motp'] == 0.0
        assert result['tp'] == 0
        assert result['fn'] == 1


class TestMultiFrameMOTA:
    """Test multi-frame MOTA calculation."""
    
    @pytest.fixture
    def simple_sequence(self):
        """Simple 2-frame sequence with one object."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        return predictions, ground_truth
    
    def test_perfect_tracking_sequence(self, simple_sequence):
        """Test perfect tracking across frames."""
        predictions, ground_truth = simple_sequence
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['mota'] == 1.0
        assert result['num_matches'] == 2
        assert result['num_false_positives'] == 0
        assert result['num_misses'] == 0
        assert result['num_switches'] == 0
        assert result['mostly_tracked'] == 1
    
    def test_id_switch(self):
        """Test ID switch detection."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}]  # ID changed!
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['num_switches'] == 1
        assert result['mota'] < 1.0  # MOTA penalizes ID switches
    
    def test_fragmentation(self):
        """Test fragmentation detection."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [],  # Track lost
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]  # Track recovered
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['num_fragmentations'] == 1
        assert result['num_misses'] == 1  # Frame 1 was missed
    
    def test_mostly_tracked_ratio(self):
        """Test mostly tracked trajectory classification."""
        # Track detected in 4 out of 5 frames (80%)
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            2: [],  # Missed
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # 4/5 = 80% >= 80% threshold
        assert result['mostly_tracked'] == 1
        assert result['partially_tracked'] == 0
        assert result['mostly_lost'] == 0


class TestHOTA:
    """Test HOTA metric."""
    
    def test_perfect_hota(self):
        """Test HOTA with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_hota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['hota'] > 0
        assert result['det_a'] == 1.0  # Perfect detection
        assert result['tp'] == 2
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_hota_with_errors(self):
        """Test HOTA with detection errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # FP
            ],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [11, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # FN
            ]
        }
        
        result = calculate_hota(predictions, ground_truth, iou_threshold=0.5)
        
        assert 0 <= result['hota'] <= 1
        assert result['det_a'] < 1.0  # Has errors
        assert result['tp'] == 2
        assert result['fp'] == 1
        assert result['fn'] == 1


class TestIDMetrics:
    """Test ID-based metrics (IDF1)."""
    
    def test_perfect_id_f1(self):
        """Test IDF1 with perfect ID consistency."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_id_f1(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['idf1'] == 1.0
        assert result['idp'] == 1.0
        assert result['idr'] == 1.0
    
    def test_id_switch_penalty(self):
        """Test IDF1 penalty for ID switches."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}]  # ID switched
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_id_f1(predictions, ground_truth, iou_threshold=0.5)
        
        # ID switch should reduce IDF1
        assert result['idf1'] < 1.0
        assert result['idfp'] > 0
        assert result['idfn'] > 0


class TestAMOTA:
    """Test AMOTA (Average MOTA) calculation."""
    
    def test_amota_basic(self):
        """Test AMOTA calculation with basic scenario."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'score': 0.9, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'score': 0.8, 'class': 'car'}
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'score': 0.85, 'class': 'car'},
                {'box': [6, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'score': 0.75, 'class': 'car'}
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [6, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}
            ]
        }
        
        result = calculate_amota(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'amota' in result
        assert 'amotp' in result
        assert 'motas' in result
        assert 'motps' in result
        assert isinstance(result['motas'], list)
        assert len(result['motas']) == len(result['recall_thresholds'])
        assert -1.0 <= result['amota'] <= 1.0
    
    def test_amota_custom_thresholds(self):
        """Test AMOTA with custom recall thresholds."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'score': 0.9, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        custom_thresholds = [0.3, 0.5, 0.7]
        result = calculate_amota(predictions, ground_truth, 
                                recall_thresholds=custom_thresholds,
                                iou_threshold=0.5)
        
        assert len(result['motas']) == len(custom_thresholds)
        assert result['recall_thresholds'] == custom_thresholds


class TestMOTAR:
    """Test MOTAR (MOTA at Recall) calculation."""
    
    def test_motar_basic(self):
        """Test MOTAR at specific recall threshold."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'score': 0.9, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'score': 0.5, 'class': 'car'}
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}
            ]
        }
        
        result = calculate_motar(predictions, ground_truth, 
                                recall_threshold=0.5, iou_threshold=0.5)
        
        assert 'motar' in result
        assert 'target_recall' in result
        assert 'actual_recall' in result
        assert result['target_recall'] == 0.5
        assert -1.0 <= result['motar'] <= 1.0


class TestFalseAlarmRate:
    """Test False Alarm Rate metrics."""
    
    def test_faf_no_false_alarms(self):
        """Test FAF with no false alarms."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}]
        }
        
        result = calculate_false_alarm_rate(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['faf'] == 0.0
        assert result['total_false_positives'] == 0
        assert result['num_frames'] == 2
        assert result['far'] == 0.0
    
    def test_faf_with_false_alarms(self):
        """Test FAF with false alarms."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}  # False positive
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
                {'box': [11, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}  # False positive
            ]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}]
        }
        
        result = calculate_false_alarm_rate(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['faf'] == 1.0  # 2 FP / 2 frames
        assert result['total_false_positives'] == 2
        assert result['num_frames'] == 2
        assert result['far'] == 0.5  # 2 FP / 4 total predictions


class TestTrackMetrics:
    """Test track-level metrics."""
    
    def test_track_recall_perfect(self):
        """Test track recall with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_track_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['track_recall'] == 1.0
        assert result['track_precision'] == 1.0
        assert result['num_gt_tracks'] == 1
        assert result['num_pred_tracks'] == 1
        assert result['num_matched_gt_tracks'] == 1
        assert result['num_matched_pred_tracks'] == 1
    
    def test_track_recall_missed_track(self):
        """Test track recall with missed tracks."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # Missed
            ]
        }
        
        result = calculate_track_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['track_recall'] == 0.5  # 1 matched out of 2 GT tracks
        assert result['num_gt_tracks'] == 2
        assert result['num_matched_gt_tracks'] == 1


class TestMODA:
    """Test MODA (Multiple Object Detection Accuracy)."""
    
    def test_moda_perfect(self):
        """Test MODA with perfect detections."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_moda(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['moda'] == 1.0
        assert result['num_false_positives'] == 0
        assert result['num_misses'] == 0
    
    def test_moda_with_errors(self):
        """Test MODA with detection errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # FP
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # FN
            ]
        }
        
        result = calculate_moda(predictions, ground_truth, iou_threshold=0.5)
        
        # MODA = 1 - (FN + FP) / GT = 1 - (1 + 1) / 2 = 0.0
        assert result['moda'] == 0.0
        assert result['num_false_positives'] == 1
        assert result['num_misses'] == 1
        assert result['total_gt'] == 2


class TestHOTAComponents:
    """Test detailed HOTA components."""
    
    def test_hota_components_perfect(self):
        """Test HOTA components with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_hota_components(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'hota' in result
        assert 'det_a' in result
        assert 'det_re' in result
        assert 'det_pr' in result
        assert 'ass_a' in result
        assert 'ass_re' in result
        assert 'ass_pr' in result
        assert 'loc_a' in result
        
        # Perfect tracking should have high scores
        assert result['det_re'] == 1.0
        assert result['det_pr'] == 1.0
        assert result['det_a'] == 1.0
        assert result['loc_a'] == 1.0  # Perfect IoU
    
    def test_hota_components_with_errors(self):
        """Test HOTA components with detection errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # FP
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # FN
            ]
        }
        
        result = calculate_hota_components(predictions, ground_truth, iou_threshold=0.5)
        
        # With 1 TP, 1 FP, 1 FN
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 1
        assert result['det_re'] == 0.5  # 1 / (1 + 1)
        assert result['det_pr'] == 0.5  # 1 / (1 + 1)
        assert 0.0 <= result['hota'] <= 1.0


class TestTrajectoryMetrics:
    """Test trajectory-level metrics."""
    
    def test_trajectory_classification(self):
        """Test MT/ML/PT classification."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},  # Track 1: 100% coverage
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # Track 2: 50% coverage
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}
                # Track 2 missing (50% coverage)
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},  # GT 100: fully tracked
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},  # GT 101: partial
                {'box': [20, 0, 0, 4, 2, 1.5, 0], 'track_id': 102, 'class': 'car'}   # GT 102: mostly lost
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [11, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
                {'box': [21, 0, 0, 4, 2, 1.5, 0], 'track_id': 102, 'class': 'car'}
            ]
        }
        
        result = calculate_trajectory_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['total_tracks'] == 3
        assert result['mt_count'] == 1  # Track 100: 100% coverage
        assert result['pt_count'] == 1  # Track 101: 50% coverage
        assert result['ml_count'] == 1  # Track 102: 0% coverage
        assert result['mt_ratio'] == 1/3
        assert result['pt_ratio'] == 1/3
        assert result['ml_ratio'] == 1/3


class TestDetectionMetrics:
    """Test frame-level detection metrics."""
    
    def test_perfect_detection(self):
        """Test detection metrics with perfect detections."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}]
        }
        
        result = calculate_detection_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0
        assert result['tp'] == 2
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_detection_with_errors(self):
        """Test detection metrics with errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}  # FP
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}   # FN
            ]
        }
        
        result = calculate_detection_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 1
        assert result['precision'] == 0.5  # 1 / (1 + 1)
        assert result['recall'] == 0.5     # 1 / (1 + 1)
        assert result['f1'] == 0.5


class TestSMOTA:
    """Test soft MOTA."""
    
    def test_smota_perfect(self):
        """Test sMOTA with perfect matches."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_smota(predictions, ground_truth, iou_threshold=0.5, use_soft_matching=True)
        
        assert result['smota'] == 1.0  # Perfect IoU = 1.0
        assert result['soft_tp_error'] == 0.0
        assert result['num_matches'] == 2
        assert result['num_false_positives'] == 0
        assert result['num_switches'] == 0
    
    def test_smota_with_imperfect_iou(self):
        """Test sMOTA with imperfect IoU."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]  # Slightly offset
        }
        
        result = calculate_smota(predictions, ground_truth, iou_threshold=0.5, use_soft_matching=True)
        
        # With soft matching, sMOTA should be < 1.0 due to imperfect IoU
        assert result['smota'] < 1.0
        assert result['soft_tp_error'] > 0.0
        assert result['num_matches'] == 1


class TestCompleteness:
    """Test tracking completeness metrics."""
    
    def test_full_coverage(self):
        """Test completeness with full coverage."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_completeness(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['gt_covered_ratio'] == 1.0
        assert result['avg_gt_coverage'] == 1.0
        assert result['frame_coverage'] == 1.0
        assert result['num_gt_objects'] == 1
        assert result['num_detected_objects'] == 1
    
    def test_partial_coverage(self):
        """Test completeness with partial coverage."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: []  # No detections in frame 1
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # Never detected
            ],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}
            ]
        }
        
        result = calculate_completeness(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['gt_covered_ratio'] == 0.5  # Only 1 of 2 GT objects detected
        assert result['frame_coverage'] == 0.5    # Only 1 of 2 frames has detections
        assert result['num_gt_objects'] == 2
        assert result['num_detected_objects'] == 1


class TestIdentityMetrics:
    """Test detailed identity metrics."""
    
    def test_identity_perfect(self):
        """Test identity metrics with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_identity_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['id_switches'] == 0
        assert result['id_switch_rate'] == 0.0
        assert result['avg_track_purity'] == 1.0
        assert result['avg_track_completeness'] == 1.0
        assert result['num_fragmentations'] == 0
        assert result['fragmentation_rate'] == 0.0
    
    def test_identity_with_switches(self):
        """Test identity metrics with ID switches."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}]  # ID switched
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_identity_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['id_switches'] == 1
        assert result['id_switch_rate'] == 1.0  # 1 switch / 1 GT track
        assert result['avg_track_completeness'] == 0.5  # Each pred track has 50% of GT


class TestTIDLGD:
    """Test Track Initialization Duration and Longest Gap Duration."""
    
    def test_immediate_detection(self):
        """Test TID/LGD when tracks are detected immediately."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_tid_lgd(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['tid'] == 0.0  # Detected in first frame
        assert result['lgd'] == 0.0  # No gaps
        assert result['num_tracks'] == 1
        assert result['num_detected_tracks'] == 1
    
    def test_delayed_detection(self):
        """Test TID when detection is delayed."""
        predictions = {
            0: [],  # Not detected in first frame
            1: [],  # Not detected in second frame
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],  # First detection
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_tid_lgd(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['tid'] == 2.0  # First detected at frame 2 (0-indexed)
        assert result['num_detected_tracks'] == 1
    
    def test_with_gaps(self):
        """Test LGD with tracking gaps."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [],  # Gap
            2: [],  # Gap
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_tid_lgd(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['lgd'] == 2.0  # Gap of 2 frames (frames 1-2)
        assert result['tid'] == 0.0  # Detected immediately


class TestMOTAL:
    """Test MOTAL (MOTA with Logarithmic ID Switches)."""
    
    def test_motal_no_switches(self):
        """Test MOTAL with no ID switches."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_motal(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['motal'] == result['mota']  # Same when no ID switches
        assert result['id_switches'] == 0
        assert result['log_id_switches'] == 0.0
    
    def test_motal_with_switches(self):
        """Test MOTAL with ID switches (should be higher than MOTA)."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}],  # ID switch
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 3, 'class': 'car'}],  # ID switch
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_motal(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['id_switches'] == 2
        assert result['log_id_switches'] > 0
        assert result['motal'] > result['mota']  # MOTAL should be higher (less penalty)


class TestCLRMetrics:
    """Test CLEAR MOT metrics (CLR_Re, CLR_Pr, CLR_F1)."""
    
    def test_clr_perfect(self):
        """Test CLR metrics with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_clr_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['clr_re'] == 1.0  # Perfect recall
        assert result['clr_pr'] == 1.0  # Perfect precision
        assert result['clr_f1'] == 1.0  # Perfect F1
        assert result['tp'] == 2
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_clr_with_errors(self):
        """Test CLR metrics with errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # FP
            ]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [5, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # FN
            ]
        }
        
        result = calculate_clr_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 1
        assert result['clr_re'] == 0.5  # 1 / (1 + 1)
        assert result['clr_pr'] == 0.5  # 1 / (1 + 1)
        assert result['clr_f1'] == 0.5  # 1 / (1 + 0.5 + 0.5)


class TestOWTA:
    """Test Open World Tracking Accuracy."""
    
    def test_owta_perfect(self):
        """Test OWTA with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_owta(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['owta'] == 1.0  # sqrt(1.0 * 1.0)
        assert result['det_re'] == 1.0
        assert result['ass_a'] == 1.0
    
    def test_owta_with_errors(self):
        """Test OWTA with detection and association errors."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # Missed
            ]
        }
        
        result = calculate_owta(predictions, ground_truth, iou_threshold=0.5)
        
        assert 0.0 <= result['owta'] <= 1.0
        assert result['owta'] == np.sqrt(result['det_re'] * result['ass_a'])



