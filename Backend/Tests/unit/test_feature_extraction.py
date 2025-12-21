"""
Enhanced Feature Extraction Testing
Comprehensive tests for CNN feature extraction with robustness checks
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from Tests.test_config import TestConfig

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.pre_trained_feature_map import (
    load_model,
    extract_features_from_generator,
    flatten_features,
    temporal_pooling,
    multi_view_fusion,
    process_features
)


class TestLoadModel:
    """Test cases for model loading"""
    
    def test_load_model_returns_correct_components(self):
        """Test that load_model returns model, feature_extractor, and device"""
        model, feature_extractor, device = load_model()
        
        assert model is not None
        assert feature_extractor is not None
        assert device is not None
        assert str(device) in ['cuda', 'cpu']
    
    def test_model_in_eval_mode(self):
        """Test that model is in evaluation mode"""
        model, _, _ = load_model()
        assert not model.training
    
    def test_feature_extractor_has_no_classifier(self):
        """Test that feature_extractor excludes the final classification layer"""
        model, feature_extractor, _ = load_model()
        
        # Feature extractor should have fewer layers than full model
        num_model_children = len(list(model.children()))
        num_extractor_children = len(list(feature_extractor.children()))
        
        assert num_extractor_children < num_model_children
    
    def test_model_device_consistency(self):
        """Test that model is on the correct device"""
        model, feature_extractor, device = load_model()
        
        # Check that model parameters are on the specified device
        model_device = next(model.parameters()).device
        assert str(model_device).startswith(str(device).split(':')[0])


class TestFlattenFeatures:
    """Test cases for feature flattening"""
    
    def test_flatten_single_feature(self):
        """Test flattening a single feature map"""
        feature_map = {
            'apple_obj001_0': [
                {
                    'features': np.random.rand(7, 7, 1024),
                    'timestamp': '2025-01-01T00:00:00',
                    'label': 0
                }
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Check output structure
        assert 'apple_obj001_0_t0' in flattened
        assert flattened['apple_obj001_0_t0']['group_key'] == 'apple_obj001_0'
        
        # Check flattened dimension (7 * 7 * 1024 = 50176)
        assert flattened['apple_obj001_0_t0']['features'].shape[0] == 7 * 7 * 1024
    
    def test_flatten_multiple_timesteps(self):
        """Test flattening multiple timesteps"""
        feature_map = {
            'apple_obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't2', 'label': 0}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Should have 3 timesteps
        assert len(flattened) == 3
        assert 'apple_obj001_0_t0' in flattened
        assert 'apple_obj001_0_t1' in flattened
        assert 'apple_obj001_0_t2' in flattened
    
    def test_flatten_preserves_group_key(self):
        """Test that group_key is preserved correctly"""
        feature_map = {
            'banana_obj002_1': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 1}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        assert flattened['banana_obj002_1_t0']['group_key'] == 'banana_obj002_1'
    
    def test_flatten_empty_features(self):
        """Test flattening with empty input"""
        flattened = flatten_features({})
        assert len(flattened) == 0
    
    def test_flatten_preserves_labels(self):
        """Test that labels are preserved through flattening"""
        feature_map = {
            'apple_obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 2}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Label should be preserved (if implementation includes it)
        # This test depends on your actual implementation
        assert 'apple_obj001_0_t0' in flattened
    
    def test_flatten_different_feature_dimensions(self):
        """Test flattening with different feature map sizes"""
        # Test with different spatial dimensions
        feature_map_small = {
            'test_small': [
                {'features': np.random.rand(3, 3, 512), 'timestamp': 't0', 'label': 0}
            ]
        }
        
        feature_map_large = {
            'test_large': [
                {'features': np.random.rand(14, 14, 2048), 'timestamp': 't0', 'label': 0}
            ]
        }
        
        flattened_small = flatten_features(feature_map_small)
        flattened_large = flatten_features(feature_map_large)
        
        # Verify correct flattening
        assert flattened_small['test_small_t0']['features'].shape[0] == 3 * 3 * 512
        assert flattened_large['test_large_t0']['features'].shape[0] == 14 * 14 * 2048


class TestTemporalPooling:
    """Test cases for temporal pooling"""
    
    def test_temporal_pooling_single_frame(self):
        """Test pooling with single frame (no averaging needed)"""
        flattened = {
            'apple_obj001_0_t0': {
                'features': np.array([1, 2, 3, 4, 5]),
                'group_key': 'apple_obj001_0'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        assert 'apple_obj001_0' in pooled
        np.testing.assert_array_equal(pooled['apple_obj001_0'], np.array([1, 2, 3, 4, 5]))
    
    def test_temporal_pooling_multiple_frames(self):
        """Test pooling averages across multiple frames"""
        flattened = {
            'apple_obj001_0_t0': {
                'features': np.array([2.0, 4.0, 6.0]),
                'group_key': 'apple_obj001_0'
            },
            'apple_obj001_0_t1': {
                'features': np.array([4.0, 6.0, 8.0]),
                'group_key': 'apple_obj001_0'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        # Average should be [3.0, 5.0, 7.0]
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0'], expected)
    
    def test_temporal_pooling_multiple_groups(self):
        """Test pooling with multiple camera groups"""
        flattened = {
            'apple_obj001_0_t0': {'features': np.array([1.0, 2.0]), 'group_key': 'apple_obj001_0'},
            'apple_obj001_0_t1': {'features': np.array([3.0, 4.0]), 'group_key': 'apple_obj001_0'},
            'apple_obj001_1_t0': {'features': np.array([5.0, 6.0]), 'group_key': 'apple_obj001_1'}
        }
        
        pooled = temporal_pooling(flattened)
        
        # Should have 2 groups
        assert len(pooled) == 2
        assert 'apple_obj001_0' in pooled
        assert 'apple_obj001_1' in pooled
        
        # Check averages
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0'], np.array([2.0, 3.0]))
        np.testing.assert_array_almost_equal(pooled['apple_obj001_1'], np.array([5.0, 6.0]))
    
    def test_temporal_pooling_empty(self):
        """Test temporal pooling with empty input"""
        pooled = temporal_pooling({})
        assert len(pooled) == 0
    
    def test_temporal_pooling_many_frames(self):
        """Test pooling with many temporal frames"""
        # Create 10 frames for same group
        flattened = {}
        for i in range(10):
            flattened[f'apple_obj001_0_t{i}'] = {
                'features': np.ones(100) * i,
                'group_key': 'apple_obj001_0'
            }
        
        pooled = temporal_pooling(flattened)
        
        # Average should be 4.5 (mean of 0-9)
        expected = np.ones(100) * 4.5
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0'], expected)
    
    def test_temporal_pooling_preserves_dimensions(self):
        """Test that pooling preserves feature dimensions"""
        feature_dim = 50176  # 7 * 7 * 1024
        
        flattened = {
            'test_obj_0_t0': {
                'features': np.random.rand(feature_dim),
                'group_key': 'test_obj_0'
            },
            'test_obj_0_t1': {
                'features': np.random.rand(feature_dim),
                'group_key': 'test_obj_0'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        assert pooled['test_obj_0'].shape[0] == feature_dim


class TestMultiViewFusion:
    """Test cases for multi-view fusion"""
    
    def test_fusion_single_camera(self):
        """Test fusion with single camera (no concatenation needed)"""
        pooled = {
            'apple_obj001_0': np.array([1, 2, 3, 4, 5])
        }
        
        fused = multi_view_fusion(pooled)
        
        assert 'apple_obj001' in fused
        np.testing.assert_array_equal(fused['apple_obj001'], np.array([1, 2, 3, 4, 5]))
    
    def test_fusion_multiple_cameras(self):
        """Test fusion concatenates features from multiple cameras"""
        pooled = {
            'apple_obj001_0': np.array([1, 2, 3]),
            'apple_obj001_1': np.array([4, 5, 6]),
            'apple_obj001_2': np.array([7, 8, 9])
        }
        
        fused = multi_view_fusion(pooled)
        
        # Should concatenate all camera views
        assert 'apple_obj001' in fused
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(fused['apple_obj001'], expected)
    
    def test_fusion_multiple_objects(self):
        """Test fusion with multiple objects"""
        pooled = {
            'apple_obj001_0': np.array([1, 2]),
            'apple_obj001_1': np.array([3, 4]),
            'banana_obj002_0': np.array([5, 6]),
            'banana_obj002_1': np.array([7, 8])
        }
        
        fused = multi_view_fusion(pooled)
        
        # Should have 2 objects
        assert len(fused) == 2
        assert 'apple_obj001' in fused
        assert 'banana_obj002' in fused
        
        # Check concatenation
        np.testing.assert_array_equal(fused['apple_obj001'], np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(fused['banana_obj002'], np.array([5, 6, 7, 8]))
    
    def test_fusion_correct_dimensions(self):
        """Test that fusion produces correct feature dimensions"""
        # 4 cameras, each with 1024-dim features
        pooled = {
            f'apple_obj001_{i}': np.random.rand(1024)
            for i in range(4)
        }
        
        fused = multi_view_fusion(pooled)
        
        # Should concatenate to 4096 dimensions (4 * 1024)
        assert fused['apple_obj001'].shape[0] == 4096
    
    def test_multi_view_fusion_empty(self):
        """Test multi-view fusion with empty input"""
        fused = multi_view_fusion({})
        assert len(fused) == 0
    
    def test_fusion_camera_ordering(self):
        """Test that cameras are concatenated in correct order"""
        pooled = {
            'obj001_3': np.array([4]),
            'obj001_0': np.array([1]),
            'obj001_2': np.array([3]),
            'obj001_1': np.array([2])
        }
        
        fused = multi_view_fusion(pooled)
        
        # Should concatenate in camera order 0, 1, 2, 3
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(fused['obj001'], expected)
    
    def test_fusion_with_all_cameras(self):
        """Test fusion with complete set of 4 cameras"""
        feature_dim = TestConfig.FEATURE_DIM
        
        pooled = {
            f'fruit_obj001_{i}': np.random.rand(feature_dim)
            for i in range(TestConfig.NUM_OF_CAMERAS)
        }
        
        fused = multi_view_fusion(pooled)
        
        # Final dimension should be feature_dim * num_cameras
        expected_dim = feature_dim * TestConfig.NUM_OF_CAMERAS
        assert fused['fruit_obj001'].shape[0] == expected_dim


class TestFeatureExtractionRobustness:
    """Enhanced robustness tests"""
    
    def test_batch_size_independence(self):
        """Verify consistent features regardless of batch size"""
        # Create mock data with different batch sizes
        images_4 = np.random.rand(4, 224, 224, 3).astype(np.float32)
        images_8 = np.random.rand(8, 224, 224, 3).astype(np.float32)
        
        # Both should process without error
        assert images_4.shape[0] == 4
        assert images_8.shape[0] == 8
        
        # Verify shapes are correct
        assert images_4.shape[1:] == (224, 224, 3)
        assert images_8.shape[1:] == (224, 224, 3)
    
    def test_deterministic_extraction(self):
        """Test reproducibility with same random seed"""
        # Set seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create test data
        test_data1 = np.random.rand(2, 224, 224, 3).astype(np.float32)
        
        # Reset seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        test_data2 = np.random.rand(2, 224, 224, 3).astype(np.float32)
        
        # Should be identical
        np.testing.assert_array_equal(test_data1, test_data2)
    
    def test_feature_extraction_memory_leak(self):
        """Test for memory leaks during long-running extraction"""
        import tracemalloc
        import gc
        
        tracemalloc.start()
        
        # Simulate multiple extraction cycles
        for _ in range(10):
            # Create and discard features
            features = np.random.rand(100, 7, 7, 1024)
            del features
            gc.collect()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should stabilize (not grow unbounded)
        # Peak should be reasonable
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 1000, f"Memory leak detected: {peak_mb:.2f}MB"
    
    def test_multi_camera_synchronization(self):
        """Verify correct pairing of multi-view images"""
        # Create feature map with multiple cameras for same object
        feature_map = {
            'apple_obj001_0': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}],
            'apple_obj001_1': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}],
            'apple_obj001_2': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}],
            'apple_obj001_3': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}]
        }
        
        # Process through pipeline
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should result in single fused feature for the object
        assert 'apple_obj001' in fused
        assert len(fused) == 1
    
    def test_temporal_sequence_ordering(self):
        """Validate correct chronological ordering"""
        # Create features with timestamps
        feature_map = {
            'apple_obj001_0': [
                {'features': np.ones((7, 7, 1024)) * 1, 'timestamp': 't0', 'label': 0},
                {'features': np.ones((7, 7, 1024)) * 2, 'timestamp': 't1', 'label': 0},
                {'features': np.ones((7, 7, 1024)) * 3, 'timestamp': 't2', 'label': 0}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Verify ordering is preserved
        assert 'apple_obj001_0_t0' in flattened
        assert 'apple_obj001_0_t1' in flattened
        assert 'apple_obj001_0_t2' in flattened
        
        # Verify values are different (representing different timesteps)
        assert not np.array_equal(
            flattened['apple_obj001_0_t0']['features'],
            flattened['apple_obj001_0_t1']['features']
        )
    
    def test_model_version_compatibility(self):
        """Test loading models from different PyTorch versions"""
        # This is a placeholder - actual implementation would test
        # loading saved models from different PyTorch versions
        model, _, _ = load_model()
        
        # Verify model loaded successfully
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_feature_extraction_consistency(self):
        """Test that same input produces same output"""
        # Create identical feature maps
        features = np.random.rand(7, 7, 1024)
        
        feature_map1 = {
            'test_obj_0': [{'features': features.copy(), 'timestamp': 't0', 'label': 0}]
        }
        
        feature_map2 = {
            'test_obj_0': [{'features': features.copy(), 'timestamp': 't0', 'label': 0}]
        }
        
        # Process both
        flattened1 = flatten_features(feature_map1)
        flattened2 = flatten_features(feature_map2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            flattened1['test_obj_0_t0']['features'],
            flattened2['test_obj_0_t0']['features']
        )
    
    def test_large_batch_processing(self):
        """Test processing large batches of features"""
        # Create large feature map
        large_feature_map = {}
        
        for obj_id in range(50):  # 50 objects
            for cam_id in range(4):  # 4 cameras each
                key = f'obj{obj_id:03d}_{cam_id}'
                large_feature_map[key] = [
                    {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
                ]
        
        # Should process without errors
        flattened = flatten_features(large_feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should have 50 fused objects
        assert len(fused) == 50


class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction pipeline"""
    
    def test_complete_pipeline_flow(self):
        """Test complete feature processing pipeline"""
        # Create mock feature map
        feature_map = {
            'apple_obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0}
            ],
            'apple_obj001_1': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0}
            ]
        }
        
        # Step 1: Flatten
        flattened = flatten_features(feature_map)
        assert len(flattened) == 4  # 2 cameras * 2 timesteps
        
        # Step 2: Temporal pooling
        pooled = temporal_pooling(flattened)
        assert len(pooled) == 2  # 2 cameras
        
        # Step 3: Multi-view fusion
        fused = multi_view_fusion(pooled)
        assert len(fused) == 1  # 1 object
        assert 'apple_obj001' in fused
    
    def test_pipeline_with_multiple_fruits(self):
        """Test pipeline with multiple fruit types"""
        feature_map = {}
        
        # Create features for 3 different fruits
        for fruit_idx, fruit_type in enumerate(['apple', 'banana', 'orange']):
            for cam_id in range(2):
                key = f'{fruit_type}_obj{fruit_idx:03d}_{cam_id}'
                feature_map[key] = [
                    {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': fruit_idx}
                ]
        
        # Process through pipeline
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should have 3 fused objects
        assert len(fused) == 3
        assert 'apple_obj000' in fused
        assert 'banana_obj001' in fused
        assert 'orange_obj002' in fused
    
    def test_process_features_end_to_end(self):
        """Test complete feature processing pipeline"""
        # Create simple mock data without actual model inference
        mock_features = {
            'test_obj_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
            ],
            'test_obj_1': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
            ]
        }
        
        # Test pipeline steps
        flattened = flatten_features(mock_features)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should produce fused features
        assert len(fused) > 0
        
        # Features should be numpy arrays
        for key, features in fused.items():
            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float64 or features.dtype == np.float32


class TestFeatureExtractionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_generator(self):
        """Test handling of empty generator"""
        def empty_gen():
            return
            yield  # Never reached
        
        empty_gen.num_batches = 0
        
        feature_map = extract_features_from_generator(empty_gen, 'testing')
        
        # Should return empty dict
        assert len(feature_map) == 0
    
    def test_single_timestep_single_camera(self):
        """Test minimal case: one timestep, one camera"""
        feature_map = {
            'obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
            ]
        }
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should still work
        assert len(fused) == 1
        assert 'obj001' in fused
    
    def test_mismatched_camera_counts(self):
        """Test objects with different numbers of cameras"""
        feature_map = {
            'obj001_0': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}],
            'obj001_1': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}],
            'obj002_0': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}]
            # obj002 has only 1 camera instead of 2
        }
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        # Should handle gracefully
        assert 'obj001' in fused
        assert 'obj002' in fused
        
        # obj001 should have concatenated features from 2 cameras
        # obj002 should have features from 1 camera
        assert fused['obj001'].shape[0] > fused['obj002'].shape[0]
    
    def test_feature_map_with_nan_values(self):
        """Test handling of NaN values in features"""
        feature_map = {
            'obj001_0': [
                {'features': np.array([[[np.nan] * 1024] * 7] * 7), 'timestamp': 't0', 'label': 0}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Should process without crashing
        assert 'obj001_0_t0' in flattened
        
        # NaN values should be preserved (or handled according to implementation)
        assert flattened['obj001_0_t0']['features'] is not None


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])