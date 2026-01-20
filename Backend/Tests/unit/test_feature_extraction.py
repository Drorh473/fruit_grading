import pytest
import numpy as np
import torch
from pathlib import Path


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


# Cache the model at module level to avoid repeated downloads during tests
@pytest.fixture(scope="module")
def cached_model():
    """Load model once for all tests in this module to avoid timeout from repeated downloads"""
    return load_model()


class TestLoadModel:
    """Test cases for model loading"""
    
    def test_load_model_returns_correct_components(self, cached_model):
        """Test that load_model returns model, feature_extractor, and device"""
        model, feature_extractor, device = cached_model
        
        assert model is not None
        assert feature_extractor is not None
        assert device is not None
        assert str(device) in ['cuda', 'cpu']
    
    def test_feature_extractor_has_no_classifier(self, cached_model):
        """Test that feature_extractor excludes the final classification layer"""
        model, feature_extractor, _ = cached_model
        
        # Feature extractor should have fewer layers than full model
        num_model_children = len(list(model.children()))
        num_extractor_children = len(list(feature_extractor.children()))
        
        assert num_extractor_children < num_model_children
    
    def test_model_device_consistency(self, cached_model):
        """Test that model is on the correct device"""
        model, feature_extractor, device = cached_model
        
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
                    'label': 0,
                    'fruit_type': 'apple'
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
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'apple'},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0, 'fruit_type': 'apple'},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't2', 'label': 0, 'fruit_type': 'apple'}
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
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 1, 'fruit_type': 'banana'}
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
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 2, 'fruit_type': 'unknown'}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        assert flattened['apple_obj001_0_t0']['label'] == 2
        assert flattened['apple_obj001_0_t0']['fruit_type'] == 'unknown'
    
    def test_flatten_different_feature_dimensions(self):
        """Test flattening with different feature map sizes"""
        feature_map_small = {
            'test_small': [
                {'features': np.random.rand(3, 3, 512), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}
            ]
        }
        
        feature_map_large = {
            'test_large': [
                {'features': np.random.rand(14, 14, 2048), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}
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
                'group_key': 'apple_obj001_0',
                'label': 0,
                'fruit_type': 'apple'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        assert 'apple_obj001_0' in pooled
        # Access the 'features' key from the dict
        np.testing.assert_array_equal(pooled['apple_obj001_0']['features'], np.array([1, 2, 3, 4, 5]))
    
    def test_temporal_pooling_multiple_frames(self):
        """Test pooling averages across multiple frames"""
        flattened = {
            'apple_obj001_0_t0': {
                'features': np.array([2.0, 4.0, 6.0]),
                'group_key': 'apple_obj001_0',
                'label': 0,
                'fruit_type': 'apple'
            },
            'apple_obj001_0_t1': {
                'features': np.array([4.0, 6.0, 8.0]),
                'group_key': 'apple_obj001_0',
                'label': 0,
                'fruit_type': 'apple'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        # Average should be [3.0, 5.0, 7.0]
        expected = np.array([3.0, 5.0, 7.0])
        # Access the 'features' key
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0']['features'], expected)
    
    def test_temporal_pooling_empty(self):
        """Test temporal pooling with empty input"""
        pooled = temporal_pooling({})
        assert len(pooled) == 0
    
    def test_temporal_pooling_many_frames(self):
        """Test pooling with many temporal frames"""
        flattened = {}
        for i in range(10):
            flattened[f'apple_obj001_0_t{i}'] = {
                'features': np.ones(100) * i,
                'group_key': 'apple_obj001_0',
                'label': 0,
                'fruit_type': 'apple'
            }
        
        pooled = temporal_pooling(flattened)
        
        # Average should be 4.5 (mean of 0-9)
        expected = np.ones(100) * 4.5
        # Access the 'features' key
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0']['features'], expected)
    
    def test_temporal_pooling_preserves_dimensions(self):
        """Test that pooling preserves feature dimensions"""
        feature_dim = 50176  # 7 * 7 * 1024
        
        flattened = {
            'test_obj_0_t0': {
                'features': np.random.rand(feature_dim),
                'group_key': 'test_obj_0',
                'label': 0,
                'fruit_type': 'test'
            },
            'test_obj_0_t1': {
                'features': np.random.rand(feature_dim),
                'group_key': 'test_obj_0',
                'label': 0,
                'fruit_type': 'test'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        # Access the 'features' key
        assert pooled['test_obj_0']['features'].shape[0] == feature_dim
    
    def test_temporal_pooling_preserves_labels(self):
        """Test that labels are preserved through pooling"""
        flattened = {
            'test_obj_0_t0': {
                'features': np.array([1.0, 2.0]),
                'group_key': 'test_obj_0',
                'label': 2,
                'fruit_type': 'unknown'
            }
        }
        
        pooled = temporal_pooling(flattened)
        
        #Check label and fruit_type in the dict
        assert pooled['test_obj_0']['label'] == 2
        assert pooled['test_obj_0']['fruit_type'] == 'unknown'


class TestMultiViewFusion:
    """Test cases for multi-view fusion"""
    def test_fusion_multiple_cameras(self):
        """Test fusion concatenates features from multiple cameras"""
        pooled = {
            'apple_obj001_0': {'features': np.array([1, 2, 3]), 'label': 0, 'fruit_type': 'apple'},
            'apple_obj001_1': {'features': np.array([4, 5, 6]), 'label': 0, 'fruit_type': 'apple'},
            'apple_obj001_2': {'features': np.array([7, 8, 9]), 'label': 0, 'fruit_type': 'apple'}
        }
        
        fused = multi_view_fusion(pooled, target_views=3)
        
        # Should concatenate all camera views
        assert 'apple_obj001' in fused
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        #Access the 'features' key
        np.testing.assert_array_equal(fused['apple_obj001']['features'], expected)
    
    def test_fusion_multiple_objects(self):
        """Test fusion with multiple objects"""
        pooled = {
            'apple_obj001_0': {'features': np.array([1, 2]), 'label': 0, 'fruit_type': 'apple'},
            'apple_obj001_1': {'features': np.array([3, 4]), 'label': 0, 'fruit_type': 'apple'},
            'banana_obj002_0': {'features': np.array([5, 6]), 'label': 1, 'fruit_type': 'banana'},
            'banana_obj002_1': {'features': np.array([7, 8]), 'label': 1, 'fruit_type': 'banana'}
        }
        
        fused = multi_view_fusion(pooled, target_views=2)
        
        # Should have 2 objects
        assert len(fused) == 2
        assert 'apple_obj001' in fused
        assert 'banana_obj002' in fused
        
        #Access the 'features' key
        np.testing.assert_array_equal(fused['apple_obj001']['features'], np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(fused['banana_obj002']['features'], np.array([5, 6, 7, 8]))
    
    def test_fusion_correct_dimensions(self):
        """Test that fusion produces correct feature dimensions"""
        pooled = {
            f'apple_obj001_{i}': {'features': np.random.rand(1024), 'label': 0, 'fruit_type': 'apple'}
            for i in range(4)
        }
        
        fused = multi_view_fusion(pooled, target_views=4)
        
        # Should concatenate to 4096 dimensions (4 * 1024)
        # Access the 'features' key
        assert fused['apple_obj001']['features'].shape[0] == 4096
    
    def test_multi_view_fusion_empty(self):
        """Test multi-view fusion with empty input"""
        fused = multi_view_fusion({})
        assert len(fused) == 0
    
    def test_fusion_with_padding(self):
        """Test fusion pads missing cameras with zeros"""
        pooled = {
            'obj001_0': {'features': np.array([1, 2]), 'label': 0, 'fruit_type': 'test'},
            'obj001_1': {'features': np.array([3, 4]), 'label': 0, 'fruit_type': 'test'}
            # Only 2 cameras, but target is 4
        }
        
        fused = multi_view_fusion(pooled, target_views=4)
        
        # Should pad with zeros
        # Access the 'features' key
        expected = np.array([1, 2, 3, 4, 0, 0, 0, 0])
        np.testing.assert_array_equal(fused['obj001']['features'], expected)
    
    def test_fusion_preserves_labels(self):
        """Test that labels are preserved through fusion"""
        pooled = {
            'test_obj_0': {'features': np.array([1, 2]), 'label': 2, 'fruit_type': 'unknown'}
        }
        
        fused = multi_view_fusion(pooled, target_views=1)
        
        # Check label and fruit_type in the dict
        assert fused['test_obj']['label'] == 2
        assert fused['test_obj']['fruit_type'] == 'unknown'


class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction pipeline"""
    
    def test_complete_pipeline_flow(self):
        """Test complete feature processing pipeline"""
        feature_map = {
            'obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'apple'},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0, 'fruit_type': 'apple'}
            ],
            'obj001_1': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'apple'},
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't1', 'label': 0, 'fruit_type': 'apple'}
            ]
        }
        
        # Step 1: Flatten
        flattened = flatten_features(feature_map)
        assert len(flattened) == 4  # 2 cameras * 2 timesteps
        
        # Step 2: Temporal pooling
        pooled = temporal_pooling(flattened)
        assert len(pooled) == 2  # 2 cameras
        
        # Step 3: Multi-view fusion
        fused = multi_view_fusion(pooled, target_views=2)
        assert len(fused) == 1  # 1 object
        assert 'obj001' in fused
        
        # Verify structure
        assert 'features' in fused['obj001']
        assert 'label' in fused['obj001']
        assert isinstance(fused['obj001']['features'], np.ndarray)
    
    def test_pipeline_with_multiple_fruits(self):
        """Test pipeline with multiple fruit types"""
        feature_map = {}
        
        for fruit_idx, fruit_type in enumerate(['standard', 'premium', 'market']):
            for cam_id in range(2):
                key = f'{fruit_type}_obj{fruit_idx:03d}_{cam_id}'
                feature_map[key] = [
                    {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': fruit_idx, 'fruit_type': fruit_type}
                ]
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled, target_views=2)
        
        # Should have 3 fused objects
        assert len(fused) == 3
        assert 'standard_obj000' in fused
        assert 'premium_obj001' in fused
        assert 'market_obj002' in fused
        
        # Verify labels are preserved
        assert fused['standard_obj000']['label'] == 0
        assert fused['premium_obj001']['label'] == 1
        assert fused['market_obj002']['label'] == 2


class TestFeatureExtractionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_timestep_single_camera(self):
        """Test minimal case: one timestep, one camera"""
        feature_map = {
            'obj001_0': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}
            ]
        }
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled, target_views=1)
        
        # Should still work
        assert len(fused) == 1
        assert 'obj001' in fused
        # Access the 'features' key
        assert isinstance(fused['obj001']['features'], np.ndarray)
    
    def test_mismatched_camera_counts(self):
        """Test objects with different numbers of cameras"""
        feature_map = {
            'obj001_0': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}],
            'obj001_1': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}],
            'obj002_0': [{'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}]
        }
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled, target_views=2)
        
        # Should handle gracefully with padding
        assert 'obj001' in fused
        assert 'obj002' in fused
        
        # Access the 'features' key
        # Both should have same dimension due to padding
        assert fused['obj001']['features'].shape[0] == fused['obj002']['features'].shape[0]
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])