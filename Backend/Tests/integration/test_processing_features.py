import pytest
import numpy as np
from pathlib import Path
from Tests.test_config import TestConfig
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.preprocessing_from_db import custom_preprocessing
from cnn.pre_trained_feature_map import (
    flatten_features,
    temporal_pooling,
    multi_view_fusion
)


class TestPreprocessedToFeatures:
    """Test preprocessed images to feature extraction pipeline"""
    
    def test_preprocessed_images_feature_extraction(self, sample_image):
        """Test extracting features from preprocessed images"""
        # Step 1: Preprocess image
        preprocessed = custom_preprocessing(sample_image)
        
        assert preprocessed is not None
        assert preprocessed.dtype == np.float32
        
        # Step 2: Simulate feature extraction (mock)
        # In real scenario, this would go through CNN
        mock_features = np.random.rand(7, 7, 1024)
        
        # Step 3: Flatten features
        feature_map = {
            'test_obj_0': [
                {'features': mock_features, 'timestamp': 't0', 'label': 0}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        assert 'test_obj_0_t0' in flattened
        assert flattened['test_obj_0_t0']['features'].shape[0] == 7 * 7 * 1024
    
    def test_multi_camera_feature_fusion(self):
        """Test complete multi-view fusion pipeline"""
        # Simulate features from 4 cameras
        camera_features = {}
        
        for camera_id in range(TestConfig.NUM_OF_CAMERAS):
            key = f'apple_obj001_{camera_id}'
            camera_features[key] = [
                {
                    'features': np.random.rand(7, 7, 1024),
                    'timestamp': 't0',
                    'label': 0
                }
            ]
        
        # Step 1: Flatten
        flattened = flatten_features(camera_features)
        assert len(flattened) == TestConfig.NUM_OF_CAMERAS
        
        # Step 2: Temporal pooling
        pooled = temporal_pooling(flattened)
        assert len(pooled) == TestConfig.NUM_OF_CAMERAS
        
        # Step 3: Multi-view fusion
        fused = multi_view_fusion(pooled)
        
        assert 'apple_obj001' in fused
        assert len(fused) == 1
        
        # Final feature should be concatenation of all cameras
        expected_dim = TestConfig.FEATURE_DIM * TestConfig.NUM_OF_CAMERAS
        assert fused['apple_obj001'].shape[0] == expected_dim
    
    def test_batch_processing_consistency(self):
        """Test batch size doesn't affect final features"""
        # Create same features in different batch sizes
        features = np.random.rand(7, 7, 1024)
        
        # Batch 1: Process individually
        feature_map_1 = {
            'obj001_0': [{'features': features.copy(), 'timestamp': 't0', 'label': 0}]
        }
        
        flattened_1 = flatten_features(feature_map_1)
        pooled_1 = temporal_pooling(flattened_1)
        fused_1 = multi_view_fusion(pooled_1)
        
        # Batch 2: Same features
        feature_map_2 = {
            'obj001_0': [{'features': features.copy(), 'timestamp': 't0', 'label': 0}]
        }
        
        flattened_2 = flatten_features(feature_map_2)
        pooled_2 = temporal_pooling(flattened_2)
        fused_2 = multi_view_fusion(pooled_2)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(fused_1['obj001'], fused_2['obj001'])


class TestPreprocessingQuality:
    """Test preprocessing quality for feature extraction"""
    
    def test_preprocessing_normalization(self, sample_image):
        """Test that preprocessing produces properly normalized images"""
        preprocessed = custom_preprocessing(sample_image)
        
        # Should be in [0, 1] range
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 1.0
        
        # Should be float32
        assert preprocessed.dtype == np.float32
class TestFeaturePipelineRobustness:
    """Test robustness of preprocessing to feature pipeline"""
    
    def test_handle_varying_image_sizes(self):
        """Test pipeline handles different input sizes"""
        sizes = [(100, 100, 3), (300, 400, 3), (500, 200, 3)]
        
        for size in sizes:
            # Create and preprocess
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            preprocessed = custom_preprocessing(img)
            
            # Should handle gracefully
            assert preprocessed is not None
            assert preprocessed.dtype == np.float32
    
    def test_temporal_sequence_consistency(self):
        """Test temporal sequence maintains order"""
        # Create sequence of features
        feature_map = {
            'obj001_0': [
                {'features': np.ones((7, 7, 1024)) * i, 'timestamp': f't{i}', 'label': 0}
                for i in range(5)
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        # Verify all timesteps present
        for i in range(5):
            assert f'obj001_0_t{i}' in flattened
            
if __name__ == "__main__":
    pytest.main([__file__, "-v"])