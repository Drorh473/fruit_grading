"""
Integration tests for preprocessing to feature extraction pipeline.
Tests the full flow from raw images through preprocessing to fused features.

Note: Unit tests for individual functions (flatten_features, temporal_pooling,
multi_view_fusion) are in Tests/unit/test_feature_extraction.py
"""
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


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


class TestPreprocessingToFeaturesPipeline:
    """Integration tests for complete preprocessing-to-features pipeline.

    These tests verify the end-to-end flow from raw images through
    preprocessing to final fused feature vectors. Individual function
    unit tests are in test_feature_extraction.py.
    """

    def test_full_pipeline_single_object(self, sample_image):
        """Test complete pipeline for a single object"""
        # Step 1: Preprocess image
        preprocessed = custom_preprocessing(sample_image)

        assert preprocessed is not None
        assert preprocessed.dtype == np.float32
        assert preprocessed.max() <= 1.0
        assert preprocessed.min() >= 0.0

        # Step 2: Simulate feature extraction output (mock CNN features)
        mock_features = np.random.rand(7, 7, 1024)

        # Step 3: Run through fusion pipeline
        feature_map = {
            'test_obj_0': [
                {'features': mock_features, 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}
            ]
        }

        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)

        # Verify final output
        assert 'test_obj' in fused
        assert 'features' in fused['test_obj']
        assert 'label' in fused['test_obj']
        assert fused['test_obj']['features'].shape[0] == 7 * 7 * 1024

    def test_full_pipeline_multi_camera(self, sample_image):
        """Test complete pipeline with multiple camera views"""
        # Preprocess multiple images (simulating multi-camera setup)
        preprocessed_images = [custom_preprocessing(sample_image) for _ in range(TestConfig.NUM_OF_CAMERAS)]

        for img in preprocessed_images:
            assert img is not None
            assert img.dtype == np.float32

        # Simulate features from all cameras
        camera_features = {}
        for camera_id in range(TestConfig.NUM_OF_CAMERAS):
            key = f'apple_obj001_{camera_id}'
            camera_features[key] = [
                {
                    'features': np.random.rand(7, 7, 1024),
                    'timestamp': 't0',
                    'label': 0,
                    'fruit_type': 'apple'
                }
            ]

        # Run fusion pipeline
        flattened = flatten_features(camera_features)
        assert len(flattened) == TestConfig.NUM_OF_CAMERAS

        pooled = temporal_pooling(flattened)
        assert len(pooled) == TestConfig.NUM_OF_CAMERAS

        fused = multi_view_fusion(pooled)

        # Should fuse all cameras into one object
        assert 'apple_obj001' in fused
        assert len(fused) == 1

        # Verify feature dimension
        expected_dim = TestConfig.FEATURE_DIM
        assert fused['apple_obj001']['features'].shape[0] == expected_dim

    def test_full_pipeline_multiple_objects(self, sample_image):
        """Test pipeline with multiple fruit objects of different types"""
        # Create features for multiple objects
        feature_map = {}
        fruit_types = ['market', 'standard', 'premium']

        for obj_idx, fruit_type in enumerate(fruit_types):
            for cam_id in range(2):  # 2 cameras per object
                key = f'{fruit_type}_obj{obj_idx:03d}_{cam_id}'
                feature_map[key] = [
                    {
                        'features': np.random.rand(7, 7, 1024),
                        'timestamp': 't0',
                        'label': obj_idx,
                        'fruit_type': fruit_type
                    }
                ]

        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled, target_views=2)

        # Should have 3 objects
        assert len(fused) == 3
        assert 'market_obj000' in fused
        assert 'standard_obj001' in fused
        assert 'premium_obj002' in fused

        # Verify labels preserved through pipeline
        assert fused['market_obj000']['label'] == 0
        assert fused['standard_obj001']['label'] == 1
        assert fused['premium_obj002']['label'] == 2


class TestPipelineConsistency:
    """Test consistency and determinism of the full pipeline"""

    def test_pipeline_produces_consistent_results(self, sample_image):
        """Test pipeline produces identical results for identical inputs"""
        np.random.seed(42)
        features = np.random.rand(7, 7, 1024)

        feature_map = {
            'obj001_0': [{'features': features.copy(), 'timestamp': 't0', 'label': 0, 'fruit_type': 'test'}]
        }

        # Run pipeline twice
        result1 = multi_view_fusion(temporal_pooling(flatten_features(feature_map)))
        result2 = multi_view_fusion(temporal_pooling(flatten_features(feature_map)))

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1['obj001']['features'],
            result2['obj001']['features']
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
