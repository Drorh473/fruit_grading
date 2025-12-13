import unittest
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cnn.pre_trained_feature_map import (
    load_model,
    extract_features_from_generator,
    flatten_features,
    temporal_pooling,
    multi_view_fusion,
    process_features
)

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class TestLoadModel(unittest.TestCase):
    """Test cases for model loading"""
    
    def test_load_model_returns_correct_components(self):
        """Test that load_model returns model, feature_extractor, and device"""
        model, feature_extractor, device = load_model()
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(feature_extractor)
        self.assertIsNotNone(device)
        self.assertIn(str(device), ['cuda', 'cpu'])
    
    def test_model_in_eval_mode(self):
        """Test that model is in evaluation mode"""
        model, _, _ = load_model()
        self.assertFalse(model.training)
    
    def test_feature_extractor_has_no_classifier(self):
        """Test that feature_extractor excludes the final classification layer"""
        model, feature_extractor, _ = load_model()
        
        # Feature extractor should have fewer layers than full model
        num_model_children = len(list(model.children()))
        num_extractor_children = len(list(feature_extractor.children()))
        
        self.assertLess(num_extractor_children, num_model_children)


class TestFlattenFeatures(unittest.TestCase):
    """Test cases for feature flattening"""
    
    def test_flatten_single_feature(self):
        """Test flattening a single feature map"""
        # Create mock feature data
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
        self.assertIn('apple_obj001_0_t0', flattened)
        self.assertEqual(flattened['apple_obj001_0_t0']['group_key'], 'apple_obj001_0')
        
        # Check flattened dimension (7 * 7 * 1024 = 50176)
        self.assertEqual(flattened['apple_obj001_0_t0']['features'].shape[0], 7 * 7 * 1024)
    
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
        self.assertEqual(len(flattened), 3)
        self.assertIn('apple_obj001_0_t0', flattened)
        self.assertIn('apple_obj001_0_t1', flattened)
        self.assertIn('apple_obj001_0_t2', flattened)
    
    def test_flatten_preserves_group_key(self):
        """Test that group_key is preserved correctly"""
        feature_map = {
            'banana_obj002_1': [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 1}
            ]
        }
        
        flattened = flatten_features(feature_map)
        
        self.assertEqual(flattened['banana_obj002_1_t0']['group_key'], 'banana_obj002_1')


class TestTemporalPooling(unittest.TestCase):
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
        
        self.assertIn('apple_obj001_0', pooled)
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
        self.assertEqual(len(pooled), 2)
        self.assertIn('apple_obj001_0', pooled)
        self.assertIn('apple_obj001_1', pooled)
        
        # Check averages
        np.testing.assert_array_almost_equal(pooled['apple_obj001_0'], np.array([2.0, 3.0]))
        np.testing.assert_array_almost_equal(pooled['apple_obj001_1'], np.array([5.0, 6.0]))


class TestMultiViewFusion(unittest.TestCase):
    """Test cases for multi-view fusion"""
    
    def test_fusion_single_camera(self):
        """Test fusion with single camera (no concatenation needed)"""
        pooled = {
            'apple_obj001_0': np.array([1, 2, 3, 4, 5])
        }
        
        fused = multi_view_fusion(pooled)
        
        self.assertIn('apple_obj001', fused)
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
        self.assertIn('apple_obj001', fused)
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
        self.assertEqual(len(fused), 2)
        self.assertIn('apple_obj001', fused)
        self.assertIn('banana_obj002', fused)
        
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
        self.assertEqual(fused['apple_obj001'].shape[0], 4096)


class TestFeatureExtractionIntegration(unittest.TestCase):
    """Integration tests for feature extraction pipeline"""
    
    def setUp(self):
        """Create mock generator for testing"""
        # Create mock batch data
        self.mock_images = np.random.rand(2, 224, 224, 3).astype(np.float32)
        self.mock_metadata = [
            {
                'fruit_type': 'apple',
                'object_id': 'obj001',
                'camera_id': 0,
                'timestamp': '2025-01-01T00:00:00'
            },
            {
                'fruit_type': 'apple',
                'object_id': 'obj001',
                'camera_id': 0,
                'timestamp': '2025-01-01T00:00:01'
            }
        ]
    
    def create_mock_generator(self):
        """Create a mock generator that yields one batch"""
        def gen():
            yield (self.mock_images, self.mock_metadata)
        
        gen_func = gen
        gen_func.num_batches = 1
        return gen_func
    
    def test_extract_features_from_generator_structure(self):
        """Test that extract_features_from_generator returns correct structure"""
        generator = self.create_mock_generator()
        
        feature_map = extract_features_from_generator(generator, 'testing')
        
        # Should have features for the object
        self.assertGreater(len(feature_map), 0)
        
        # Check structure of first entry
        first_key = list(feature_map.keys())[0]
        self.assertIsInstance(feature_map[first_key], list)
        self.assertIn('features', feature_map[first_key][0])
        self.assertIn('timestamp', feature_map[first_key][0])
    
    def test_process_features_end_to_end(self):
        """Test complete feature processing pipeline"""
        generator = self.create_mock_generator()
        
        # Run complete pipeline
        fused = process_features(generator, 'testing')
        
        # Should produce fused features
        self.assertGreater(len(fused), 0)
        
        # Features should be numpy arrays
        first_key = list(fused.keys())[0]
        self.assertIsInstance(fused[first_key], np.ndarray)


class TestFeatureExtractionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_generator(self):
        """Test handling of empty generator"""
        def empty_gen():
            return
            yield  # Never reached
        
        empty_gen.num_batches = 0
        
        feature_map = extract_features_from_generator(empty_gen, 'testing')
        
        # Should return empty dict
        self.assertEqual(len(feature_map), 0)
    
    def test_flatten_empty_features(self):
        """Test flattening with empty input"""
        flattened = flatten_features({})
        self.assertEqual(len(flattened), 0)
    
    def test_temporal_pooling_empty(self):
        """Test temporal pooling with empty input"""
        pooled = temporal_pooling({})
        self.assertEqual(len(pooled), 0)
    
    def test_multi_view_fusion_empty(self):
        """Test multi-view fusion with empty input"""
        fused = multi_view_fusion({})
        self.assertEqual(len(fused), 0)


if __name__ == '__main__':
    unittest.main()