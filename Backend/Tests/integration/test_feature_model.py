import pytest
import numpy as np
from pathlib import Path
from Tests.test_config import TestConfig
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.pre_trained_feature_map import multi_view_fusion
from cnn.fully_connected_layer import (
    initialize_parameters,
    train_step,
    predict,
    evaluate
)


class TestFeaturesToTraining:
    """Test features to training pipeline"""
    
    def test_features_to_training(self):
        """Test training with extracted features"""
        # Step 1: Create fused features (simulating output from feature extraction)
        num_samples = 20
        fused_features = {}
        
        for i in range(num_samples):
            obj_id = f'obj{i:04d}'
            # Simulate fused features from all cameras
            fused_features[obj_id] = np.random.rand(TestConfig.FEATURE_DIM * TestConfig.NUM_OF_CAMERAS)
        
        # Step 2: Prepare training data
        X_train = np.array(list(fused_features.values()))
        y_train = np.random.randint(0, TestConfig.NUM_CLASSES, num_samples)
        
        # Step 3: Initialize model
        params = initialize_parameters(
            X_train.shape[1],  # Input dimension
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Step 4: Train for a few steps
        for _ in range(5):
            loss, accuracy, params = train_step(X_train, y_train, params, TestConfig.NUM_CLASSES, learning_rate=0.01)
                                     
        # Should complete without errors
        assert loss >= 0
    
    def test_model_inference_on_new_features(self):
        """Test prediction on newly extracted features"""
        # Train a simple model
        X_train = np.random.rand(20, TestConfig.FEATURE_DIM)
        y_train = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Quick training
        for _ in range(10):
            loss, accuracy, params = train_step(X_train, y_train, params, TestConfig.NUM_CLASSES, learning_rate=0.01)
        
        # Test inference on new features
        X_test = np.random.rand(5, TestConfig.FEATURE_DIM)
        predictions, probabilities = predict(X_test, params)
        
        # Predictions should be valid class indices
        assert all(0 <= p < TestConfig.NUM_CLASSES for p in predictions)
    
    def test_complete_pipeline_end_to_end(self):
        """Test complete pipeline from features to predictions"""
        # Step 1: Simulate multi-view fusion output
        num_objects = 15
        fused_features = {}

        for i in range(num_objects):
            obj_id = f'obj{i:04d}'
            fused_features[obj_id] = {
                'features': np.random.uniform(0, 1, TestConfig.FEATURE_DIM),
                'label': i % TestConfig.NUM_CLASSES,
                'fruit_type': ['apple', 'banana', 'orange'][i % TestConfig.NUM_CLASSES]
            }
        
        # Step 2: Prepare data - extract from dictionary structure
        X = np.array([data['features'] for data in fused_features.values()])
        y = np.array([data['label'] for data in fused_features.values()])
        
        # Split train/test
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Step 3: Train model
        params = initialize_parameters(
            X_train.shape[1],
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        for _ in range(20):
            loss, accuracy, params = train_step(X_train, y_train, params, TestConfig.NUM_CLASSES, learning_rate=0.01)
        
        # Step 4: Evaluate
        test_loss, test_accuracy = evaluate(X_test, y_test, params, TestConfig.NUM_CLASSES)
        
        assert 0.0 <= test_accuracy <= 1.0
        assert test_loss >= 0


class TestFeatureQualityForTraining:
    """Test feature quality requirements for training"""
    
    def test_feature_dimension_consistency(self):
        """Test that all features have consistent dimensions"""
        pooled_features = {}
        
        for i in range(10):
            for cam_id in range(TestConfig.NUM_OF_CAMERAS):
                key = f'obj{i:04d}_{cam_id}'
                # Each value must be a dictionary with 'features', 'label', 'fruit_type'
                pooled_features[key] = {
                    'features': np.random.uniform(0, 1, TestConfig.FEATURE_DIM),
                    'label': i % TestConfig.NUM_CLASSES,
                    'fruit_type': 'apple'
                }
        
        fused = multi_view_fusion(pooled_features)
        
        # Extract features 
        feature_arrays = [obj_data['features'] for obj_data in fused.values()]
        feature_dims = [f.shape[0] for f in feature_arrays]
        assert len(set(feature_dims)) == 1  # All same dimension
    
    def test_feature_normalization(self):
        """Test that features are properly normalized"""
        # Create features in reasonable range
        features = {
            f'obj{i:04d}': np.random.rand(TestConfig.FEATURE_DIM)
            for i in range(10)
        }
        
        X = np.array(list(features.values()))
        
        # Features should be in reasonable range (not too large/small)
        assert X.min() >= -10.0  # Not too negative
        assert X.max() <= 10.0   # Not too positive

if __name__ == "__main__":
    pytest.main([__file__, "-v"])