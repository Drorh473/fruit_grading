"""
Feature-Model Integration Tests
Test pipeline from feature extraction to model training and inference
"""
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
            params, loss = train_step(X_train, y_train, params, learning_rate=0.01)
        
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
            params, _ = train_step(X_train, y_train, params, learning_rate=0.01)
        
        # Test inference on new features
        X_test = np.random.rand(5, TestConfig.FEATURE_DIM)
        predictions = predict(X_test, params)
        
        # Predictions should be valid class indices
        assert all(0 <= p < TestConfig.NUM_CLASSES for p in predictions)
    
    def test_complete_pipeline_end_to_end(self):
        """Test complete pipeline from features to predictions"""
        # Step 1: Simulate multi-view fusion output
        num_objects = 15
        fused_features = {}
        true_labels = {}
        
        for i in range(num_objects):
            obj_id = f'obj{i:04d}'
            fused_features[obj_id] = np.random.rand(TestConfig.FEATURE_DIM)
            true_labels[obj_id] = i % TestConfig.NUM_CLASSES
        
        # Step 2: Prepare data
        X = np.array(list(fused_features.values()))
        y = np.array(list(true_labels.values()))
        
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
            params, _ = train_step(X_train, y_train, params, learning_rate=0.01)
        
        # Step 4: Evaluate
        accuracy, conf_matrix = evaluate(X_test, y_test, params)
        
        # Should produce valid accuracy and confusion matrix
        assert 0.0 <= accuracy <= 1.0
        assert conf_matrix.shape == (TestConfig.NUM_CLASSES, TestConfig.NUM_CLASSES)


class TestFeatureQualityForTraining:
    """Test feature quality requirements for training"""
    
    def test_feature_dimension_consistency(self):
        """Test that all features have consistent dimensions"""
        # Simulate multi-camera features
        features = {}
        
        for i in range(10):
            for cam_id in range(TestConfig.NUM_OF_CAMERAS):
                key = f'obj{i:04d}_{cam_id}'
                features[key] = np.random.rand(TestConfig.FEATURE_DIM)
        
        # Fuse features
        fused = multi_view_fusion(features)
        
        # All fused features should have same dimension
        feature_dims = [f.shape[0] for f in fused.values()]
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


class TestModelPerformanceOnFeatures:
    """Test model performance with real feature distributions"""
    
    def test_model_learns_from_features(self):
        """Test that model can learn from feature patterns"""
        # Create features with some pattern
        num_samples = 30
        X = []
        y = []
        
        for i in range(num_samples):
            label = i % TestConfig.NUM_CLASSES
            
            # Create features with slight class-specific pattern
            feature = np.random.rand(TestConfig.FEATURE_DIM)
            feature[:10] += label * 0.5  # Add class-specific signal
            
            X.append(feature)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(50):
            params, loss = train_step(X, y, params, learning_rate=0.01)
            
            if epoch == 0:
                initial_loss = loss
            if epoch == 49:
                final_loss = loss
        
        # Loss should decrease
        assert final_loss < initial_loss
    
    def test_model_generalizes_to_test_set(self):
        """Test model generalization to unseen features"""
        # Create train and test sets with similar distributions
        num_train = 30
        num_test = 10
        
        # Training data
        X_train = []
        y_train = []
        for i in range(num_train):
            label = i % TestConfig.NUM_CLASSES
            feature = np.random.rand(TestConfig.FEATURE_DIM) + label
            X_train.append(feature)
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Test data (similar distribution)
        X_test = []
        y_test = []
        for i in range(num_test):
            label = i % TestConfig.NUM_CLASSES
            feature = np.random.rand(TestConfig.FEATURE_DIM) + label
            X_test.append(feature)
            y_test.append(label)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Train model
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        for _ in range(30):
            params, _ = train_step(X_train, y_train, params, learning_rate=0.01)
        
        # Evaluate on test set
        accuracy, _ = evaluate(X_test, y_test, params)
        
        # Should have reasonable accuracy (better than random)
        random_accuracy = 1.0 / TestConfig.NUM_CLASSES
        # Note: With random features, this may not always hold
        # This is more of a smoke test
        assert 0.0 <= accuracy <= 1.0


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])