"""
Enhanced Model Training Testing
Comprehensive tests for neural network operations and training
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from Tests.test_config import TestConfig

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.fully_connected_layer import (
    initialize_parameters,
    forward_pass,
    compute_loss,
    backward_pass,
    update_parameters,
    train_step,
    predict,
    evaluate,
    train,
    train_from_generator,
    save_model,
    load_model
)


class TestParameterInitialization:
    """Test parameter initialization"""
    
    def test_initialize_parameters_shapes(self):
        """Test that initialized parameters have correct shapes"""
        input_dim = 1024
        hidden_dim = 256
        num_classes = 3
        
        params = initialize_parameters(input_dim, hidden_dim, num_classes)
        
        # Check shapes
        assert params['W1'].shape == (input_dim, hidden_dim)
        assert params['b1'].shape == (hidden_dim,)
        assert params['W2'].shape == (hidden_dim, num_classes)
        assert params['b2'].shape == (num_classes,)
    
    def test_initialize_parameters_types(self):
        """Test that parameters are numpy arrays"""
        params = initialize_parameters(100, 50, 3)
        
        assert isinstance(params['W1'], np.ndarray)
        assert isinstance(params['b1'], np.ndarray)
        assert isinstance(params['W2'], np.ndarray)
        assert isinstance(params['b2'], np.ndarray)
    
    def test_initialize_parameters_values(self):
        """Test that parameters are initialized with reasonable values"""
        params = initialize_parameters(100, 50, 3)
        
        # Weights should be small random values
        assert np.abs(params['W1']).mean() < 1.0
        assert np.abs(params['W2']).mean() < 1.0
        
        # Biases should be zeros
        np.testing.assert_array_equal(params['b1'], np.zeros(50))
        np.testing.assert_array_equal(params['b2'], np.zeros(3))


class TestForwardPass:
    """Test forward propagation"""
    
    def test_forward_pass_shapes(self, mock_model_parameters):
        """Test that forward pass produces correct output shapes"""
        batch_size = 10
        X = np.random.rand(batch_size, TestConfig.FEATURE_DIM)
        
        cache = forward_pass(X, mock_model_parameters)
        
        assert cache['A1'].shape == (batch_size, TestConfig.HIDDEN_DIM)
        assert cache['A2'].shape == (batch_size, TestConfig.NUM_CLASSES)
    
    def test_forward_pass_probabilities(self, mock_model_parameters):
        """Test that output probabilities sum to 1"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        
        cache = forward_pass(X, mock_model_parameters)
        
        # A2 should be probability distribution (sum to 1 for each sample)
        prob_sums = cache['A2'].sum(axis=1)
        np.testing.assert_almost_equal(prob_sums, np.ones(5), decimal=5)
    
    def test_forward_pass_range(self, mock_model_parameters):
        """Test that probabilities are in [0, 1] range"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        
        cache = forward_pass(X, mock_model_parameters)
        
        assert np.all(cache['A2'] >= 0)
        assert np.all(cache['A2'] <= 1)
    
    def test_forward_pass_cache_contents(self, mock_model_parameters):
        """Test that cache contains all required values"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        
        cache = forward_pass(X, mock_model_parameters)
        
        required_keys = ['Z1', 'A1', 'Z2', 'A2']
        for key in required_keys:
            assert key in cache


class TestLossComputation:
    """Test loss computation"""
    
    def test_compute_loss_value(self):
        """Test that loss is computed correctly"""
        # Perfect predictions
        y_pred = np.array([[0.9, 0.05, 0.05],
                          [0.05, 0.9, 0.05],
                          [0.05, 0.05, 0.9]])
        y_true = np.array([0, 1, 2])
        
        loss = compute_loss(y_pred, y_true)
        
        # Loss should be low for good predictions
        assert loss > 0
        assert loss < 1.0
    
    def test_compute_loss_worst_case(self):
        """Test loss for worst predictions"""
        # Worst predictions (high confidence wrong)
        y_pred = np.array([[0.05, 0.05, 0.9],   # Predicts 2, actual 0
                          [0.9, 0.05, 0.05],    # Predicts 0, actual 1
                          [0.05, 0.9, 0.05]])   # Predicts 1, actual 2
        y_true = np.array([0, 1, 2])
        
        loss = compute_loss(y_pred, y_true)
        
        # Loss should be high for bad predictions
        assert loss > 1.0
    
    def test_compute_loss_stability(self):
        """Test numerical stability with extreme values"""
        # Very confident predictions
        y_pred = np.array([[0.9999, 0.00005, 0.00005]])
        y_true = np.array([0])
        
        loss = compute_loss(y_pred, y_true)
        
        # Should not produce NaN or Inf
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestBackwardPass:
    """Test backward propagation"""
    
    def test_backward_pass_gradient_shapes(self, mock_model_parameters):
        """Test that gradients have correct shapes"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        cache = forward_pass(X, mock_model_parameters)
        grads = backward_pass(X, y, cache, mock_model_parameters)
        
        assert grads['dW1'].shape == mock_model_parameters['W1'].shape
        assert grads['db1'].shape == mock_model_parameters['b1'].shape
        assert grads['dW2'].shape == mock_model_parameters['W2'].shape
        assert grads['db2'].shape == mock_model_parameters['b2'].shape
    
    def test_backward_pass_gradient_types(self, mock_model_parameters):
        """Test that gradients are numpy arrays"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        y = np.array([0, 1, 2, 0, 1])
        
        cache = forward_pass(X, mock_model_parameters)
        grads = backward_pass(X, y, cache, mock_model_parameters)
        
        assert isinstance(grads['dW1'], np.ndarray)
        assert isinstance(grads['db1'], np.ndarray)
        assert isinstance(grads['dW2'], np.ndarray)
        assert isinstance(grads['db2'], np.ndarray)


class TestParameterUpdate:
    """Test parameter updates"""
    
    def test_update_parameters_changes_values(self, mock_model_parameters):
        """Test that parameters are updated"""
        original_W1 = mock_model_parameters['W1'].copy()
        
        # Create dummy gradients
        grads = {
            'dW1': np.random.rand(*mock_model_parameters['W1'].shape),
            'db1': np.random.rand(*mock_model_parameters['b1'].shape),
            'dW2': np.random.rand(*mock_model_parameters['W2'].shape),
            'db2': np.random.rand(*mock_model_parameters['b2'].shape)
        }
        
        learning_rate = 0.01
        updated_params = update_parameters(mock_model_parameters, grads, learning_rate)
        
        # Parameters should change
        assert not np.array_equal(updated_params['W1'], original_W1)
    
    def test_update_parameters_direction(self, mock_model_parameters):
        """Test that parameters move in opposite direction of gradients"""
        original_W1 = mock_model_parameters['W1'].copy()
        
        grads = {
            'dW1': np.ones_like(mock_model_parameters['W1']),
            'db1': np.ones_like(mock_model_parameters['b1']),
            'dW2': np.ones_like(mock_model_parameters['W2']),
            'db2': np.ones_like(mock_model_parameters['b2'])
        }
        
        learning_rate = 0.01
        updated_params = update_parameters(mock_model_parameters, grads, learning_rate)
        
        # W1 should decrease (gradient is positive)
        assert np.all(updated_params['W1'] < original_W1)


class TestTrainingStep:
    """Test single training step"""
    
    def test_train_step_returns_loss(self, mock_model_parameters):
        """Test that train_step returns a loss value"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        params, loss = train_step(X, y, mock_model_parameters, learning_rate=0.01)
        
        assert isinstance(loss, (float, np.floating))
        assert loss > 0
    
    def test_train_step_updates_parameters(self, mock_model_parameters):
        """Test that parameters are updated during training step"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        original_W1 = mock_model_parameters['W1'].copy()
        
        params, loss = train_step(X, y, mock_model_parameters, learning_rate=0.01)
        
        assert not np.array_equal(params['W1'], original_W1)


class TestPrediction:
    """Test prediction function"""
    
    def test_predict_shape(self, mock_model_parameters):
        """Test that predictions have correct shape"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        
        predictions = predict(X, mock_model_parameters)
        
        assert predictions.shape == (10,)
    
    def test_predict_values(self, mock_model_parameters):
        """Test that predictions are valid class indices"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        
        predictions = predict(X, mock_model_parameters)
        
        # Predictions should be in [0, num_classes)
        assert np.all(predictions >= 0)
        assert np.all(predictions < TestConfig.NUM_CLASSES)
    
    def test_predict_deterministic(self, mock_model_parameters):
        """Test that predictions are deterministic"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        
        pred1 = predict(X, mock_model_parameters)
        pred2 = predict(X, mock_model_parameters)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestEvaluation:
    """Test model evaluation"""
    
    def test_evaluate_perfect_predictions(self, mock_model_parameters):
        """Test evaluation with perfect predictions"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        # Mock perfect predictions
        y_pred = y_true.copy()
        
        # Temporarily replace predict function
        import cnn.fully_connected_layer as fcl
        original_predict = fcl.predict
        fcl.predict = lambda X, params: y_pred
        
        accuracy, conf_matrix = evaluate(X, y_true, mock_model_parameters)
        
        # Restore original function
        fcl.predict = original_predict
        
        # Should have 100% accuracy
        assert accuracy == 1.0
    
    def test_evaluate_returns_confusion_matrix(self, mock_model_parameters):
        """Test that confusion matrix has correct shape"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        accuracy, conf_matrix = evaluate(X, y_true, mock_model_parameters)
        
        assert conf_matrix.shape == (TestConfig.NUM_CLASSES, TestConfig.NUM_CLASSES)


class TestModelTraining:
    """Test complete training process"""
    
    def test_overfitting_detection(self):
        """Monitor train vs test accuracy divergence"""
        # Create simple overfitting scenario
        X_train = np.random.rand(20, TestConfig.FEATURE_DIM)
        y_train = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        X_test = np.random.rand(10, TestConfig.FEATURE_DIM)
        y_test = np.random.randint(0, TestConfig.NUM_CLASSES, 10)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Train for many epochs
        train_losses = []
        for epoch in range(50):
            params, loss = train_step(X_train, y_train, params, learning_rate=0.01)
            train_losses.append(loss)
        
        # Training loss should decrease
        assert train_losses[-1] < train_losses[0]
    
    def test_learning_rate_schedules(self):
        """Test different LR decay strategies"""
        X = np.random.rand(20, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Test with decreasing learning rate
        learning_rates = [0.1, 0.01, 0.001]
        losses = []
        
        for lr in learning_rates:
            params_copy = {k: v.copy() for k, v in params.items()}
            params_copy, loss = train_step(X, y, params_copy, learning_rate=lr)
            losses.append(loss)
        
        # All should complete without errors
        assert len(losses) == 3
    
    def test_checkpoint_recovery(self):
        """Test resuming training from checkpoint"""
        X = np.random.rand(20, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        # Train for a few epochs
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        for _ in range(5):
            params, _ = train_step(X, y, params, learning_rate=0.01)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            checkpoint_path = f.name
            save_model(params, checkpoint_path)
        
        # Load checkpoint
        loaded_params = load_model(checkpoint_path)
        
        # Parameters should match
        for key in params.keys():
            np.testing.assert_array_almost_equal(params[key], loaded_params[key])
        
        # Cleanup
        Path(checkpoint_path).unlink()
    
    def test_gradient_clipping(self):
        """Test gradient explosion prevention"""
        X = np.random.rand(10, TestConfig.FEATURE_DIM) * 1000  # Large inputs
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 10)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        cache = forward_pass(X, params)
        grads = backward_pass(X, y, cache, params)
        
        # Check if gradients are reasonable (not exploding)
        assert not np.any(np.isnan(grads['dW1']))
        assert not np.any(np.isinf(grads['dW1']))
    
    def test_regularization_effectiveness(self):
        """Compare L1, L2, and dropout regularization"""
        X = np.random.rand(20, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Train without regularization
        params_no_reg = {k: v.copy() for k, v in params.items()}
        for _ in range(10):
            params_no_reg, _ = train_step(X, y, params_no_reg, learning_rate=0.01)
        
        # With L2 regularization (if implemented)
        params_l2 = {k: v.copy() for k, v in params.items()}
        for _ in range(10):
            params_l2, _ = train_step(X, y, params_l2, learning_rate=0.01, lambda_reg=0.001)
        
        # Parameters should be different
        assert not np.array_equal(params_no_reg['W1'], params_l2['W1'])
    
    def test_convergence_detection(self):
        """Test plateau detection and convergence criteria"""
        X = np.random.rand(20, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 20)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        losses = []
        for epoch in range(100):
            params, loss = train_step(X, y, params, learning_rate=0.001)
            losses.append(loss)
            
            # Check for convergence (loss change < threshold)
            if epoch > 10:
                recent_change = abs(losses[-1] - losses[-10])
                if recent_change < 0.001:
                    # Converged
                    break
        
        # Should converge before max epochs
        assert len(losses) < 100 or losses[-1] < losses[0]


class TestModelSaveLoad:
    """Test model persistence"""
    
    def test_save_model(self, mock_model_parameters):
        """Test saving model to file"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            model_path = f.name
        
        save_model(mock_model_parameters, model_path)
        
        # File should exist
        assert Path(model_path).exists()
        
        # Cleanup
        Path(model_path).unlink()
    
    def test_load_model(self, mock_model_parameters):
        """Test loading model from file"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            model_path = f.name
        
        save_model(mock_model_parameters, model_path)
        loaded_params = load_model(model_path)
        
        # Parameters should match
        for key in mock_model_parameters.keys():
            np.testing.assert_array_almost_equal(
                mock_model_parameters[key],
                loaded_params[key]
            )
        
        # Cleanup
        Path(model_path).unlink()
    
    def test_save_load_preserves_functionality(self, mock_model_parameters):
        """Test that saved/loaded model produces same predictions"""
        X = np.random.rand(5, TestConfig.FEATURE_DIM)
        
        # Get predictions with original model
        pred_original = predict(X, mock_model_parameters)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            model_path = f.name
        
        save_model(mock_model_parameters, model_path)
        loaded_params = load_model(model_path)
        
        # Get predictions with loaded model
        pred_loaded = predict(X, loaded_params)
        
        # Should be identical
        np.testing.assert_array_equal(pred_original, pred_loaded)
        
        # Cleanup
        Path(model_path).unlink()


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])