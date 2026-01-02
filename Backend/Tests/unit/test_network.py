"""
Enhanced Model Training Testing
Comprehensive tests for neural network operations and training
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path


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
    save_model,
    load_model
)


# ==================== Fixtures ====================

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'FEATURE_DIM': 50176,  # 7 * 7 * 1024 (ShuffleNet output)
        'HIDDEN_DIM': 256,
        'NUM_CLASSES': 3
    }


@pytest.fixture
def mock_model_parameters(test_config):
    """Create mock model parameters with correct dimensions"""
    return initialize_parameters(
        test_config['FEATURE_DIM'],
        test_config['HIDDEN_DIM'],
        test_config['NUM_CLASSES']
    )


# ==================== Tests ====================

class TestParameterInitialization:
    """Test parameter initialization"""
    
    def test_initialize_parameters_shapes(self, test_config):
        """Test that initialized parameters have correct shapes"""
        params = initialize_parameters(
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES']
        )
        
        # Check shapes - FIXED: biases are (1, dim) not (dim,)
        assert params['W1'].shape == (test_config['FEATURE_DIM'], test_config['HIDDEN_DIM'])
        assert params['b1'].shape == (1, test_config['HIDDEN_DIM'])  # FIXED
        assert params['W2'].shape == (test_config['HIDDEN_DIM'], test_config['NUM_CLASSES'])
        assert params['b2'].shape == (1, test_config['NUM_CLASSES'])  # FIXED
    
    def test_initialize_parameters_types(self, test_config):
        """Test that parameters are numpy arrays"""
        params = initialize_parameters(
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES']
        )
        
        assert isinstance(params['W1'], np.ndarray)
        assert isinstance(params['b1'], np.ndarray)
        assert isinstance(params['W2'], np.ndarray)
        assert isinstance(params['b2'], np.ndarray)
    
    def test_initialize_parameters_values(self, test_config):
        """Test that parameters are initialized with reasonable values"""
        params = initialize_parameters(
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES']
        )
        
        # Weights should be small random values
        assert np.abs(params['W1']).mean() < 1.0
        assert np.abs(params['W2']).mean() < 1.0
        
        # Biases should be zeros - FIXED: shape is (1, dim)
        np.testing.assert_array_equal(params['b1'], np.zeros((1, test_config['HIDDEN_DIM'])))
        np.testing.assert_array_equal(params['b2'], np.zeros((1, test_config['NUM_CLASSES'])))


class TestForwardPass:
    """Test forward propagation"""
    
    def test_forward_pass_shapes(self, mock_model_parameters, test_config):
        """Test that forward pass produces correct output shapes"""
        batch_size = 10
        X = np.random.rand(batch_size, test_config['FEATURE_DIM'])
        
        # FIXED: forward_pass returns (A2, cache)
        A2, cache = forward_pass(X, mock_model_parameters)
        
        assert cache['A1'].shape == (batch_size, test_config['HIDDEN_DIM'])
        assert A2.shape == (batch_size, test_config['NUM_CLASSES'])
    
    def test_forward_pass_probabilities(self, mock_model_parameters, test_config):
        """Test that output probabilities sum to 1"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        
        A2, cache = forward_pass(X, mock_model_parameters)
        
        # A2 should be probability distribution (sum to 1 for each sample)
        prob_sums = A2.sum(axis=1)
        np.testing.assert_almost_equal(prob_sums, np.ones(5), decimal=5)
    
    def test_forward_pass_range(self, mock_model_parameters, test_config):
        """Test that probabilities are in [0, 1] range"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        
        A2, cache = forward_pass(X, mock_model_parameters)
        
        assert np.all(A2 >= 0)
        assert np.all(A2 <= 1)
    
    def test_forward_pass_cache_contents(self, mock_model_parameters, test_config):
        """Test that cache contains all required values"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        
        A2, cache = forward_pass(X, mock_model_parameters)
        
        required_keys = ['X', 'Z1', 'A1', 'Z2', 'A2']
        for key in required_keys:
            assert key in cache


class TestLossComputation:
    """Test loss computation"""
    
    def test_compute_loss_value(self, test_config):
        """Test that loss is computed correctly"""
        # Perfect predictions
        y_pred = np.array([[0.9, 0.05, 0.05],
                          [0.05, 0.9, 0.05],
                          [0.05, 0.05, 0.9]])
        y_true = np.array([0, 1, 2])
        
        loss = compute_loss(y_pred, y_true, num_classes=test_config['NUM_CLASSES'])
        
        # Loss should be low for good predictions
        assert loss > 0
        assert loss < 1.0
    
    def test_compute_loss_worst_case(self, test_config):
        """Test loss for worst predictions"""
        y_pred = np.array([[0.05, 0.05, 0.9],
                          [0.9, 0.05, 0.05],
                          [0.05, 0.9, 0.05]])
        y_true = np.array([0, 1, 2])
        
        loss = compute_loss(y_pred, y_true, num_classes=test_config['NUM_CLASSES'])
        
        # Loss should be high for bad predictions
        assert loss > 1.0
    
    def test_compute_loss_stability(self, test_config):
        """Test numerical stability with extreme values"""
        y_pred = np.array([[0.9999, 0.00005, 0.00005]])
        y_true = np.array([0])
        
        loss = compute_loss(y_pred, y_true, num_classes=test_config['NUM_CLASSES'])
        
        # Should not produce NaN or Inf
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestBackwardPass:
    """Test backward propagation"""
    
    def test_backward_pass_gradient_shapes(self, mock_model_parameters, test_config):
        """Test that gradients have correct shapes"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        A2, cache = forward_pass(X, mock_model_parameters)
        # FIXED: backward_pass signature is (y_true, params, cache, num_classes, lambda_reg)
        grads = backward_pass(y, mock_model_parameters, cache, num_classes=test_config['NUM_CLASSES'])
        
        assert grads['dW1'].shape == mock_model_parameters['W1'].shape
        assert grads['db1'].shape == mock_model_parameters['b1'].shape
        assert grads['dW2'].shape == mock_model_parameters['W2'].shape
        assert grads['db2'].shape == mock_model_parameters['b2'].shape
    
    def test_backward_pass_gradient_types(self, mock_model_parameters, test_config):
        """Test that gradients are numpy arrays"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        y = np.array([0, 1, 2, 0, 1])
        
        A2, cache = forward_pass(X, mock_model_parameters)
        grads = backward_pass(y, mock_model_parameters, cache, num_classes=test_config['NUM_CLASSES'])
        
        assert isinstance(grads['dW1'], np.ndarray)
        assert isinstance(grads['db1'], np.ndarray)
        assert isinstance(grads['dW2'], np.ndarray)
        assert isinstance(grads['db2'], np.ndarray)


class TestParameterUpdate:
    """Test parameter updates"""
    
    def test_update_parameters_changes_values(self, mock_model_parameters):
        """Test that parameters are updated"""
        original_W1 = mock_model_parameters['W1'].copy()
        
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
    
    def test_train_step_returns_loss(self, mock_model_parameters, test_config):
        """Test that train_step returns correct values"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        loss, accuracy, params = train_step(X, y, mock_model_parameters, 3, learning_rate=0.01)
        
        assert isinstance(loss, (float, np.floating))
        assert loss > 0
        assert 0 <= accuracy <= 1
    
    def test_train_step_updates_parameters(self, mock_model_parameters, test_config):
        """Test that parameters are updated during training step"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        original_W1 = mock_model_parameters['W1'].copy()
        
        # FIXED: Unpack 3 values
        loss, accuracy, params = train_step(X, y, mock_model_parameters, 3, learning_rate=0.01)
        
        assert not np.array_equal(params['W1'], original_W1)


class TestPrediction:
    """Test prediction function"""
    
    def test_predict_shape(self, mock_model_parameters, test_config):
        """Test that predictions have correct shape"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        
        # FIXED: predict returns (predictions, probabilities)
        predictions, probabilities = predict(X, mock_model_parameters)
        
        assert predictions.shape == (10,)
        assert probabilities.shape == (10, test_config['NUM_CLASSES'])
    
    def test_predict_values(self, mock_model_parameters, test_config):
        """Test that predictions are valid class indices"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        
        predictions, _ = predict(X, mock_model_parameters)
        
        # Predictions should be in [0, num_classes)
        assert np.all(predictions >= 0)
        assert np.all(predictions < test_config['NUM_CLASSES'])
    
    def test_predict_deterministic(self, mock_model_parameters, test_config):
        """Test that predictions are deterministic"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        
        pred1, _ = predict(X, mock_model_parameters)
        pred2, _ = predict(X, mock_model_parameters)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestEvaluation:
    """Test model evaluation"""
    
    def test_evaluate_returns_correct_values(self, mock_model_parameters, test_config):
        """Test evaluate function returns loss and accuracy"""
        X = np.random.rand(10, test_config['FEATURE_DIM'])
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        # FIXED: evaluate returns (loss, accuracy)
        loss, accuracy = evaluate(X, y, mock_model_parameters, num_classes=test_config['NUM_CLASSES'])
        
        assert isinstance(loss, (float, np.floating))
        assert 0 <= accuracy <= 1


class TestModelTraining:
    """Test complete training process"""
    
    def test_training_convergence(self, test_config):
        """Test that training reduces loss"""
        X = np.random.rand(20, test_config['FEATURE_DIM'])
        y = np.random.randint(0, test_config['NUM_CLASSES'], 20)
        
        params = initialize_parameters(
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES']
        )
        
        train_losses = []
        for epoch in range(20):
            # FIXED: Unpack 3 values
            loss, accuracy, params = train_step(X, y, params, 3, learning_rate=0.01)
            train_losses.append(loss)
        
        # Training loss should generally decrease
        assert train_losses[-1] < train_losses[0]
    
    def test_regularization_effectiveness(self, test_config):
        """Test L2 regularization"""
        X = np.random.rand(20, test_config['FEATURE_DIM'])
        y = np.random.randint(0, test_config['NUM_CLASSES'], 20)
        
        params = initialize_parameters(
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES']
        )
        
        # Train without regularization
        params_no_reg = {k: v.copy() for k, v in params.items()}
        for _ in range(10):
            loss, acc, params_no_reg = train_step(X, y, params_no_reg, 3, learning_rate=0.01, lambda_reg=0.0)
        
        # Train with L2 regularization
        params_l2 = {k: v.copy() for k, v in params.items()}
        for _ in range(10):
            loss, acc, params_l2 = train_step(X, y, params_l2, 3, learning_rate=0.01, lambda_reg=0.001)
                                             
        # Parameters should be different
        assert not np.array_equal(params_no_reg['W1'], params_l2['W1'])


class TestModelSaveLoad:
    """Test model persistence"""
    
    def test_save_model(self, mock_model_parameters, test_config):
        """Test saving model to file"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        # FIXED: save_model needs (params, history, input_dim, hidden_dim, num_classes, filepath)
        history = {'train_loss': [0.5, 0.4], 'train_accuracy': [0.7, 0.8]}
        save_model(
            mock_model_parameters,
            history,
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES'],
            model_path
        )
        
        # File should exist
        assert Path(model_path).exists()
        
        # Cleanup
        Path(model_path).unlink()
    
    def test_load_model(self, mock_model_parameters, test_config):
        """Test loading model from file"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        history = {'train_loss': [0.5, 0.4], 'train_accuracy': [0.7, 0.8]}
        save_model(
            mock_model_parameters,
            history,
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES'],
            model_path
        )
        
        # FIXED: load_model returns (params, model_info)
        loaded_params, model_info = load_model(model_path)
        
        # Parameters should match
        for key in mock_model_parameters.keys():
            np.testing.assert_array_almost_equal(
                mock_model_parameters[key],
                loaded_params[key]
            )
        
        # Model info should match
        assert model_info['input_dim'] == test_config['FEATURE_DIM']
        assert model_info['hidden_dim'] == test_config['HIDDEN_DIM']
        assert model_info['num_classes'] == test_config['NUM_CLASSES']
        
        # Cleanup
        Path(model_path).unlink()
    
    def test_save_load_preserves_functionality(self, mock_model_parameters, test_config):
        """Test that saved/loaded model produces same predictions"""
        X = np.random.rand(5, test_config['FEATURE_DIM'])
        
        # Get predictions with original model
        pred_original, _ = predict(X, mock_model_parameters)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        history = {'train_loss': [], 'train_accuracy': []}
        save_model(
            mock_model_parameters,
            history,
            test_config['FEATURE_DIM'],
            test_config['HIDDEN_DIM'],
            test_config['NUM_CLASSES'],
            model_path
        )
        
        loaded_params, _ = load_model(model_path)
        
        # Get predictions with loaded model
        pred_loaded, _ = predict(X, loaded_params)
        
        # Should be identical
        np.testing.assert_array_equal(pred_original, pred_loaded)
        
        # Cleanup
        Path(model_path).unlink()


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
