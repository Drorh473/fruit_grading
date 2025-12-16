import unittest
import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

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
from cnn.activation_functions import relu, softmax, relu_derivative

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class TestActivationFunctions(unittest.TestCase):
    """Test cases for activation functions"""
    
    def test_relu_positive_values(self):
        """Test ReLU with positive values"""
        x = np.array([1, 2, 3, 4, 5])
        result = relu(x)
        np.testing.assert_array_equal(result, x)
    
    def test_relu_negative_values(self):
        """Test ReLU zeros negative values"""
        x = np.array([-1, -2, -3, -4, -5])
        result = relu(x)
        np.testing.assert_array_equal(result, np.zeros_like(x))
    
    def test_relu_mixed_values(self):
        """Test ReLU with mixed positive and negative"""
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        result = relu(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative"""
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 1])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_softmax_output_range(self):
        """Test softmax outputs are in [0, 1] and sum to 1"""
        x = np.array([[1, 2, 3, 4, 5]])
        result = softmax(x)
        
        # Check range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
        # Check sum to 1
        np.testing.assert_almost_equal(result.sum(axis=1), 1.0)
    
    def test_softmax_numerical_stability(self):
        """Test softmax handles large values without overflow"""
        x = np.array([[1000, 2000, 3000]])
        result = softmax(x)
        
        # Should not have NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        
        # Should still sum to 1
        np.testing.assert_almost_equal(result.sum(axis=1), 1.0)


class TestParameterInitialization(unittest.TestCase):
    """Test cases for parameter initialization"""
    
    def test_initialize_parameters_shapes(self):
        """Test that initialized parameters have correct shapes"""
        input_dim = 1024
        hidden_dim = 256
        num_classes = 3
        
        params = initialize_parameters(input_dim, hidden_dim, num_classes)
        
        # Check shapes
        self.assertEqual(params['W1'].shape, (input_dim, hidden_dim))
        self.assertEqual(params['b1'].shape, (1, hidden_dim))
        self.assertEqual(params['W2'].shape, (hidden_dim, num_classes))
        self.assertEqual(params['b2'].shape, (1, num_classes))
    
    def test_initialize_parameters_bias_zeros(self):
        """Test that biases are initialized to zeros"""
        params = initialize_parameters(100, 50, 3)
        
        np.testing.assert_array_equal(params['b1'], np.zeros((1, 50)))
        np.testing.assert_array_equal(params['b2'], np.zeros((1, 3)))
    
    def test_initialize_parameters_weights_non_zero(self):
        """Test that weights are non-zero (He initialization)"""
        params = initialize_parameters(100, 50, 3)
        
        # Weights should not all be zero
        self.assertFalse(np.all(params['W1'] == 0))
        self.assertFalse(np.all(params['W2'] == 0))
    
    def test_initialize_parameters_reproducibility(self):
        """Test that initialization with same seed gives same results"""
        np.random.seed(42)
        params1 = initialize_parameters(100, 50, 3)
        
        np.random.seed(42)
        params2 = initialize_parameters(100, 50, 3)
        
        np.testing.assert_array_equal(params1['W1'], params2['W1'])
        np.testing.assert_array_equal(params1['W2'], params2['W2'])


class TestForwardPass(unittest.TestCase):
    """Test cases for forward pass"""
    
    def setUp(self):
        """Set up test parameters"""
        np.random.seed(42)
        self.params = initialize_parameters(10, 5, 3)
        self.X = np.random.rand(4, 10).astype(np.float32)
    
    def test_forward_pass_output_shape(self):
        """Test forward pass output has correct shape"""
        A2, cache = forward_pass(self.X, self.params)
        
        # Output should be (batch_size, num_classes)
        self.assertEqual(A2.shape, (4, 3))
    
    def test_forward_pass_output_probabilities(self):
        """Test forward pass outputs valid probabilities"""
        A2, cache = forward_pass(self.X, self.params)
        
        # All values should be in [0, 1]
        self.assertTrue(np.all(A2 >= 0))
        self.assertTrue(np.all(A2 <= 1))
        
        # Each row should sum to 1
        row_sums = A2.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))
    
    def test_forward_pass_cache_contents(self):
        """Test forward pass caches intermediate values"""
        A2, cache = forward_pass(self.X, self.params)
        
        # Cache should contain all intermediate values
        self.assertIn('X', cache)
        self.assertIn('Z1', cache)
        self.assertIn('A1', cache)
        self.assertIn('Z2', cache)
        self.assertIn('A2', cache)
        
        # Check shapes
        self.assertEqual(cache['X'].shape, (4, 10))
        self.assertEqual(cache['Z1'].shape, (4, 5))
        self.assertEqual(cache['A1'].shape, (4, 5))
        self.assertEqual(cache['Z2'].shape, (4, 3))
        self.assertEqual(cache['A2'].shape, (4, 3))


class TestLossComputation(unittest.TestCase):
    """Test cases for loss computation"""
    
    def test_compute_loss_perfect_prediction(self):
        """Test loss is near zero for perfect predictions"""
        y_pred = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
        y_true = np.array([2, 0, 1])
        
        loss = compute_loss(y_pred, y_true, 3)
        
        # Loss should be very small (numerical stability prevents exactly 0)
        self.assertLess(loss, 0.01)
    
    def test_compute_loss_worst_prediction(self):
        """Test loss is high for worst predictions"""
        y_pred = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        y_true = np.array([2, 0, 1])  # Opposite of predictions
        
        loss = compute_loss(y_pred, y_true, 3)
        
        # Loss should be large
        self.assertGreater(loss, 1.0)
    
    def test_compute_loss_with_onehot_labels(self):
        """Test loss computation with one-hot encoded labels"""
        y_pred = np.array([[0.1, 0.2, 0.7],
                          [0.8, 0.1, 0.1]])
        y_true_onehot = np.array([[0, 0, 1],
                                   [1, 0, 0]])
        
        loss = compute_loss(y_pred, y_true_onehot, 3)
        
        # Should compute valid loss
        self.assertGreater(loss, 0)
        self.assertFalse(np.isnan(loss))


class TestBackwardPass(unittest.TestCase):
    """Test cases for backward pass"""
    
    def setUp(self):
        """Set up test parameters and cache"""
        np.random.seed(42)
        self.params = initialize_parameters(10, 5, 3)
        self.X = np.random.rand(4, 10).astype(np.float32)
        self.y = np.array([0, 1, 2, 1])
        _, self.cache = forward_pass(self.X, self.params)
    
    def test_backward_pass_gradient_shapes(self):
        """Test backward pass produces gradients with correct shapes"""
        grads = backward_pass(self.y, self.params, self.cache, 3)
        
        # Gradients should match parameter shapes
        self.assertEqual(grads['dW1'].shape, self.params['W1'].shape)
        self.assertEqual(grads['db1'].shape, self.params['b1'].shape)
        self.assertEqual(grads['dW2'].shape, self.params['W2'].shape)
        self.assertEqual(grads['db2'].shape, self.params['b2'].shape)
    
    def test_backward_pass_gradient_finite(self):
        """Test that gradients are finite (no NaN or Inf)"""
        grads = backward_pass(self.y, self.params, self.cache, 3)
        
        for key in grads:
            self.assertFalse(np.any(np.isnan(grads[key])))
            self.assertFalse(np.any(np.isinf(grads[key])))


class TestUpdateParameters(unittest.TestCase):
    """Test cases for parameter updates"""
    
    def setUp(self):
        """Set up test parameters"""
        np.random.seed(42)
        self.params = initialize_parameters(10, 5, 3)
        self.original_params = {k: v.copy() for k, v in self.params.items()}
        
        # Create mock gradients
        self.grads = {
            'dW1': np.ones_like(self.params['W1']),
            'db1': np.ones_like(self.params['b1']),
            'dW2': np.ones_like(self.params['W2']),
            'db2': np.ones_like(self.params['b2'])
        }
    
    def test_update_parameters_modifies_params(self):
        """Test that update_parameters changes parameter values"""
        updated_params = update_parameters(self.params, self.grads, 0.01)
        
        # Parameters should be different after update
        self.assertFalse(np.array_equal(updated_params['W1'], self.original_params['W1']))
        self.assertFalse(np.array_equal(updated_params['W2'], self.original_params['W2']))
    
    def test_update_parameters_gradient_descent(self):
        """Test that parameters move in opposite direction of gradients"""
        learning_rate = 0.1
        updated_params = update_parameters(self.params, self.grads, learning_rate)
        
        # W1 should decrease by learning_rate * gradient (all ones)
        expected_W1 = self.original_params['W1'] - learning_rate * self.grads['dW1']
        np.testing.assert_array_almost_equal(updated_params['W1'], expected_W1)
    
    def test_update_parameters_zero_learning_rate(self):
        """Test that zero learning rate doesn't change parameters"""
        updated_params = update_parameters(self.params, self.grads, 0.0)
        
        np.testing.assert_array_equal(updated_params['W1'], self.original_params['W1'])
        np.testing.assert_array_equal(updated_params['W2'], self.original_params['W2'])


class TestTrainStep(unittest.TestCase):
    """Test cases for single training step"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.rand(10, 50).astype(np.float32)
        self.y = np.random.randint(0, 3, size=10)
        self.params = initialize_parameters(50, 25, 3)
    
    def test_train_step_returns_valid_metrics(self):
        """Test that train_step returns valid loss and accuracy"""
        loss, accuracy, updated_params = train_step(
            self.X, self.y, self.params, 3, 0.01
        )
        
        # Loss should be positive
        self.assertGreater(loss, 0)
        self.assertFalse(np.isnan(loss))
        
        # Accuracy should be in [0, 1]
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_train_step_updates_parameters(self):
        """Test that train_step modifies parameters"""
        original_W1 = self.params['W1'].copy()
        
        _, _, updated_params = train_step(
            self.X, self.y, self.params, 3, 0.01
        )
        
        # Parameters should be updated
        self.assertFalse(np.array_equal(updated_params['W1'], original_W1))


class TestPrediction(unittest.TestCase):
    """Test cases for prediction"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.rand(5, 20).astype(np.float32)
        self.params = initialize_parameters(20, 10, 3)
    
    def test_predict_output_shapes(self):
        """Test predict returns correct shapes"""
        predictions, probabilities = predict(self.X, self.params)
        
        self.assertEqual(predictions.shape, (5,))
        self.assertEqual(probabilities.shape, (5, 3))
    
    def test_predict_valid_class_indices(self):
        """Test predictions are valid class indices"""
        predictions, _ = predict(self.X, self.params)
        
        # All predictions should be in [0, 1, 2]
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < 3))
    
    def test_predict_probabilities_sum_to_one(self):
        """Test predicted probabilities sum to 1"""
        _, probabilities = predict(self.X, self.params)
        
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5))


class TestEvaluate(unittest.TestCase):
    """Test cases for model evaluation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.rand(20, 30).astype(np.float32)
        self.y = np.random.randint(0, 3, size=20)
        self.params = initialize_parameters(30, 15, 3)
    
    def test_evaluate_returns_metrics(self):
        """Test evaluate returns loss and accuracy"""
        loss, accuracy = evaluate(self.X, self.y, self.params, 3)
        
        self.assertIsInstance(loss, (float, np.floating))
        self.assertIsInstance(accuracy, (float, np.floating))
        
        # Check valid ranges
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)


class TestTraining(unittest.TestCase):
    """Test cases for training loop"""
    
    def test_train_simple_dataset(self):
        """Test training on a simple dataset"""
        np.random.seed(42)
        
        # Create simple linearly separable dataset
        X_train = np.vstack([
            np.random.randn(10, 5) + [1, 1, 0, 0, 0],  # Class 0
            np.random.randn(10, 5) + [0, 0, 1, 1, 0],  # Class 1
            np.random.randn(10, 5) + [0, 0, 0, 0, 1],  # Class 2
        ]).astype(np.float32)
        y_train = np.array([0]*10 + [1]*10 + [2]*10)
        
        X_val = np.vstack([
            np.random.randn(5, 5) + [1, 1, 0, 0, 0],
            np.random.randn(5, 5) + [0, 0, 1, 1, 0],
            np.random.randn(5, 5) + [0, 0, 0, 0, 1],
        ]).astype(np.float32)
        y_val = np.array([0]*5 + [1]*5 + [2]*5)
        
        # Train
        params, history = train(
            X_train, y_train,
            X_val, y_val,
            input_dim=5,
            hidden_dim=10,
            num_classes=3,
            epochs=20,
            batch_size=10,
            learning_rate=0.01,
            verbose=False
        )
        
        # Check that training improved
        self.assertLess(history['train_loss'][-1], history['train_loss'][0])
        
        # Check history structure
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_accuracy', history)
        
        self.assertEqual(len(history['train_loss']), 20)


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for saving and loading models"""
    
    def setUp(self):
        """Set up test parameters"""
        np.random.seed(42)
        self.params = initialize_parameters(100, 50, 3)
        self.history = {
            'train_loss': [1.0, 0.8, 0.6],
            'train_accuracy': [0.5, 0.6, 0.7],
            'val_loss': [1.1, 0.9, 0.7],
            'val_accuracy': [0.4, 0.5, 0.6]
        }
        self.input_dim = 100
        self.hidden_dim = 50
        self.num_classes = 3
    
    def test_save_and_load_model(self):
        """Test saving and loading preserves model parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pkl')
            
            # Save model
            save_model(
                self.params, self.history,
                self.input_dim, self.hidden_dim, self.num_classes,
                filepath
            )
            
            # Check file exists
            self.assertTrue(os.path.exists(filepath))
            
            # Load model
            loaded_params, model_info = load_model(filepath)
            
            # Check parameters match
            np.testing.assert_array_equal(loaded_params['W1'], self.params['W1'])
            np.testing.assert_array_equal(loaded_params['W2'], self.params['W2'])
            
            # Check metadata
            self.assertEqual(model_info['input_dim'], self.input_dim)
            self.assertEqual(model_info['hidden_dim'], self.hidden_dim)
            self.assertEqual(model_info['num_classes'], self.num_classes)


class TestTrainingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_train_with_single_batch(self):
        """Test training with batch size equal to dataset size"""
        np.random.seed(42)
        X_train = np.random.rand(10, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, size=10)
        
        params, history = train(
            X_train, y_train,
            None, None,
            input_dim=5,
            hidden_dim=5,
            num_classes=3,
            epochs=5,
            batch_size=10,
            learning_rate=0.01,
            verbose=False
        )
        
        # Should complete without errors
        self.assertIsNotNone(params)
        self.assertEqual(len(history['train_loss']), 5)
    
    def test_train_without_validation(self):
        """Test training without validation data"""
        np.random.seed(42)
        X_train = np.random.rand(20, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, size=20)
        
        params, history = train(
            X_train, y_train,
            None, None,
            input_dim=5,
            hidden_dim=5,
            num_classes=3,
            epochs=5,
            batch_size=5,
            learning_rate=0.01,
            verbose=False
        )
        
        # Should have training metrics but no validation metrics
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        # Val metrics will be empty lists or not meaningful
        self.assertEqual(len(history['val_loss']), 0)


if __name__ == '__main__':
    unittest.main()