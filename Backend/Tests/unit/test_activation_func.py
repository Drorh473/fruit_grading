
import pytest
import numpy as np
from pathlib import Path

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.activation_functions import relu, softmax, relu_derivative


class TestReLU:
    """Test ReLU activation function"""
    
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
    
    def test_relu_preserves_shape(self):
        """Test that ReLU preserves input shape"""
        shapes = [(5,), (3, 4), (2, 3, 4)]
        
        for shape in shapes:
            x = np.random.randn(*shape)
            result = relu(x)
            assert result.shape == x.shape


class TestReLUDerivative:
    """Test ReLU derivative"""
    
    def test_relu_derivative_positive(self):
        """Test derivative for positive values"""
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array([1, 1, 1, 1, 1])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative_negative(self):
        """Test derivative for negative values"""
        x = np.array([-1, -2, -3, -4, -5])
        expected = np.array([0, 0, 0, 0, 0])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative_mixed(self):
        """Test derivative with mixed values"""
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 1])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected)
    def test_relu_derivative_preserves_shape(self):
        """Test that derivative preserves input shape"""
        shapes = [(5,), (3, 4), (2, 3, 4)]
        
        for shape in shapes:
            x = np.random.randn(*shape)
            result = relu_derivative(x)
            assert result.shape == x.shape


class TestSoftmax:
    """Test Softmax activation function"""
    
    def test_softmax_output_range(self):
        """Test softmax outputs are in [0, 1] and sum to 1"""
        x = np.array([[1, 2, 3, 4, 5]])
        result = softmax(x)
        
        # Check range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        
        # Check sum to 1
        np.testing.assert_almost_equal(result.sum(axis=1), 1.0)
    
    def test_softmax_numerical_stability(self):
        """Test softmax handles large values without overflow"""
        x = np.array([[1000, 2000, 3000]])
        result = softmax(x)
        
        # Should not have NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Should still sum to 1
        np.testing.assert_almost_equal(result.sum(axis=1), 1.0)
    
    def test_softmax_negative_values(self):
        """Test softmax with negative values"""
        x = np.array([[-1, -2, -3]])
        result = softmax(x)
        
        # Check range and sum
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        np.testing.assert_almost_equal(result.sum(axis=1), 1.0)
    
    def test_softmax_equal_values(self):
        """Test softmax with equal values produces uniform distribution"""
        x = np.array([[5, 5, 5]])
        result = softmax(x)
        
        # All probabilities should be equal (1/3)
        expected = np.array([[1/3, 1/3, 1/3]])
        np.testing.assert_almost_equal(result, expected)
    
    def test_softmax_max_element(self):
        """Test that max element gets highest probability"""
        x = np.array([[1, 5, 2]])
        result = softmax(x)
        
        # Index 1 should have highest probability
        assert np.argmax(result[0]) == 1
    
    def test_softmax_preserves_batch_size(self):
        """Test that softmax preserves batch dimension"""
        batch_sizes = [1, 5, 10, 50]
        num_classes = 3
        
        for batch_size in batch_sizes:
            x = np.random.randn(batch_size, num_classes)
            result = softmax(x)
            assert result.shape == (batch_size, num_classes)
    
    def test_softmax_zero_input(self):
        """Test softmax with zero input"""
        x = np.array([[0, 0, 0]])
        result = softmax(x)
        
        # Should produce uniform distribution
        expected = np.array([[1/3, 1/3, 1/3]])
        np.testing.assert_almost_equal(result, expected)

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_relu_very_large_values(self):
        """Test ReLU with very large values"""
        x = np.array([1e10, -1e10])
        result = relu(x)
        
        assert result[0] == 1e10
        assert result[1] == 0
    
    def test_softmax_extreme_differences(self):
        """Test softmax with extreme value differences"""
        x = np.array([[0, 1000]])
        result = softmax(x)
        
        # Second element should be ~1, first ~0
        assert result[0, 1] > 0.99
        assert result[0, 0] < 0.01
    
    def test_relu_empty_array(self):
        """Test ReLU with empty array"""
        x = np.array([])
        result = relu(x)
        
        assert result.shape == (0,)
    
    def test_single_value_softmax(self):
        """Test softmax with single class"""
        x = np.array([[5]])
        result = softmax(x)
        
        # Should be 1.0
        np.testing.assert_almost_equal(result, np.array([[1.0]]))
if __name__ == "__main__":
    pytest.main([__file__, "-v"])