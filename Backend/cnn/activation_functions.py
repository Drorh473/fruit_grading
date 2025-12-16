import numpy as np
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


def softmax(x):
    """
    Softmax activation function (numerically stable)
    Args:
        x: Input array (batch_size, num_classes)
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)