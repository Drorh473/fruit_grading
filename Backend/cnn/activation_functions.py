"""Neural network activation functions."""
import numpy as np
def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative."""
    return (x > 0).astype(float)

def softmax(x):
    """Numerically stable softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)