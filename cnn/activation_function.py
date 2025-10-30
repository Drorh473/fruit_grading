import numpy as np


def relu(feature_vectors):
    """
    Apply ReLU (Rectified Linear Unit) activation to feature vectors.
    Args:
        feature_vectors: Dictionary mapping object IDs to feature arrays
                        Format: {object_id: np.array([...])}
    
    Returns:
        Dictionary with ReLU applied to all feature vectors
    """
    if not isinstance(feature_vectors, dict):
        raise TypeError("Input must be a dictionary of feature vectors")
    
    relu_vectors = {}
    
    for obj_id, feature_array in feature_vectors.items():
        if not isinstance(feature_array, np.ndarray):
            # Try to convert to numpy array
            feature_array = np.array(feature_array)
        
        # Apply ReLU: max(0, x)
        relu_vectors[obj_id] = np.maximum(0, feature_array)
         
    return relu_vectors


def softmax(feature_vectors):
    """
    Apply Softmax activation to feature vectors.
    Args:
        feature_vectors: Dictionary mapping object IDs to feature arrays
                        Format: {object_id: np.array([...])}
    
    Returns:
        Dictionary with Softmax applied to all feature vectors
    """
    if not isinstance(feature_vectors, dict):
        raise TypeError("Input must be a dictionary of feature vectors")
    
    softmax_vectors = {}
    
    for obj_id, feature_array in feature_vectors.items():
        if not isinstance(feature_array, np.ndarray):
            # Try to convert to numpy array
            feature_array = np.array(feature_array)
        
        # Numerical stability: subtract max value
        # This prevents overflow when computing exp() of large numbers
        shifted_array = feature_array - np.max(feature_array)
        
        # Compute exp
        exp_array = np.exp(shifted_array)
        
        # Normalize by sum to get probabilities
        sum_exp = np.sum(exp_array)
        
        # Handle edge case where sum is 0 (shouldn't happen with exp, but be safe)
        if sum_exp == 0:
            print(f"Warning: Sum of exponentials is 0 for {obj_id}, using uniform distribution")
            softmax_vectors[obj_id] = np.ones_like(feature_array) / len(feature_array)
        else:
            softmax_vectors[obj_id] = exp_array / sum_exp
    
    # Verify that all vectors sum to 1.0 (within numerical precision)
    for obj_id, vector in softmax_vectors.items():
        vector_sum = np.sum(vector)
        if not np.isclose(vector_sum, 1.0, rtol=1e-5):
            print(f"Warning: Softmax vector {obj_id} sums to {vector_sum}, not 1.0")
    
    return softmax_vectors