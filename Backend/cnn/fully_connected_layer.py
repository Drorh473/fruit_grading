import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
from cnn.activation_functions import relu, softmax, relu_derivative
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')

# Weight Initialization
def initialize_parameters(input_dim, hidden_dim, num_classes):
    """
    Initialize network parameters with He initialization
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of output classes
    
    Returns:
        Dictionary containing W1, b1, W2, b2
    """
    params = {
        'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim),
        'b2': np.zeros((1, num_classes))
    }
    return params


# Forward Pass
def forward_pass(X, params):
    """
    Forward pass through the network
    Architecture: Input -> FC1 -> ReLU -> FC2 -> Softmax -> Output
    
    Args:
        X: Input features (batch_size, input_dim)
        params: Dictionary with network parameters (W1, b1, W2, b2)
    
    Returns:
        A2: Output probabilities (batch_size, num_classes)
        cache: Dictionary with intermediate values for backprop
    """
    # First layer: Linear -> ReLU
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    
    # Second layer: Linear -> Softmax
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = softmax(Z2)
    
    # Cache for backpropagation
    cache = {
        'X': X,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }
    
    return A2, cache


# Loss Computation
def compute_loss(y_pred, y_true, num_classes, params=None, lambda_reg=0.0):
    """
    Compute cross-entropy loss with optional L2 regularization
    
    Args:
        y_pred: Predicted probabilities (batch_size, num_classes)
        y_true: True labels (batch_size,) or one-hot (batch_size, num_classes)
        num_classes: Number of classes
        params: Dictionary with network parameters (needed for L2 regularization)
        lambda_reg: L2 regularization strength (0 = no regularization)
    
    Returns:
        Loss value (cross-entropy + L2 penalty)
    """
    batch_size = y_pred.shape[0]
    
    # Convert to one-hot if needed
    if y_true.ndim == 1:
        y_true_onehot = np.zeros((batch_size, num_classes))
        y_true_onehot[np.arange(batch_size), y_true] = 1
    else:
        y_true_onehot = y_true
    
    # Cross-entropy loss (with numerical stability)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    ce_loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / batch_size
    
    # Add L2 regularization if params provided and lambda_reg > 0
    if params is not None and lambda_reg > 0:
        l2_penalty = lambda_reg * (np.sum(params['W1']**2) + np.sum(params['W2']**2))
        total_loss = ce_loss + l2_penalty
        return total_loss
    
    return ce_loss


# Backward Pass
def backward_pass(y_true, params, cache, num_classes, lambda_reg=0.0):
    """
    Backward pass (backpropagation) with optional L2 regularization
    
    Args:
        y_true: True labels (batch_size,) or one-hot (batch_size, num_classes)
        params: Dictionary with network parameters
        cache: Dictionary with cached values from forward pass
        num_classes: Number of classes
        lambda_reg: L2 regularization strength (0 = no regularization)
    
    Returns:
        grads: Dictionary with gradients (dW1, db1, dW2, db2)
    """
    batch_size = cache['X'].shape[0]
    
    # Convert to one-hot if needed
    if y_true.ndim == 1:
        y_true_onehot = np.zeros((batch_size, num_classes))
        y_true_onehot[np.arange(batch_size), y_true] = 1
    else:
        y_true_onehot = y_true
    
    # Gradient of loss w.r.t Z2 (softmax + cross-entropy)
    dZ2 = cache['A2'] - y_true_onehot
    
    # Gradients for W2 and b2 (with L2 regularization on W2)
    dW2 = np.dot(cache['A1'].T, dZ2) / batch_size
    if lambda_reg > 0:
        dW2 += 2 * lambda_reg * params['W2']  # L2 gradient
    db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
    
    # Gradient w.r.t A1
    dA1 = np.dot(dZ2, params['W2'].T)
    
    # Gradient w.r.t Z1 (through ReLU)
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    
    # Gradients for W1 and b1 (with L2 regularization on W1)
    dW1 = np.dot(cache['X'].T, dZ1) / batch_size
    if lambda_reg > 0:
        dW1 += 2 * lambda_reg * params['W1']  # L2 gradient
    db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    
    return grads

# Parameter Update
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Args:
        params: Dictionary with current parameters
        grads: Dictionary with gradients
        learning_rate: Learning rate for gradient descent
    
    Returns:
        Updated params dictionary
    """
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    
    return params


# Training Step
def train_step(X, y, params, num_classes, learning_rate, lambda_reg=0.0):
    """
    Single training step (forward + backward + update) with L2 regularization
    
    Args:
        X: Input features (batch_size, input_dim)
        y: True labels (batch_size,)
        params: Dictionary with network parameters
        num_classes: Number of classes
        learning_rate: Learning rate
        lambda_reg: L2 regularization strength
    
    Returns:
        loss: Loss value
        accuracy: Accuracy value
        params: Updated parameters
    """
    # Forward pass
    y_pred, cache = forward_pass(X, params)
    
    # Compute loss (with L2 regularization)
    loss = compute_loss(y_pred, y, num_classes, params, lambda_reg)
    
    # Compute accuracy
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y)
    
    # Backward pass (with L2 regularization)
    grads = backward_pass(y, params, cache, num_classes, lambda_reg)
    
    # Update parameters
    params = update_parameters(params, grads, learning_rate)
    
    return loss, accuracy, params


# Prediction
def predict(X, params):
    """
    Make predictions
    
    Args:
        X: Input features (batch_size, input_dim)
        params: Dictionary with network parameters
    
    Returns:
        predictions: Predicted class indices (batch_size,)
        probabilities: Class probabilities (batch_size, num_classes)
    """
    probabilities, _ = forward_pass(X, params)
    predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities


# Evaluation
def evaluate(X, y, params, num_classes, lambda_reg=0.0):
    """
    Evaluate the model with optional L2 regularization
    
    Args:
        X: Input features
        y: True labels
        params: Dictionary with network parameters
        num_classes: Number of classes
        lambda_reg: L2 regularization strength
    
    Returns:
        loss: Loss value
        accuracy: Accuracy value
    """
    y_pred, _ = forward_pass(X, params)
    loss = compute_loss(y_pred, y, num_classes, params, lambda_reg)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y)
    
    return loss, accuracy


# Training Loop (works with pre-batched data from generator)
def train_from_generator(train_generator, val_generator, input_dim, hidden_dim, num_classes,
                        epochs=100, learning_rate=0.001, verbose=True):
    """
    Train the neural network using a batch generator
    
    Args:
        train_generator: Generator that yields (batch_x, batch_y) tuples
        val_generator: Validation generator (optional, can be None)
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of classes
        epochs: Number of training epochs
        learning_rate: Learning rate
        verbose: Print training progress
    
    Returns:
        params: Trained parameters
        history: Training history
    """
    # Initialize parameters
    params = initialize_parameters(input_dim, hidden_dim, num_classes)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Process each batch from generator
        for batch_x, batch_y in train_generator():
            loss, accuracy, params = train_step(batch_x, batch_y, params, 
                                               num_classes, learning_rate)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        # Average metrics for the epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        
        # Validation
        if val_generator is not None:
            val_losses = []
            val_accs = []
            
            for batch_x, batch_y in val_generator():
                val_loss, val_acc = evaluate(batch_x, batch_y, params, num_classes)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
    
    return params, history


# Training Loop (for pre-loaded numpy arrays)
def train(X_train, y_train, X_val, y_val, input_dim, hidden_dim, num_classes,
          epochs=100, batch_size=32, learning_rate=0.001, lambda_reg=0.0, verbose=True):
    """
    Train the neural network with pre-loaded data and L2 regularization
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of classes
        epochs: Number of training epochs
        batch_size: Batch size for mini-batch training
        learning_rate: Learning rate
        lambda_reg: L2 regularization strength (0 = no regularization)
        verbose: Print training progress
    
    Returns:
        params: Trained parameters
        history: Training history
    """
    # Initialize parameters
    params = initialize_parameters(input_dim, hidden_dim, num_classes)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Mini-batch training
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            loss, accuracy, params = train_step(X_batch, y_batch, params, 
                                               num_classes, learning_rate, lambda_reg)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        # Average metrics for the epoch
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        
        # Validation
        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate(X_val, y_val, params, num_classes, lambda_reg)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
    
    return params, history


# Save/Load Functions
def save_model(params, history, input_dim, hidden_dim, num_classes, filepath):
    """
    Save model parameters to file
    
    Args:
        params: Dictionary with network parameters
        history: Training history
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_dict = {
        'params': params,
        'history': history,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model parameters from file
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        params: Network parameters
        model_info: Dictionary with model metadata
    """
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    
    params = model_dict['params']
    model_info = {
        'input_dim': model_dict['input_dim'],
        'hidden_dim': model_dict['hidden_dim'],
        'num_classes': model_dict['num_classes'],
        'history': model_dict['history']
    }
    
    print(f"Model loaded from {filepath}")
    return params, model_info







