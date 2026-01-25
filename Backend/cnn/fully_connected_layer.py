import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
from cnn.activation_functions import relu, softmax, relu_derivative

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')


def initialize_parameters(input_dim, hidden_dim, num_classes):
    """Initialize network parameters with He initialization"""
    params = {
        'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim),
        'b2': np.zeros((1, num_classes))
    }
    return params


def forward_pass(X, params, dropout_rate=0.0, training=True):
    """
    Forward pass with optional dropout
    Architecture: Input -> FC1 -> ReLU -> Dropout -> FC2 -> Softmax -> Output
    """
    # First layer: Linear -> ReLU
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    
    # Apply dropout during training
    dropout_mask = None
    if training and dropout_rate > 0:
        dropout_mask = (np.random.rand(*A1.shape) > dropout_rate) / (1 - dropout_rate)
        A1 = A1 * dropout_mask
    
    # Second layer: Linear -> Softmax
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = softmax(Z2)
    
    cache = {
        'X': X,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2,
        'dropout_mask': dropout_mask
    }
    
    return A2, cache


def compute_loss(y_pred, y_true, num_classes=3, params=None, lambda_reg=0.0):
    """Compute cross-entropy loss with optional L2 regularization"""
    batch_size = y_pred.shape[0]

    if y_true.ndim == 1:
        y_true_onehot = np.zeros((batch_size, num_classes))
        y_true_onehot[np.arange(batch_size), y_true] = 1
    else:
        y_true_onehot = y_true
    
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    ce_loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / batch_size
    
    if params is not None and lambda_reg > 0:
        l2_penalty = lambda_reg * (np.sum(params['W1']**2) + np.sum(params['W2']**2))
        total_loss = ce_loss + l2_penalty
        return total_loss
    
    return ce_loss


def backward_pass(y_true, params, cache, num_classes=3, lambda_reg=0.0, dropout_rate=0.0):
    """Backward pass with L2 regularization and dropout support"""
    batch_size = cache['X'].shape[0]
    
    if y_true.ndim == 1:
        y_true_onehot = np.zeros((batch_size, num_classes))
        y_true_onehot[np.arange(batch_size), y_true] = 1
    else:
        y_true_onehot = y_true
    
    dZ2 = cache['A2'] - y_true_onehot
    
    dW2 = np.dot(cache['A1'].T, dZ2) / batch_size
    if lambda_reg > 0:
        dW2 += 2 * lambda_reg * params['W2']
    db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
    
    dA1 = np.dot(dZ2, params['W2'].T)
    
    # Apply dropout mask to gradient
    if cache['dropout_mask'] is not None:
        dA1 = dA1 * cache['dropout_mask']
    
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    
    dW1 = np.dot(cache['X'].T, dZ1) / batch_size
    if lambda_reg > 0:
        dW1 += 2 * lambda_reg * params['W1']
    db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
    
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


def update_parameters(params, grads, learning_rate):
    """Update parameters using gradient descent"""
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params


def train_step(X, y, params, num_classes, learning_rate, lambda_reg=0.005, dropout_rate=0.0):
    """Single training step with dropout"""
    y_pred, cache = forward_pass(X, params, dropout_rate=dropout_rate, training=True)
    loss = compute_loss(y_pred, y, num_classes, params, lambda_reg)
    
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y)
    
    grads = backward_pass(y, params, cache, num_classes, lambda_reg, dropout_rate)
    params = update_parameters(params, grads, learning_rate)
    
    return loss, accuracy, params


def predict(X, params):
    """Make predictions (no dropout during inference)"""
    probabilities, _ = forward_pass(X, params, dropout_rate=0.0, training=False)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def evaluate(X, y, params, num_classes=3, lambda_reg=0.0):
    """Evaluate the model"""
    y_pred, _ = forward_pass(X, params, dropout_rate=0.0, training=False)
    loss = compute_loss(y_pred, y, num_classes, params, lambda_reg)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y)
    return loss, accuracy


def compute_avg_confidence(X, y, params):
    """
    Compute average confidence of predictions
    Returns confidence for predicted class (not necessarily correct class)
    """
    probabilities, _ = forward_pass(X, params, dropout_rate=0.0, training=False)
    predictions = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    return np.mean(confidences)


def train(X_train, y_train, X_val, y_val, input_dim, hidden_dim, num_classes,
          epochs=100, batch_size=32, learning_rate=0.001, lambda_reg=0.005, 
          dropout_rate=0.0, early_stopping_patience=50, verbose=True):
    """
    Train with early stopping and dropout
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_dim, hidden_dim, num_classes: Network architecture
        epochs: Maximum training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate
        lambda_reg: L2 regularization strength
        dropout_rate: Dropout rate (0 = no dropout)
        early_stopping_patience: Stop if val_loss doesn't improve for this many epochs
        verbose: Print progress
    
    Returns:
        params: Best parameters (by validation loss)
        history: Training history
    """
    params = initialize_parameters(input_dim, hidden_dim, num_classes)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    n_samples = X_train.shape[0]
    n_batches = max(1, int(np.ceil(n_samples / batch_size)))
    
    best_val_loss = float('inf')
    best_params = None
    patience_counter = 0
    
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
            
            loss, accuracy, params = train_step(
                X_batch, y_batch, params, num_classes, 
                learning_rate, lambda_reg, dropout_rate
            )
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        
        # Validation
        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate(X_val, y_val, params, num_classes, lambda_reg)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {k: v.copy() for k, v in params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
    
    # Return best parameters if early stopping was used
    if best_params is not None:
        params = best_params
        if verbose:
            print(f"Restored best parameters (val_loss: {best_val_loss:.4f})")
    
    return params, history


def train_from_generator(train_generator, val_generator, input_dim=None, hidden_dim=None, 
                         num_classes=3, epochs=100, learning_rate=0.001, verbose=True, lambda_reg=0.005):
    """Train using a batch generator"""
    params = initialize_parameters(input_dim, hidden_dim, num_classes)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_x, batch_y in train_generator():
            loss, accuracy, params = train_step(batch_x, batch_y, params, 
                                               num_classes, learning_rate, lambda_reg)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        
        if val_generator is not None:
            val_losses = []
            val_accs = []
            
            for batch_x, batch_y in val_generator():
                val_loss, val_acc = evaluate(batch_x, batch_y, params, num_classes, lambda_reg)
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


def save_model(params, history, input_dim, hidden_dim, num_classes, filepath, pca=None):
    """Save model parameters to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_dict = {
        'params': params,
        'history': history,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'pca': pca
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model parameters from file"""
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    
    params = model_dict['params']
    model_info = {
        'input_dim': model_dict['input_dim'],
        'hidden_dim': model_dict['hidden_dim'],
        'num_classes': model_dict['num_classes'],
        'history': model_dict['history'],
        'pca': model_dict.get('pca', None)
    }
    
    print(f"Model loaded from {filepath}")
    return params, model_info