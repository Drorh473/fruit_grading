import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from dotenv import load_dotenv
from cnn.fully_connected_layer import predict

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')


def plot_confusion_matrix(y_true, y_pred, label_mapping, save_path=None):
    """
    Plot confusion matrix with counts
    
    Args:
        y_true: True labels (numpy array of integers)
        y_pred: Predicted labels (numpy array of integers)
        label_mapping: Dictionary mapping fruit_type -> integer
        save_path: Optional path to save the figure
    
    Returns:
        cm: Confusion matrix (numpy array)
    """
    # Get class names sorted by their index
    class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=1,
                linecolor='gray')
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved to {save_path}")
    plt.close()
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, label_mapping, save_path=None):
    """
    Plot normalized confusion matrix (percentages)
    
    Args:
        y_true: True labels (numpy array of integers)
        y_pred: Predicted labels (numpy array of integers)
        label_mapping: Dictionary mapping fruit_type -> integer
        save_path: Optional path to save the figure
    
    Returns:
        cm_normalized: Normalized confusion matrix (numpy array)
    """
    # Get class names sorted by their index
    class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap with percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage'},
                vmin=0, vmax=1,
                linewidths=1,
                linecolor='gray')
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Normalized confusion matrix saved to {save_path}")
    plt.close()
    return cm_normalized


def print_confusion_matrix_metrics(cm, label_mapping):
    """
    Print per-class metrics from confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        label_mapping: Dictionary mapping fruit_type -> integer
    """
    # Get class names sorted by their index
    class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
    
    print("\n" + "="*70)
    print("PER-CLASS METRICS FROM CONFUSION MATRIX")
    print("="*70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    num_classes = len(class_names)
    
    for i in range(num_classes):
        # True positives
        tp = cm[i, i]
        
        # False positives (predicted as i but actually other classes)
        fp = cm[:, i].sum() - tp
        
        # False negatives (actually i but predicted as other classes)
        fn = cm[i, :].sum() - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()
        
        print(f"{class_names[i]:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print("-"*70)
    print(f"{'Overall Accuracy':<15} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70 + "\n")


def print_confusion_matrix_breakdown(cm, label_mapping):
    """
    Print per-class confusion matrix breakdown
    
    Args:
        cm: Confusion matrix (numpy array)
        label_mapping: Dictionary mapping fruit_type -> integer
    """
    class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX BREAKDOWN (Per Class)")
    print("="*60)
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name.upper()}:")
        print(f"  Correctly predicted: {cm[i, i]}/{cm[i, :].sum()} samples")
        
        # Show misclassifications if any
        for j, other_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                print(f"  Misclassified as '{other_class}': {cm[i, j]} samples")
    
    # Overall summary
    print(f"\n" + "-"*60)
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    print(f"Total: {total_correct}/{total_samples} correctly classified ({total_correct/total_samples*100:.2f}%)")
    print("="*60 + "\n")


def generate_full_confusion_matrix_report(results, save_dir=None):
    """
    Generate complete confusion matrix report with all visualizations and metrics
    
    This is the main function that uses all other functions to create a comprehensive report.
    
    Args:
        results: Results dictionary from train_classifier containing:
                - X_test: Test features
                - y_test: Test labels
                - params: Trained parameters
                - label_mapping: Label mapping dictionary
        save_dir: Directory to save the confusion matrix visualizations
                 (defaults to MODEL_DIR from .env)
    
    Returns:
        report: Dictionary containing:
               - cm: Confusion matrix
               - cm_normalized: Normalized confusion matrix
               - metrics: Per-class metrics
    """
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRIX REPORT")
    print("="*60 + "\n")
    
    try:
        # Set save directory
        if save_dir is None:
            save_dir = MODEL_DIR
        
        # Extract data from results
        X_test = results['X_test']
        y_test = results['y_test']
        params = results['params']
        label_mapping = results['label_mapping']
        
        # Get predictions
        print("Making predictions on test set...")
        y_pred, probabilities = predict(X_test, params)
        
        # 1. Generate count-based confusion matrix
        print("\n1. Generating confusion matrix (counts)...")
        cm = plot_confusion_matrix(
            y_test, y_pred, label_mapping,
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # 2. Generate normalized confusion matrix
        print("\n2. Generating normalized confusion matrix (percentages)...")
        cm_normalized = plot_normalized_confusion_matrix(
            y_test, y_pred, label_mapping,
            save_path=os.path.join(save_dir, 'confusion_matrix_normalized.png')
        )
        
        # 3. Print detailed per-class metrics
        print("\n3. Computing per-class metrics...")
        print_confusion_matrix_metrics(cm, label_mapping)
        
        # 4. Print breakdown
        print_confusion_matrix_breakdown(cm, label_mapping)
        
        # Compile metrics
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        num_classes = len(class_names)
        
        metrics = {}
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = cm[i, :].sum()
            
            metrics[class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }
        
        # Create report dictionary
        report = {
            'cm': cm,
            'cm_normalized': cm_normalized,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_true': y_test,
            'probabilities': probabilities
        }
        
        print(" Confusion matrix report generation complete!\n")
        
        return report
        
    except Exception as e:
        print(f"\n Confusion matrix report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

