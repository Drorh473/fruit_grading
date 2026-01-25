"""
Dashboard Metadata Management
Saves and loads model training results and dataset information for dashboards
"""
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')


def save_dashboard_metadata(results, train_count, test_count, feature_dim, params, label_mapping, confusion_matrix=None, train_features=None, test_features=None, avg_confidence=None):
    """
    Save comprehensive dashboard metadata after model training
    
    Args:
        results: Dictionary with training results from train_classifier
        train_count: Number of training samples
        test_count: Number of testing samples
        feature_dim: Feature dimension (after PCA if applied)
        params: Trained model parameters
        label_mapping: Dictionary mapping fruit types to labels
        confusion_matrix: Optional confusion matrix
        train_features: Training feature dictionary (for class distribution)
        test_features: Testing feature dictionary (for class distribution)
        avg_confidence: Average prediction confidence (0-1 scale)
    
    Returns:
        filepath: Path to saved metadata file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Calculate dataset statistics
    training_count = train_count
    testing_count = test_count
    
    # Count images per object (assuming 4 cameras, multiple frames)
    total_images = (training_count + testing_count) * 4
    
    # Calculate class distribution
    class_distribution = {}
    if train_features is not None and test_features is not None:
        for fruit_type in label_mapping.keys():
            train_type_count = sum(1 for data in train_features.values() 
                             if data.get('fruit_type') == fruit_type)
            test_type_count = sum(1 for data in test_features.values() 
                            if data.get('fruit_type') == fruit_type)
            class_distribution[fruit_type] = {
                'train': train_type_count,
                'test': test_type_count,
                'total': train_type_count + test_type_count
            }
    else:
        for fruit_type in label_mapping.keys():
            class_distribution[fruit_type] = {'train': 0, 'test': 0, 'total': 0}
    
    # Extract training history
    history = results.get('history', {})
    
    # Calculate per-class performance from confusion matrix
    per_class_metrics = {}
    if confusion_matrix is not None:
        cm = confusion_matrix
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        
        for i, class_name in enumerate(class_names):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(cm[i, :].sum())
            }
    
    # Build comprehensive metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'architecture': f'2-Layer FC (input->{params["W1"].shape[1]}->output)',
            'input_dim': int(feature_dim),
            'hidden_dim': int(params['W1'].shape[1]),
            'num_classes': int(params['W2'].shape[1]),
            'total_parameters': int(
                params['W1'].size + params['b1'].size + 
                params['W2'].size + params['b2'].size
            )
        },
        'dataset_info': {
            'training_count': training_count,
            'testing_count': testing_count,
            'total_objects': training_count + testing_count,
            'total_images': total_images,
            'feature_dim': int(feature_dim),
            'class_distribution': class_distribution
        },
        'performance': {
            'train_accuracy': float(results['train_accuracy']),
            'train_loss': float(results['train_loss']),
            'test_accuracy': float(results['test_accuracy']),
            'test_loss': float(results['test_loss']),
            'avg_confidence': float(avg_confidence) if avg_confidence is not None else None,
            'final_epoch': len(history.get('train_loss', [])),
        },
        'training_history': {
            'train_loss': [float(x) for x in history.get('train_loss', [])],
            'train_accuracy': [float(x) for x in history.get('train_accuracy', [])],
            'val_loss': [float(x) for x in history.get('val_loss', [])],
            'val_accuracy': [float(x) for x in history.get('val_accuracy', [])]
        },
        'per_class_performance': per_class_metrics,
        'label_mapping': label_mapping,
        'confusion_matrix': confusion_matrix.tolist() if confusion_matrix is not None else None
    }
    
    # Save as JSON
    metadata_path = os.path.join(MODEL_DIR, 'dashboard_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n Dashboard metadata saved to {metadata_path}")
    
    # Save pickle for backward compatibility
    pickle_path = os.path.join(MODEL_DIR, 'dashboard_metadata.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata_path


def load_dashboard_metadata():
    """Load the latest dashboard metadata"""
    metadata_path = os.path.join(MODEL_DIR, 'dashboard_metadata.json')
    
    if not os.path.exists(metadata_path):
        print(f"No dashboard metadata found at {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f" Dashboard metadata loaded from {metadata_path}")
        return metadata
    
    except Exception as e:
        print(f"Error loading dashboard metadata: {e}")
        return None


def get_latest_model_info():
    """Get formatted information about the latest trained model"""
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    return {
        'timestamp': metadata['timestamp'],
        'train_accuracy': metadata['performance']['train_accuracy'],
        'test_accuracy': metadata['performance']['test_accuracy'],
        'avg_confidence': metadata['performance'].get('avg_confidence'),
        'total_objects': metadata['dataset_info']['total_objects'],
        'architecture': metadata['model_info']['architecture']
    }


def format_for_admin_dashboard():
    """Format metadata for admin dashboard API"""
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    return {
        'system_status': {
            'database': 'connected',
            'model': 'loaded',
            'cameras': [True, True, True, True]
        },
        'processing_stats': {
            'totalProcessed': metadata['dataset_info']['total_objects'],
            'accuracy': metadata['performance']['test_accuracy'],
            'avgConfidence': metadata['performance'].get('avg_confidence'),
            'lastUpdate': metadata['timestamp']
        },
        'dataset_info': {
            'trainingCount': metadata['dataset_info']['training_count'],
            'testingCount': metadata['dataset_info']['testing_count'],
            'totalImages': metadata['dataset_info']['total_images'],
            'featureDim': metadata['dataset_info']['feature_dim']
        },
        'model_performance': {
            'architecture': metadata['model_info']['architecture'],
            'trainAccuracy': metadata['performance']['train_accuracy'],
            'testAccuracy': metadata['performance']['test_accuracy'],
            'avgConfidence': metadata['performance'].get('avg_confidence'),
            'classes': metadata['model_info']['num_classes']
        },
        'class_distribution': metadata['dataset_info']['class_distribution'],
        'per_class_performance': metadata.get('per_class_performance', {})
    }


def format_for_user_dashboard():
    """Format metadata for user/operator dashboard API"""
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    class_dist = metadata['dataset_info']['class_distribution']
    
    return {
        'stats': {
            'totalToday': metadata['dataset_info']['total_objects'],
            'marketCount': class_dist.get('market', {}).get('total', 0),
            'standardCount': class_dist.get('standard', {}).get('total', 0),
            'premiumCount': class_dist.get('premium', {}).get('total', 0)
        },
        'model_accuracy': metadata['performance']['test_accuracy'],
        'avg_confidence': metadata['performance'].get('avg_confidence'),
        'last_update': metadata['timestamp']
    }