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


def save_dashboard_metadata(results, train_features, test_features, params, label_mapping, confusion_matrix=None):
    """
    Save comprehensive dashboard metadata after model training
    
    Args:
        results: Dictionary with training results from train_classifier
        train_features: Training feature dictionary
        test_features: Testing feature dictionary
        params: Trained model parameters
        label_mapping: Dictionary mapping fruit types to labels
        confusion_matrix: Optional confusion matrix
    
    Returns:
        filepath: Path to saved metadata file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Calculate dataset statistics
    training_count = len(train_features)
    testing_count = len(test_features)
    
    # Get feature dimension from first sample
    first_key = list(train_features.keys())[0]
    feature_dim = train_features[first_key]['features'].shape[0]
    
    # Count images per object (assuming 4 cameras, multiple frames)
    # This is an estimate based on your 4-camera setup
    total_images = (training_count + testing_count) * 4  # 4 cameras per object
    
    # Calculate class distribution
    class_distribution = {}
    for fruit_type in label_mapping.keys():
        train_count = sum(1 for data in train_features.values() 
                         if data.get('fruit_type') == fruit_type)
        test_count = sum(1 for data in test_features.values() 
                        if data.get('fruit_type') == fruit_type)
        class_distribution[fruit_type] = {
            'train': train_count,
            'test': test_count,
            'total': train_count + test_count
        }
    
    # Extract training history
    history = results.get('history', {})
    
    # Calculate per-class performance from confusion matrix
    per_class_metrics = {}
    if confusion_matrix is not None:
        cm = confusion_matrix
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        
        for i, class_name in enumerate(class_names):
            # Calculate precision, recall, f1-score for each class
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
    
    # Save as JSON for easy loading
    metadata_path = os.path.join(MODEL_DIR, 'dashboard_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dashboard metadata saved to {metadata_path}")
    
    # Also save a pickle version for backward compatibility
    pickle_path = os.path.join(MODEL_DIR, 'dashboard_metadata.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata_path


def load_dashboard_metadata():
    """
    Load the latest dashboard metadata
    
    Returns:
        metadata: Dictionary with all dashboard information
        or None if no metadata exists
    """
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
    """
    Get formatted information about the latest trained model
    Useful for quick dashboard queries
    
    Returns:
        Dictionary with key model information or None
    """
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    return {
        'timestamp': metadata['timestamp'],
        'train_accuracy': metadata['performance']['train_accuracy'],
        'test_accuracy': metadata['performance']['test_accuracy'],
        'total_objects': metadata['dataset_info']['total_objects'],
        'architecture': metadata['model_info']['architecture']
    }


def format_for_admin_dashboard():
    """
    Format metadata specifically for admin dashboard API
    
    Returns:
        Dictionary formatted for admin dashboard or None
    """
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    return {
        'system_status': {
            'database': 'connected',
            'model': 'loaded',
            'cameras': [True, True, True, True]  # Assuming 4 cameras active
        },
        'processing_stats': {
            'totalProcessed': metadata['dataset_info']['total_objects'],
            'accuracy': metadata['performance']['test_accuracy'],
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
            'classes': metadata['model_info']['num_classes']
        },
        'class_distribution': metadata['dataset_info']['class_distribution'],
        'per_class_performance': metadata.get('per_class_performance', {})
    }


def format_for_user_dashboard():
    """
    Format metadata specifically for user/operator dashboard API
    
    Returns:
        Dictionary formatted for user dashboard or None
    """
    metadata = load_dashboard_metadata()
    if not metadata:
        return None
    
    # Extract class counts from distribution
    class_dist = metadata['dataset_info']['class_distribution']
    
    return {
        'stats': {
            'totalToday': metadata['dataset_info']['total_objects'],
            'marketCount': class_dist.get('market', {}).get('total', 0),
            'standardCount': class_dist.get('standard', {}).get('total', 0),
            'premiumCount': class_dist.get('premium', {}).get('total', 0),
            'rejectCount': 0  # Add logic if you track rejects separately
        },
        'model_accuracy': metadata['performance']['test_accuracy'],
        'last_update': metadata['timestamp']
    }


if __name__ == "__main__":
    # Test loading
    print("\n=== Testing Dashboard Metadata ===\n")
    
    metadata = load_dashboard_metadata()
    if metadata:
        print("\n✓ Successfully loaded metadata")
        print(f"  Model accuracy: {metadata['performance']['test_accuracy']*100:.2f}%")
        print(f"  Training samples: {metadata['dataset_info']['training_count']}")
        print(f"  Testing samples: {metadata['dataset_info']['testing_count']}")
        
        print("\n=== Admin Dashboard Format ===")
        admin_data = format_for_admin_dashboard()
        if admin_data:
            print(json.dumps(admin_data, indent=2))
        
        print("\n=== User Dashboard Format ===")
        user_data = format_for_user_dashboard()
        if user_data:
            print(json.dumps(user_data, indent=2))
    else:
        print("✗ No metadata available. Run build_model.py first.")