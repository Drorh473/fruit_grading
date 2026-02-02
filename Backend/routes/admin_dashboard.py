"""Admin Dashboard Routes for system status and analytics."""
from flask import Blueprint, jsonify, request
from datetime import datetime
from utils.utils import get_collection, check_db_connection
from utils.shared_state import pipeline_state
from utils.model_metadata import load_dashboard_metadata, format_for_admin_dashboard

admin_dashboard_bp = Blueprint('admin_dashboard', __name__)

@admin_dashboard_bp.route('/system-status', methods=['GET'])
def get_system_status():
    """Get overall system status - uses dashboard_metadata if available"""
    try:
        db_status = 'connected' if check_db_connection() else 'disconnected'
        
        metadata = load_dashboard_metadata()
        if metadata:
            return jsonify({
                'database': db_status,
                'model': 'loaded',
                'pipeline': 'idle',  
                'cameras': [True, True, True, True]  
            }), 200
        
        state = pipeline_state.get_state()
        return jsonify({
            'database': db_status,
            'model': 'loaded' if state['results'] else 'not_trained',
            'pipeline': state['status'],
            'cameras': [True, True, True, True]
        }), 200
        
    except Exception as e:
        return jsonify({
            'database': 'disconnected',
            'model': 'unknown',
            'pipeline': 'idle',
            'cameras': [False, False, False, False]
        }), 200


@admin_dashboard_bp.route('/processing-stats', methods=['GET'])
def get_processing_stats():
    """Get processing statistics from dashboard_metadata"""
    try:
        metadata = load_dashboard_metadata()
        if metadata:
            return jsonify({
                'totalProcessed': metadata['dataset_info']['total_objects'],
                'accuracy': metadata['performance']['test_accuracy'],
                'lastUpdate': metadata['timestamp']
            }), 200

        return jsonify({
            'totalProcessed': 0,
            'accuracy': 0.0,
            'lastUpdate': None
        }), 200

    except Exception as e:
        return jsonify({
            'totalProcessed': 0,
            'accuracy': 0.0,
            'lastUpdate': None
        }), 200


@admin_dashboard_bp.route('/recent-results', methods=['GET'])
def get_recent_results():
    """Get recent classification results"""
    try:
        limit = int(request.args.get('limit', 10))
        collection = get_collection('images')
        
        agg_pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'}
            }},
            {'$sort': {'timestamp': -1}},
            {'$limit': limit}
        ]
        
        results = list(collection.aggregate(agg_pipeline))
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['_id'],
                'type': result['fruit_type'],
                'confidence': result.get('confidence', 0.90),
                'timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result['timestamp'], datetime) else str(result['timestamp'])
            })
        
        return jsonify(formatted_results), 200
        
    except Exception as e:
        return jsonify([]), 200


@admin_dashboard_bp.route('/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information from dashboard_metadata"""
    try:
        metadata = load_dashboard_metadata()
        if metadata:
            dataset_info = metadata['dataset_info']
            return jsonify({
                'trainingCount': dataset_info['training_count'],
                'testingCount': dataset_info['testing_count'],
                'totalImages': dataset_info['total_images'],
                'featureDim': dataset_info['feature_dim']
            }), 200

        return jsonify({
            'trainingCount': 0,
            'testingCount': 0,
            'totalImages': 0,
            'featureDim': 0
        }), 200

    except Exception as e:
        return jsonify({
            'trainingCount': 0,
            'testingCount': 0,
            'totalImages': 0,
            'featureDim': 0
        }), 200


@admin_dashboard_bp.route('/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics - uses dashboard_metadata"""
    try:
        metadata = load_dashboard_metadata()
        if metadata:
            model_info = metadata['model_info']
            performance = metadata['performance']
            return jsonify({
                'architecture': model_info['architecture'],
                'trainAccuracy': performance['train_accuracy'],
                'testAccuracy': performance['test_accuracy'],
                'classes': model_info['num_classes']
            }), 200
        
        results = pipeline_state.get_results()
        if results:
            return jsonify({
                'architecture': 'ShuffleNetV2 + Multi-View Fusion',
                'trainAccuracy': results.get('train_accuracy', 0.0),
                'testAccuracy': results.get('test_accuracy', 0.0),
                'classes': 3
            }), 200
        else:
            return jsonify({
                'architecture': 'ShuffleNetV2 + Multi-View Fusion',
                'trainAccuracy': 0.0,
                'testAccuracy': 0.0,
                'classes': 3
            }), 200
            
    except Exception as e:
        return jsonify({
            'architecture': 'ShuffleNetV2 + Multi-View Fusion',
            'trainAccuracy': 0.0,
            'testAccuracy': 0.0,
            'classes': 3
        }), 200


@admin_dashboard_bp.route('/per-class-performance', methods=['GET'])
def get_per_class_performance():
    """Get per-class performance metrics"""
    try:
        metadata = load_dashboard_metadata()
        if metadata and 'per_class_performance' in metadata:
            return jsonify(metadata['per_class_performance']), 200
        
        return jsonify({}), 200
        
    except Exception as e:
        return jsonify({}), 200


@admin_dashboard_bp.route('/training-history', methods=['GET'])
def get_training_history():
    """Get training history for charts."""
    try:
        metadata = load_dashboard_metadata()
        if metadata and 'training_history' in metadata:
            return jsonify(metadata['training_history']), 200
        
        return jsonify({
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }), 200
        
    except Exception as e:
        return jsonify({
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }), 200


@admin_dashboard_bp.route('/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data."""
    try:
        metadata = load_dashboard_metadata()
        if metadata and metadata.get('confusion_matrix'):
            return jsonify({
                'matrix': metadata['confusion_matrix'],
                'labels': list(metadata['label_mapping'].keys())
            }), 200
        
        return jsonify({
            'matrix': None,
            'labels': []
        }), 200
        
    except Exception as e:
        return jsonify({
            'matrix': None,
            'labels': []
        }), 200


@admin_dashboard_bp.route('/full-dashboard-data', methods=['GET'])
def get_full_dashboard_data():
    """Get all dashboard data in one call."""
    try:
        admin_data = format_for_admin_dashboard()
        if admin_data:
            return jsonify(admin_data), 200

        return jsonify({
            'system_status': {
                'database': 'connected' if check_db_connection() else 'disconnected',
                'model': 'not_trained',
                'cameras': [True, True, True, True]
            },
            'processing_stats': {
                'totalProcessed': 0,
                'accuracy': 0.0,
                'lastUpdate': None
            },
            'dataset_info': {
                'trainingCount': 0,
                'testingCount': 0,
                'totalImages': 0,
                'featureDim': 0
            },
            'model_performance': {
                'architecture': 'Not trained',
                'trainAccuracy': 0.0,
                'testAccuracy': 0.0,
                'classes': 0
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500