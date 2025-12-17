"""
Admin Dashboard Routes
Endpoints for admin dashboard, system status, and analytics
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
from utils import get_collection, check_db_connection
from shared_state import pipeline_state

admin_dashboard_bp = Blueprint('admin_dashboard', __name__)

@admin_dashboard_bp.route('/system-status', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    try:
        db_status = 'connected' if check_db_connection() else 'disconnected'
        
        state = pipeline_state.get_state()
        
        return jsonify({
            'database': db_status,
            'model': 'loaded' if state['results'] else 'not_trained',
            'pipeline': state['status'],
            'cameras': [True, True, True, True]  # Mock - update with real camera status
        }), 200
        
    except Exception as e:
        print(f"Error in get_system_status: {e}")
        return jsonify({
            'database': 'disconnected',
            'model': 'unknown',
            'pipeline': 'idle',
            'cameras': [False, False, False, False]
        }), 200


@admin_dashboard_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Get statistics for admin dashboard"""
    try:
        collection = get_collection('images')
        
        # Get total processed count
        total_processed = collection.count_documents({})
        
        # Get accuracy from pipeline results
        accuracy = 0.0
        results = pipeline_state.get_results()
        if results:
            accuracy = results.get('test_accuracy', 0.0)
            total_processed = results.get('totalProcessed', total_processed)
        
        # Count by fruit type
        type_pipeline = [
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]
        type_counts = list(collection.aggregate(type_pipeline))
        
        stats = {
            'totalProcessed': total_processed,
            'accuracy': accuracy,
            'marketCount': next((r['count'] for r in type_counts if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in type_counts if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0),
            'rejectCount': next((r['count'] for r in type_counts if r['_id'] == 'reject'), 0),
            'totalImages': total_processed,
            'totalObjects': len(collection.distinct('object_id'))
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        print(f"Error in get_dashboard_stats: {e}")
        return jsonify({
            'totalProcessed': 0,
            'accuracy': 0.0,
            'marketCount': 0,
            'standardCount': 0,
            'premiumCount': 0,
            'rejectCount': 0,
            'totalImages': 0,
            'totalObjects': 0
        }), 200


@admin_dashboard_bp.route('/recent-results', methods=['GET'])
def get_recent_results():
    """Get recent classification results for admin"""
    try:
        limit = int(request.args.get('limit', 10))
        collection = get_collection('images')
        
        # Aggregate to get unique objects
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
        print(f"Error in get_recent_results: {e}")
        return jsonify([]), 200


@admin_dashboard_bp.route('/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information"""
    try:
        collection = get_collection('images')
        
        # Count training and testing samples
        train_count = collection.count_documents({'set_type': 'train'})
        test_count = collection.count_documents({'set_type': 'test'})
        total_images = collection.count_documents({})
        
        # Get feature dimension from pipeline results
        feature_dim = 200704
        results = pipeline_state.get_results()
        if results:
            feature_dim = results.get('featureDim', 200704)
        
        return jsonify({
            'trainingCount': train_count,
            'testingCount': test_count,
            'totalImages': total_images,
            'featureDim': feature_dim
        }), 200
        
    except Exception as e:
        print(f"Error in get_dataset_info: {e}")
        return jsonify({
            'trainingCount': 0,
            'testingCount': 0,
            'totalImages': 0,
            'featureDim': 200704
        }), 200


@admin_dashboard_bp.route('/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    try:
        results = pipeline_state.get_results()
        
        if results:
            return jsonify({
                'architecture': 'ShuffleNetV2 + Multi-View Fusion',
                'trainAccuracy': results.get('train_accuracy', 0.0),
                'testAccuracy': results.get('test_accuracy', 0.0),
                'classes': 4
            }), 200
        else:
            return jsonify({
                'architecture': 'ShuffleNetV2 + Multi-View Fusion',
                'trainAccuracy': 0.0,
                'testAccuracy': 0.0,
                'classes': 4
            }), 200
            
    except Exception as e:
        print(f"Error in get_model_performance: {e}")
        return jsonify({
            'architecture': 'ShuffleNetV2 + Multi-View Fusion',
            'trainAccuracy': 0.0,
            'testAccuracy': 0.0,
            'classes': 4
        }), 200


@admin_dashboard_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    try:
        config = pipeline_state.get_config()
        results = pipeline_state.get_results()
        
        model_info = {
            'architecture': 'ShuffleNetV2 + FC',
            'inputDim': 200704,
            'hiddenDim': config['hiddenDim'],
            'classes': 4,
            'lastTrained': results.get('timestamp') if results else None,
            'accuracy': results.get('test_accuracy') if results else None
        }
        
        return jsonify(model_info), 200
        
    except Exception as e:
        print(f"Error in get_model_info: {e}")
        return jsonify({
            'architecture': 'ShuffleNetV2 + FC',
            'inputDim': 200704,
            'hiddenDim': 16,
            'classes': 4,
            'lastTrained': None,
            'accuracy': None
        }), 200