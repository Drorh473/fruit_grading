"""
User Dashboard Routes (Operator Role) - UPDATED VERSION
Endpoints for operator dashboard and recent results
Now uses dashboard_metadata.json from last model build
"""
from flask import Blueprint, jsonify
from datetime import datetime, timedelta
from utils.utils import get_collection
from utils.model_metadata import load_dashboard_metadata, format_for_user_dashboard

user_dashboard_bp = Blueprint('user_dashboard', __name__)

@user_dashboard_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """
    Get summary statistics for user dashboard
    Uses dashboard_metadata if available, falls back to database
    """
    try:
        # Try to load from dashboard_metadata first
        metadata = load_dashboard_metadata()
        if metadata:
            # Use the formatted data for user dashboard
            user_data = format_for_user_dashboard()
            if user_data and 'stats' in user_data:
                return jsonify(user_data['stats']), 200
        
        # Fallback to database query (original logic)
        collection = get_collection('images')
        
        # Get today's date range
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        # Aggregate by fruit type for today
        pipeline = [
            {'$match': {'timestamp': {'$gte': today_start, '$lt': today_end}}},
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        stats = {
            'totalToday': sum(r['count'] for r in results),
            'marketCount': next((r['count'] for r in results if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in results if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in results if r['_id'] == 'premium'), 0)
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        print(f"Error in get_dashboard_stats: {e}")
        return jsonify({
            'totalToday': 0,
            'marketCount': 0,
            'standardCount': 0,
            'premiumCount': 0
        }), 200


@user_dashboard_bp.route('/recent-results', methods=['GET'])
def get_recent_results():
    """Get recent classification results (last 10 unique objects)"""
    try:
        collection = get_collection('images')
        
        # Aggregate to get unique objects with latest timestamp
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}},
            {'$limit': 10}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Format results for frontend
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


@user_dashboard_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get basic model information for operator display (NEW ENDPOINT)
    Shows model accuracy and last training time
    """
    try:
        metadata = load_dashboard_metadata()
        if metadata:
            return jsonify({
                'accuracy': metadata['performance']['test_accuracy'],
                'lastTrained': metadata['timestamp'],
                'totalSamples': metadata['dataset_info']['total_objects']
            }), 200
        
        return jsonify({
            'accuracy': 0.0,
            'lastTrained': None,
            'totalSamples': 0
        }), 200
        
    except Exception as e:
        print(f"Error in get_model_info: {e}")
        return jsonify({
            'accuracy': 0.0,
            'lastTrained': None,
            'totalSamples': 0
        }), 200