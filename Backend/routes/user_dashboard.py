"""User Dashboard Routes for operator dashboard and recent results."""
from flask import Blueprint, jsonify
from datetime import datetime, timedelta
from utils.utils import get_collection
from utils.model_metadata import load_dashboard_metadata, format_for_user_dashboard, generate_predictions_from_confusion_matrix

user_dashboard_bp = Blueprint('user_dashboard', __name__)

@user_dashboard_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Get summary statistics for user dashboard."""
    try:
        metadata = load_dashboard_metadata()
        if metadata:
            user_data = format_for_user_dashboard()
            if user_data and 'stats' in user_data:
                return jsonify(user_data['stats']), 200

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
        return jsonify({
            'totalToday': 0,
            'marketCount': 0,
            'standardCount': 0,
            'premiumCount': 0
        }), 200


@user_dashboard_bp.route('/recent-results', methods=['GET'])
def get_recent_results():
    """Get recent classification results from test predictions (last 5)"""
    try:
        metadata = load_dashboard_metadata()

        if not metadata:
            return jsonify([]), 200

        predictions = generate_predictions_from_confusion_matrix(metadata, limit=5)
        return jsonify(predictions), 200

    except Exception as e:
        return jsonify([]), 200


@user_dashboard_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get basic model information for operator display."""
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
        return jsonify({
            'accuracy': 0.0,
            'lastTrained': None,
            'totalSamples': 0
        }), 200