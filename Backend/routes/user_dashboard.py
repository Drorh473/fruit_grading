"""User Dashboard Routes for operator dashboard and recent results."""
from flask import Blueprint, jsonify
from datetime import datetime, timedelta
from utils.utils import get_collection
from utils.model_metadata import load_dashboard_metadata, format_for_user_dashboard

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

        cm = metadata.get('confusion_matrix', [])
        label_mapping = metadata.get('label_mapping', {})
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        avg_conf = metadata.get('performance', {}).get('avg_confidence', 0.5)

        if not cm or not class_names:
            return jsonify([]), 200

        # Generate test predictions from confusion matrix (same as Results page)
        predictions = []
        pred_id = 0

        for actual_idx, actual_class in enumerate(class_names):
            for predicted_idx, predicted_class in enumerate(class_names):
                count = cm[actual_idx][predicted_idx]
                is_correct = actual_idx == predicted_idx

                for _ in range(count):
                    pred_id += 1
                    obj_id = f"test_obj_{pred_id:03d}"

                    conf = avg_conf if avg_conf else 0.5
                    conf_value = conf + 0.1 if is_correct else conf - 0.1
                    conf_value = max(0.1, min(0.99, conf_value))

                    predictions.append({
                        'id': obj_id,
                        'type': predicted_class,
                        'confidence': conf_value,
                        'timestamp': metadata.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    })

        # Return only the first 5 results
        return jsonify(predictions[:5]), 200

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