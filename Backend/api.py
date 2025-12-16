"""
Flask API Server for Fruit Grading System
Provides REST endpoints for frontend to access backend data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pymongo
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from bson import ObjectId

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# MongoDB connection
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')

def get_db():
    """Get database connection"""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    return client[DB_NAME]

# ============================================================================
# USER ENDPOINTS (Operator Role)
# ============================================================================

@app.route('/api/user/dashboard-stats', methods=['GET'])
def get_user_dashboard_stats():
    """
    Get summary statistics for user dashboard
    Returns today's processing counts by type
    """
    try:
        db = get_db()
        collection = db['images']
        
        # Get today's date range
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        # Aggregate counts by fruit type for today
        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': today_start, '$lt': today_end}
                }
            },
            {
                '$group': {
                    '_id': '$fruit_type',
                    'count': {'$sum': 1}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Build response
        stats = {
            'totalToday': sum(r['count'] for r in results),
            'marketCount': next((r['count'] for r in results if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in results if r['_id'] == 'standard'), 0),
            'rejectCount': next((r['count'] for r in results if r['_id'] == 'reject'), 0)
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user/recent-results', methods=['GET'])
def get_recent_results():
    """
    Get recent classification results (last 10)
    Returns object-level results with classification and confidence
    """
    try:
        db = get_db()
        collection = db['images']
        
        # Get unique objects sorted by timestamp
        pipeline = [
            {
                '$sort': {'timestamp': -1}
            },
            {
                '$group': {
                    '_id': '$object_id',
                    'fruit_type': {'$first': '$fruit_type'},
                    'timestamp': {'$first': '$timestamp'},
                    'confidence': {'$first': '$confidence'},
                    'image_count': {'$sum': 1}
                }
            },
            {
                '$sort': {'timestamp': -1}
            },
            {
                '$limit': 10
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Format response
        formatted_results = []
        for r in results:
            formatted_results.append({
                'id': r['_id'],
                'type': r['fruit_type'],
                'confidence': r.get('confidence', 0.0),
                'timestamp': r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp']),
                'images': r['image_count']
            })
        
        return jsonify(formatted_results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/list', methods=['GET'])
def get_results_list():
    """
    Get all classification results with optional filtering
    Query params: search, type, limit, offset
    """
    try:
        db = get_db()
        collection = db['images']
        
        # Get query parameters
        search = request.args.get('search', '')
        fruit_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Build match filter
        match_filter = {}
        if search:
            match_filter['object_id'] = {'$regex': search, '$options': 'i'}
        if fruit_type != 'all':
            match_filter['fruit_type'] = fruit_type
        
        # Aggregate by object_id
        pipeline = [
            {'$match': match_filter} if match_filter else {'$match': {}},
            {
                '$group': {
                    '_id': '$object_id',
                    'fruit_type': {'$first': '$fruit_type'},
                    'timestamp': {'$first': '$timestamp'},
                    'confidence': {'$first': '$confidence'},
                    'image_count': {'$sum': 1}
                }
            },
            {'$sort': {'timestamp': -1}},
            {'$skip': offset},
            {'$limit': limit}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Get total count
        count_pipeline = [
            {'$match': match_filter} if match_filter else {'$match': {}},
            {'$group': {'_id': '$object_id'}},
            {'$count': 'total'}
        ]
        
        count_result = list(collection.aggregate(count_pipeline))
        total = count_result[0]['total'] if count_result else 0
        
        # Format response
        formatted_results = [{
            'id': r['_id'],
            'type': r['fruit_type'],
            'confidence': r.get('confidence', 0.0),
            'timestamp': r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp']),
            'images': r['image_count']
        } for r in results]
        
        return jsonify({
            'results': formatted_results,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db = get_db()
        db.command('ping')
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)