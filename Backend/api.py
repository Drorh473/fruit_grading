from flask import Flask, jsonify, request
from flask_cors import CORS
import pymongo
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from bson import ObjectId

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Flask app setup
app = Flask(__name__)
CORS(app)

# MongoDB connection
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')

# Pipeline state (in production, use Redis or similar)
pipeline_state = {
    'running': False,
    'status': 'idle',
    'currentStep': 0,
    'progress': 0,
    'logs': []
}

def get_db():
    """Get database connection"""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    return client[DB_NAME]

# ============================================================================
# USER ENDPOINTS (Operator Role)
# ============================================================================

@app.route('/api/user/dashboard-stats', methods=['GET'])
def get_user_dashboard_stats():
    """Get summary statistics for user dashboard"""
    try:
        db = get_db()
        collection = db['images']
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        pipeline = [
            {'$match': {'timestamp': {'$gte': today_start, '$lt': today_end}}},
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
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
    """Get recent classification results"""
    try:
        db = get_db()
        collection = db['images']
        
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
        
        formatted = [{
            'id': r['_id'],
            'type': r['fruit_type'],
            'confidence': r.get('confidence', 0.0),
            'timestamp': r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp']),
            'images': r['image_count']
        } for r in results]
        
        return jsonify(formatted), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/list', methods=['GET'])
def get_results_list():
    """Get all results with filtering"""
    try:
        db = get_db()
        collection = db['images']
        
        search = request.args.get('search', '')
        fruit_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        match_filter = {}
        if search:
            match_filter['object_id'] = {'$regex': search, '$options': 'i'}
        if fruit_type != 'all':
            match_filter['fruit_type'] = fruit_type
        
        pipeline = [
            {'$match': match_filter} if match_filter else {'$match': {}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}},
            {'$skip': offset},
            {'$limit': limit}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        count_pipeline = [
            {'$match': match_filter} if match_filter else {'$match': {}},
            {'$group': {'_id': '$object_id'}},
            {'$count': 'total'}
        ]
        
        count_result = list(collection.aggregate(count_pipeline))
        total = count_result[0]['total'] if count_result else 0
        
        formatted = [{
            'id': r['_id'],
            'type': r['fruit_type'],
            'confidence': r.get('confidence', 0.0),
            'timestamp': r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp']),
            'images': r['image_count']
        } for r in results]
        
        return jsonify({
            'results': formatted,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data"""
    # In production, load from saved model evaluation
    return jsonify({
        'classes': ['market', 'standard', 'reject'],
        'matrix': [[8, 1, 0], [2, 6, 1], [0, 2, 8]],
        'metrics': {
            'market': {'precision': 0.80, 'recall': 0.89, 'f1': 0.84},
            'standard': {'precision': 0.67, 'recall': 0.67, 'f1': 0.67},
            'reject': {'precision': 0.89, 'recall': 0.80, 'f1': 0.84}
        }
    }), 200


@app.route('/api/results/export', methods=['POST'])
def export_results():
    """Export results as CSV"""
    try:
        db = get_db()
        collection = db['images']
        
        data = request.get_json() or {}
        filters = data.get('filters', {})
        
        match_filter = {}
        if filters.get('search'):
            match_filter['object_id'] = {'$regex': filters['search'], '$options': 'i'}
        if filters.get('type') and filters['type'] != 'all':
            match_filter['fruit_type'] = filters['type']
        
        pipeline = [
            {'$match': match_filter} if match_filter else {'$match': {}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        csv_lines = ['Object ID,Type,Confidence,Timestamp,Images']
        for r in results:
            timestamp_str = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp'])
            csv_lines.append(f"{r['_id']},{r['fruit_type']},{r.get('confidence', 0.0):.4f},{timestamp_str},{r['image_count']}")
        
        return '\n'.join(csv_lines), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=results.csv'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ADMIN ENDPOINTS - Dashboard
# ============================================================================

@app.route('/api/admin/system-status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        db = get_db()
        db.command('ping')
        db_status = 'connected'
    except:
        db_status = 'disconnected'
    
    # Check model file
    model_path = os.path.join(os.getenv('MODEL_DIR', 'saved_models'), 'fruit_classifier.pkl')
    model_status = 'loaded' if os.path.exists(model_path) else 'not found'
    
    # Camera status (mock - in production, check actual hardware)
    cameras = [True, True, True, True]
    
    return jsonify({
        'database': db_status,
        'model': model_status,
        'cameras': cameras
    }), 200


@app.route('/api/admin/processing-stats', methods=['GET'])
def get_processing_stats():
    """Get processing statistics"""
    try:
        db = get_db()
        collection = db['images']
        
        # Count total processed objects
        total = collection.distinct('object_id')
        
        return jsonify({
            'totalProcessed': len(total),
            'accuracy': 0.3636,  # From test results
            'lastUpdate': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/recent-results', methods=['GET'])
def get_admin_recent_results():
    """Get recent results for admin"""
    limit = int(request.args.get('limit', 10))
    return get_recent_results()


@app.route('/api/admin/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get dataset information"""
    try:
        db = get_db()
        collection = db['images']
        
        training_count = len(collection.distinct('object_id', {'set_type': 'training'}))
        testing_count = len(collection.distinct('object_id', {'set_type': 'testing'}))
        total_images = collection.count_documents({})
        
        return jsonify({
            'trainingCount': training_count,
            'testingCount': testing_count,
            'totalImages': total_images,
            'featureDim': 200704
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'trainAccuracy': 1.0,
        'testAccuracy': 0.3636,
        'architecture': 'ShuffleNetV2 + FC',
        'classes': 3
    }), 200


# ============================================================================
# ADMIN ENDPOINTS - Camera Monitor
# ============================================================================

@app.route('/api/cameras/status', methods=['GET'])
def get_camera_statuses():
    """Get all camera statuses"""
    cameras = []
    num_cameras = int(os.getenv('NUM_OF_CAMERAS', 4))
    angles = ['Front View', 'Right View', 'Back View', 'Left View']
    
    for i in range(num_cameras):
        cameras.append({
            'id': i,
            'name': f'Camera {i}',
            'status': True,  # Mock - check actual hardware
            'angle': angles[i] if i < len(angles) else f'Angle {i}',
            'fps': int(os.getenv('CAMERA_FPS', 30)),
            'resolution': '224x224'
        })
    
    return jsonify(cameras), 200


@app.route('/api/cameras/<int:camera_id>', methods=['GET'])
def get_camera_details(camera_id):
    """Get specific camera details"""
    angles = ['Front View', 'Right View', 'Back View', 'Left View']
    
    return jsonify({
        'id': camera_id,
        'name': f'Camera {camera_id}',
        'status': True,
        'angle': angles[camera_id] if camera_id < len(angles) else f'Angle {camera_id}',
        'fps': int(os.getenv('CAMERA_FPS', 30)),
        'resolution': '224x224',
        'preprocessing': 'Gaussian Blur + CLAHE'
    }), 200


@app.route('/api/cameras/<int:camera_id>/refresh', methods=['POST'])
def refresh_camera(camera_id):
    """Refresh specific camera"""
    return jsonify({'success': True, 'message': f'Camera {camera_id} refreshed'}), 200


@app.route('/api/cameras/refresh-all', methods=['POST'])
def refresh_all_cameras():
    """Refresh all cameras"""
    return jsonify({'success': True, 'message': 'All cameras refreshed'}), 200


@app.route('/api/cameras/config', methods=['GET'])
def get_camera_config():
    """Get camera configuration"""
    return jsonify({
        'fps': int(os.getenv('CAMERA_FPS', 30)),
        'numCameras': int(os.getenv('NUM_OF_CAMERAS', 4)),
        'imageSize': '224x224',
        'preprocessing': 'Gaussian Blur + CLAHE'
    }), 200


# ============================================================================
# ADMIN ENDPOINTS - Processing Pipeline
# ============================================================================

@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start processing pipeline"""
    global pipeline_state
    
    if pipeline_state['running']:
        return jsonify({'error': 'Pipeline already running'}), 400
    
    config = request.get_json() or {}
    
    # Start pipeline in background
    pipeline_state['running'] = True
    pipeline_state['status'] = 'running'
    pipeline_state['currentStep'] = 0
    pipeline_state['progress'] = 0
    pipeline_state['logs'] = [{'message': 'Pipeline started', 'type': 'info', 'timestamp': datetime.now().isoformat()}]
    
    # In production, start actual pipeline here
    
    return jsonify({
        'pipelineId': 'pipeline_1',
        'status': 'started'
    }), 200


@app.route('/api/pipeline/stop', methods=['POST'])
def stop_pipeline():
    """Stop pipeline"""
    global pipeline_state
    
    pipeline_state['running'] = False
    pipeline_state['status'] = 'stopped'
    pipeline_state['logs'].append({'message': 'Pipeline stopped', 'type': 'warning', 'timestamp': datetime.now().isoformat()})
    
    return jsonify({'success': True}), 200


@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get pipeline status"""
    return jsonify(pipeline_state), 200


@app.route('/api/pipeline/logs', methods=['GET'])
def get_pipeline_logs():
    """Get pipeline logs"""
    limit = int(request.args.get('limit', 100))
    return jsonify(pipeline_state['logs'][-limit:]), 200


@app.route('/api/pipeline/config', methods=['GET'])
def get_pipeline_config():
    """Get pipeline configuration"""
    return jsonify({
        'hiddenDim': 16,
        'epochs': 100,
        'learningRate': 0.0005,
        'lambdaReg': 0.001,
        'batchSize': 32
    }), 200


# ============================================================================
# ADMIN ENDPOINTS - Settings
# ============================================================================

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get all settings"""
    return jsonify({
        'dbName': os.getenv('DB_NAME'),
        'mongoConnection': os.getenv('MONGO_CONNECTION_STRING'),
        'storedDataset': os.getenv('STORED_DATASET_PATH'),
        'originalDataset': os.getenv('ORIGINAL_DATASET_PATH'),
        'processedDataset': os.getenv('PROCESSED_DATASET_PATH'),
        'cameraFps': int(os.getenv('CAMERA_FPS', 30)),
        'numCameras': int(os.getenv('NUM_OF_CAMERAS', 4)),
        'imageSize': '224x224',
        'batchSize': int(os.getenv('BATCH_SIZE', 128)),
        'hiddenDim': 256,
        'learningRate': 0.001,
        'epochs': 100
    }), 200


@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update settings"""
    settings = request.get_json()
    # In production, update .env file or database
    return jsonify(settings), 200


@app.route('/api/settings/test-database', methods=['POST'])
def test_database():
    """Test database connection"""
    data = request.get_json()
    try:
        client = pymongo.MongoClient(data['connectionString'], serverSelectionTimeoutMS=5000)
        client.server_info()
        return jsonify({'success': True, 'message': 'Connection successful'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 200


@app.route('/api/settings/status', methods=['GET'])
def get_settings_status():
    """Get system status for settings"""
    return get_system_status()


# ============================================================================
# ADMIN ENDPOINTS - Add Fruit
# ============================================================================

@app.route('/api/fruit/validate', methods=['POST'])
def validate_folder():
    """Validate fruit folder structure"""
    data = request.get_json()
    folder_path = data.get('folderPath')
    
    # Mock validation - in production, check actual folder
    return jsonify({
        'valid': True,
        'message': 'Folder structure is valid',
        'details': {
            'anglesFound': 4,
            'totalImages': 60
        }
    }), 200


@app.route('/api/fruit/process', methods=['POST'])
def process_fruit():
    """Process new fruit"""
    data = request.get_json()
    # In production, call actual processing pipeline
    
    return jsonify({
        'objectId': 'obj0015',
        'predictedType': 'market',
        'confidence': 0.94,
        'imagesProcessed': 60,
        'processingTime': 45.3
    }), 200


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    try:
        db = get_db()
        db.command('ping')
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)