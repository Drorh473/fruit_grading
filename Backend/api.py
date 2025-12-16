from flask import Flask, jsonify, request
from flask_cors import CORS
import pymongo
import os
import threading
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from bson import ObjectId
from contextlib import redirect_stdout, redirect_stderr
import traceback

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import build_model functions
from processes.build_model import (
    run_tests, setup_database, preprocess_data, 
    extract_features, train_classifier,
    generate_confusion_matrix
)

# Flask app setup
app = Flask(__name__)
CORS(app)

# MongoDB connection
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')

# Pipeline state (in production, use Redis or similar)
pipeline_state = {
    'running': False,
    'status': 'idle',  # idle, running, completed, failed
    'currentStep': 0,
    'progress': 0,
    'logs': [],
    'steps': [
        {'id': 1, 'name': 'Database Setup', 'status': 'pending'},
        {'id': 2, 'name': 'Data Preprocessing', 'status': 'pending'},
        {'id': 3, 'name': 'Feature Extraction', 'status': 'pending'},
        {'id': 4, 'name': 'Model Training', 'status': 'pending'},
        {'id': 5, 'name': 'Evaluation', 'status': 'pending'}
    ],
    'config': {
        'hiddenDim': 16,
        'epochs': 100,
        'learningRate': 0.0005,
        'lambdaReg': 0.001,
        'batchSize': 32
    },
    'results': None,
    'pipeline_thread': None
}

def get_db():
    """Get database connection"""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    return client[DB_NAME]

def add_log(message, log_type='info'):
    """Add log entry to pipeline state"""
    pipeline_state['logs'].append({
        'message': message,
        'type': log_type,
        'timestamp': datetime.now().isoformat()
    })
    print(f"[{log_type.upper()}] {message}")

def update_step(step_id, status):
    """Update step status"""
    for step in pipeline_state['steps']:
        if step['id'] == step_id:
            step['status'] = status
            break
    
    # Update current step and progress
    pipeline_state['currentStep'] = step_id
    pipeline_state['progress'] = int((step_id / len(pipeline_state['steps'])) * 100)

def run_pipeline_background(config):
    """Run the complete ML pipeline in background thread"""
    try:
        add_log("Starting ML pipeline...", 'info')
        pipeline_state['status'] = 'running'
        
        # Extract config
        skip_tests = config.get('skipTests', True)
        hidden_dim = config.get('hiddenDim', 16)
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learningRate', 0.0005)
        lambda_reg = config.get('lambdaReg', 0.001)
        
        # Store config
        pipeline_state['config'] = {
            'hiddenDim': hidden_dim,
            'epochs': epochs,
            'learningRate': learning_rate,
            'lambdaReg': lambda_reg,
            'batchSize': config.get('batchSize', 32)
        }
        
        # Step 0: Run tests (optional)
        if not skip_tests:
            add_log("Running test suite...", 'info')
            test_success = run_tests()
            if not test_success:
                add_log("Tests failed! Aborting pipeline.", 'error')
                pipeline_state['status'] = 'failed'
                return
        
        # Step 1: Database Setup
        update_step(1, 'processing')
        add_log("Step 1/5: Setting up database...", 'info')
        
        stored_exists = os.path.exists(STORED_DATASET_PATH) if STORED_DATASET_PATH else False
        if stored_exists:
            add_log(f"Using existing dataset at: {STORED_DATASET_PATH}", 'info')
            update_step(1, 'completed')
        else:
            if not setup_database():
                add_log("Database setup failed!", 'error')
                update_step(1, 'failed')
                pipeline_state['status'] = 'failed'
                return
            update_step(1, 'completed')
            add_log("Database setup complete", 'success')
        
        # Step 2: Preprocessing
        update_step(2, 'processing')
        add_log("Step 2/5: Preprocessing data...", 'info')
        
        processed_exists = os.path.exists(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH else False
        if processed_exists:
            add_log(f"Using existing preprocessed data at: {PROCESSED_DATASET_PATH}", 'info')
        
        train_gen, test_gen = preprocess_data()
        if not train_gen or not test_gen:
            add_log("Preprocessing failed!", 'error')
            update_step(2, 'failed')
            pipeline_state['status'] = 'failed'
            return
        
        update_step(2, 'completed')
        add_log(f"Preprocessing complete - Train: {train_gen.num_batches}, Test: {test_gen.num_batches}", 'success')
        
        # Step 3: Feature Extraction
        update_step(3, 'processing')
        add_log("Step 3/5: Extracting features...", 'info')
        
        train_features, test_features = extract_features(train_gen, test_gen)
        if not train_features or not test_features:
            add_log("Feature extraction failed!", 'error')
            update_step(3, 'failed')
            pipeline_state['status'] = 'failed'
            return
        
        update_step(3, 'completed')
        add_log(f"Feature extraction complete - {len(train_features) + len(test_features)} vectors", 'success')
        
        # Step 4: Model Training
        update_step(4, 'processing')
        add_log(f"Step 4/5: Training classifier (hidden_dim={hidden_dim}, epochs={epochs}, lr={learning_rate}, lambda={lambda_reg})...", 'info')
        
        params, results = train_classifier(
            train_features, test_features,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg
        )
        
        if params is None:
            add_log("Classifier training failed!", 'error')
            update_step(4, 'failed')
            pipeline_state['status'] = 'failed'
            return
        
        update_step(4, 'completed')
        add_log(f"Training complete - Test Accuracy: {results['test_accuracy']*100:.2f}%", 'success')
        
        # Step 5: Evaluation
        update_step(5, 'processing')
        add_log("Step 5/5: Generating confusion matrix...", 'info')
        
        cm = generate_confusion_matrix(results)
        if cm is not None:
            results['confusion_matrix'] = cm
        
        update_step(5, 'completed')
        add_log("Evaluation complete", 'success')
        
        # Store results
        pipeline_state['results'] = {
            'train_accuracy': float(results['train_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'train_loss': float(results['train_loss']),
            'test_loss': float(results['test_loss']),
            'totalProcessed': len(train_features) + len(test_features),
            'timestamp': datetime.now().isoformat()
        }
        
        # Complete
        pipeline_state['status'] = 'completed'
        pipeline_state['progress'] = 100
        add_log(f"Pipeline completed successfully! Test Accuracy: {results['test_accuracy']*100:.2f}%", 'success')
        
    except Exception as e:
        add_log(f"Pipeline error: {str(e)}", 'error')
        add_log(traceback.format_exc(), 'error')
        pipeline_state['status'] = 'failed'
        
        # Mark current step as failed
        if pipeline_state['currentStep'] > 0:
            update_step(pipeline_state['currentStep'], 'failed')
    
    finally:
        pipeline_state['running'] = False

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
    # Return from latest pipeline results if available
    if pipeline_state['results'] and 'confusion_matrix' in pipeline_state['results']:
        cm = pipeline_state['results']['confusion_matrix']
        # Format for frontend
        return jsonify({
            'classes': ['market', 'standard', 'premium'],
            'matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
            'metrics': pipeline_state['results'].get('metrics', {})
        }), 200
    
    # Default mock data
    return jsonify({
        'classes': ['market', 'standard', 'premium'],
        'matrix': [[8, 1, 0], [2, 6, 1], [0, 2, 8]],
        'metrics': {
            'market': {'precision': 0.80, 'recall': 0.89, 'f1': 0.84},
            'standard': {'precision': 0.67, 'recall': 0.67, 'f1': 0.67},
            'premium': {'precision': 0.89, 'recall': 0.80, 'f1': 0.84}
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
    
    return jsonify({
        'database': db_status,
        'pipeline': pipeline_state['status'],
        'cameras': 'operational'
    }), 200


@app.route('/api/admin/dashboard-stats', methods=['GET'])
def get_admin_dashboard_stats():
    """Get statistics for admin dashboard"""
    try:
        db = get_db()
        collection = db['images']
        
        # Get basic counts
        total_processed = collection.count_documents({})
        
        # Count by type
        pipeline = [
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]
        
        type_counts = list(collection.aggregate(pipeline))
        
        # Get recent results
        recent = collection.find().sort('timestamp', -1).limit(100)
        accuracy = 0.92  # Mock - calculate from actual results
        
        stats = {
            'totalProcessed': total_processed,
            'accuracy': accuracy,
            'marketCount': next((r['count'] for r in type_counts if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in type_counts if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0),
            'rejectCount': next((r['count'] for r in type_counts if r['_id'] == 'reject'), 0),
            'processingSpeed': 1.2,  # seconds per fruit
            'uptime': '24h'
        }
        
        # Add pipeline results if available
        if pipeline_state['results']:
            stats['accuracy'] = pipeline_state['results']['test_accuracy']
            stats['totalProcessed'] = pipeline_state['results']['totalProcessed']
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    model_info = {
        'architecture': 'ShuffleNetV2 + FC',
        'inputDim': 200704,
        'hiddenDim': pipeline_state['config']['hiddenDim'],
        'classes': 3,
        'lastTrained': pipeline_state['results']['timestamp'] if pipeline_state['results'] else None,
        'accuracy': pipeline_state['results']['test_accuracy'] if pipeline_state['results'] else None
    }
    
    return jsonify(model_info), 200


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
            'status': True,
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
        return jsonify({'success': False, 'error': 'Pipeline already running'}), 400
    
    config = request.get_json() or {}
    
    # Reset pipeline state
    pipeline_state['running'] = True
    pipeline_state['status'] = 'running'
    pipeline_state['currentStep'] = 0
    pipeline_state['progress'] = 0
    pipeline_state['logs'] = []
    pipeline_state['results'] = None
    
    # Reset step statuses
    for step in pipeline_state['steps']:
        step['status'] = 'pending'
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline_background, args=(config,), daemon=True)
    thread.start()
    pipeline_state['pipeline_thread'] = thread
    
    return jsonify({
        'success': True,
        'pipelineId': 'pipeline_' + datetime.now().strftime('%Y%m%d%H%M%S'),
        'status': 'started'
    }), 200


@app.route('/api/pipeline/stop', methods=['POST'])
def stop_pipeline():
    """Stop pipeline"""
    global pipeline_state
    
    pipeline_state['running'] = False
    pipeline_state['status'] = 'stopped'
    add_log('Pipeline stopped by user', 'warning')
    
    return jsonify({'success': True}), 200


@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get pipeline status"""
    response = {
        'running': pipeline_state['running'],
        'status': pipeline_state['status'],
        'currentStep': pipeline_state['currentStep'],
        'progress': pipeline_state['progress'],
        'steps': pipeline_state['steps']
    }
    
    # Add results if completed
    if pipeline_state['status'] == 'completed' and pipeline_state['results']:
        response['totalProcessed'] = pipeline_state['results']['totalProcessed']
        response['accuracy'] = pipeline_state['results']['test_accuracy']
    
    return jsonify(response), 200


@app.route('/api/pipeline/logs', methods=['GET'])
def get_pipeline_logs():
    """Get pipeline logs"""
    limit = int(request.args.get('limit', 100))
    return jsonify(pipeline_state['logs'][-limit:]), 200


@app.route('/api/pipeline/config', methods=['GET'])
def get_pipeline_config():
    """Get pipeline configuration"""
    return jsonify(pipeline_state['config']), 200


@app.route('/api/pipeline/config', methods=['PUT'])
def update_pipeline_config():
    """Update pipeline configuration"""
    config = request.get_json()
    
    # Update config
    if 'hiddenDim' in config:
        pipeline_state['config']['hiddenDim'] = config['hiddenDim']
    if 'epochs' in config:
        pipeline_state['config']['epochs'] = config['epochs']
    if 'learningRate' in config:
        pipeline_state['config']['learningRate'] = config['learningRate']
    if 'lambdaReg' in config:
        pipeline_state['config']['lambdaReg'] = config['lambdaReg']
    if 'batchSize' in config:
        pipeline_state['config']['batchSize'] = config['batchSize']
    
    return jsonify(pipeline_state['config']), 200


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
        'hiddenDim': pipeline_state['config']['hiddenDim'],
        'learningRate': pipeline_state['config']['learningRate'],
        'epochs': pipeline_state['config']['epochs'],
        'lambdaReg': pipeline_state['config']['lambdaReg']
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