"""
Settings Routes
Endpoints for system settings and configuration
"""
from flask import Blueprint, jsonify, request, current_app
import pymongo
import os

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('', methods=['GET'])
def get_settings():
    """Get all system settings"""
    try:
        settings = {
            'dbName': current_app.config.get('DB_NAME'),
            'mongoConnection': current_app.config.get('MONGO_CONNECTION_STRING'),
            'storedDataset': current_app.config.get('STORED_DATASET_PATH'),
            'originalDataset': current_app.config.get('ORIGINAL_DATASET_PATH'),
            'processedDataset': current_app.config.get('PROCESSED_DATASET_PATH'),
            'cameraFps': current_app.config.get('CAMERA_FPS'),
            'numCameras': current_app.config.get('NUM_OF_CAMERAS'),
            'imageSize': '224x224',
            'batchSize': current_app.config.get('BATCH_SIZE')
        }
        
        return jsonify(settings), 200
        
    except Exception as e:
        print(f"Error in get_settings: {e}")
        return jsonify({'error': str(e)}), 500


@settings_bp.route('', methods=['PUT'])
def update_settings():
    """Update system settings"""
    try:
        settings = request.get_json()
        
        # In production, update .env file or configuration storage
        # For now, just return the updated settings
        
        return jsonify(settings), 200
        
    except Exception as e:
        print(f"Error in update_settings: {e}")
        return jsonify({'error': str(e)}), 500


@settings_bp.route('/test-database', methods=['POST'])
def test_database():
    """Test database connection"""
    try:
        data = request.get_json()
        connection_string = data.get('connectionString')
        
        if not connection_string:
            return jsonify({
                'success': False,
                'message': 'Connection string is required'
            }), 400
        
        # Try to connect
        client = pymongo.MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000
        )
        client.server_info()
        
        return jsonify({
            'success': True,
            'message': 'Connection successful'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 200


@settings_bp.route('/status', methods=['GET'])
def get_settings_status():
    """Get system status for settings page"""
    try:
        from Backend.utils.utils import check_db_connection
        from Backend.utils.shared_state import pipeline_state
        
        db_status = 'connected' if check_db_connection() else 'disconnected'
        state = pipeline_state.get_state()
        
        # Get number of cameras from config
        num_cameras = current_app.config.get('NUM_OF_CAMERAS', 4)
        
        # For now, mock all cameras as operational (True)
        # In production, integrate with actual camera status checking
        camera_statuses = [True] * num_cameras
        
        # Check if model exists on disk
        model_dir = current_app.config.get('MODEL_DIR', 'saved_models')
        model_status = 'not_trained'
        
        # Check for runtime results first
        if state['results']:
            model_status = 'loaded'
        # Check for saved model files
        elif os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) 
                          if f.endswith('.pth') or f.endswith('.pkl')]
            if model_files:
                model_status = 'trained'
        
        return jsonify({
            'database': db_status,
            'model': model_status,
            'pipeline': state['status'],
            'cameras': camera_statuses
        }), 200
        
    except Exception as e:
        print(f"Error in get_settings_status: {e}")
        return jsonify({
            'database': 'disconnected',
            'model': 'unknown',
            'pipeline': 'idle',
            'cameras': [False, False, False, False]
        }), 200


@settings_bp.route('/paths', methods=['GET'])
def get_dataset_paths():
    """Get dataset paths"""
    try:
        paths = {
            'storedDataset': current_app.config.get('STORED_DATASET_PATH'),
            'originalDataset': current_app.config.get('ORIGINAL_DATASET_PATH'),
            'processedDataset': current_app.config.get('PROCESSED_DATASET_PATH')
        }
        
        return jsonify(paths), 200
        
    except Exception as e:
        print(f"Error in get_dataset_paths: {e}")
        return jsonify({'error': str(e)}), 500


@settings_bp.route('/paths', methods=['PUT'])
def update_dataset_paths():
    """Update dataset paths"""
    try:
        paths = request.get_json()
        
        # In production, update .env file
        # For now, just return the paths
        
        return jsonify(paths), 200
        
    except Exception as e:
        print(f"Error in update_dataset_paths: {e}")
        return jsonify({'error': str(e)}), 500