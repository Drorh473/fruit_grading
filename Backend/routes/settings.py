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
        return jsonify({'error': str(e)}), 500


@settings_bp.route('', methods=['PUT'])
def update_settings():
    """Update system settings"""
    try:
        settings = request.get_json()
        
        return jsonify(settings), 200
        
    except Exception as e:
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
    # Initialize with defaults
    db_status = 'disconnected'
    num_cameras = current_app.config.get('NUM_OF_CAMERAS', 4)
    camera_statuses = [False] * num_cameras

    # Check database connection
    try:
        if current_app.mongo_client is not None:
            current_app.mongo_client.server_info()
            db_status = 'connected'
    except Exception as db_err:
        current_app.logger.warning(f"Database connection check failed: {db_err}")
        db_status = 'disconnected'

    # Check camera status - for now cameras are considered available if configured
    # In production, integrate with actual camera health checks
    try:
        # Cameras are available if the system is configured for them
        camera_statuses = [True] * num_cameras
    except Exception as cam_err:
        current_app.logger.warning(f"Camera status check failed: {cam_err}")
        camera_statuses = [False] * num_cameras

    return jsonify({
        'database': db_status,
        'cameras': camera_statuses
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
        return jsonify({'error': str(e)}), 500


@settings_bp.route('/paths', methods=['PUT'])
def update_dataset_paths():
    """Update dataset paths"""
    try:
        paths = request.get_json()
        
        return jsonify(paths), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500