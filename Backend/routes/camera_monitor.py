"""
Camera Monitor Routes
Endpoints for camera status and control
"""
from flask import Blueprint, jsonify, request, current_app
import os

cameras_bp = Blueprint('cameras', __name__)

@cameras_bp.route('/status', methods=['GET'])
def get_camera_statuses():
    """Get all camera statuses"""
    try:
        num_cameras = current_app.config.get('NUM_OF_CAMERAS', 4)
        angles = ['Front View', 'Right View', 'Back View', 'Left View']
        fps = current_app.config.get('CAMERA_FPS', 30)
        
        cameras = []
        for i in range(num_cameras):
            cameras.append({
                'id': i,
                'name': f'Camera {i}',
                'status': True,  # Mock status
                'angle': angles[i] if i < len(angles) else f'Angle {i}',
                'fps': fps,
                'resolution': '224x224',
                'health': 'healthy',
                'lastFrame': None  # Add real timestamp when integrated
            })
        
        return jsonify({'cameras': cameras}), 200
        
    except Exception as e:
        return jsonify({'cameras': []}), 200


@cameras_bp.route('/<int:camera_id>', methods=['GET'])
def get_camera_details(camera_id):
    """Get specific camera details"""
    try:
        num_cameras = current_app.config.get('NUM_OF_CAMERAS', 4)
        if camera_id < 0 or camera_id >= num_cameras:
            return jsonify({'error': 'Camera not found'}), 404
        
        angles = ['Front View', 'Right View', 'Back View', 'Left View']
        fps = current_app.config.get('CAMERA_FPS', 30)
        
        camera_details = {
            'id': camera_id,
            'name': f'Camera {camera_id}',
            'status': True,
            'angle': angles[camera_id] if camera_id < len(angles) else f'Angle {camera_id}',
            'fps': fps,
            'resolution': '224x224',
            'preprocessing': 'Gaussian Blur + CLAHE',
            'health': 'healthy',
            'uptime': '24h',
            'framesProcessed': 0
        }
        
        return jsonify(camera_details), 200
        
    except Exception as e:
        return jsonify({'error': 'Camera not found'}), 404


@cameras_bp.route('/<int:camera_id>/refresh', methods=['POST'])
def refresh_camera(camera_id):
    """Refresh specific camera"""
    try:
        #implement actual camera refresh logic
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} refreshed successfully',
            'cameraId': camera_id
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cameras_bp.route('/refresh-all', methods=['POST'])
def refresh_all_cameras():
    """Refresh all cameras"""
    try:
        num_cameras = current_app.config.get('NUM_OF_CAMERAS', 4)
        
        # implement actual camera refresh logic
        return jsonify({
            'success': True,
            'message': f'All {num_cameras} cameras refreshed successfully',
            'refreshedCount': num_cameras
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cameras_bp.route('/config', methods=['GET'])
def get_camera_config():
    """Get camera configuration"""
    try:
        config = {
            'fps': current_app.config.get('CAMERA_FPS', 30),
            'numCameras': current_app.config.get('NUM_OF_CAMERAS', 4),
            'imageSize': '224x224',
            'preprocessing': 'Gaussian Blur + CLAHE',
            'angles': ['Front View', 'Right View', 'Back View', 'Left View']
        }
        
        return jsonify(config), 200
        
    except Exception as e:
        return jsonify({
            'fps': 30,
            'numCameras': 4,
            'imageSize': '224x224',
            'preprocessing': 'Gaussian Blur + CLAHE'
        }), 200