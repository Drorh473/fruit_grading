"""
Health Check Routes
Endpoints for system health monitoring
"""
from flask import Blueprint, jsonify
from utils import check_db_connection
from shared_state import pipeline_state

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check"""
    try:
        db_connected = check_db_connection()
        
        if db_connected:
            return jsonify({
                'status': 'healthy',
                'database': 'connected',
                'timestamp': pipeline_state.get_state()['logs'][-1]['timestamp'] if pipeline_state.get_state()['logs'] else None
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'database': 'disconnected',
                'error': 'Database connection failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@health_bp.route('/status', methods=['GET'])
def system_status():
    """Detailed system status"""
    try:
        db_connected = check_db_connection()
        state = pipeline_state.get_state()
        
        status = {
            'healthy': db_connected,
            'components': {
                'database': 'connected' if db_connected else 'disconnected',
                'pipeline': state['status'],
                'model': 'loaded' if state['results'] else 'not_trained'
            },
            'pipeline': {
                'running': state['running'],
                'currentStep': state['currentStep'],
                'progress': state['progress']
            }
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        print(f"Error in system_status: {e}")
        return jsonify({
            'healthy': False,
            'error': str(e)
        }), 500


@health_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return jsonify({'status': 'ok'}), 200