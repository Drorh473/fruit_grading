"""
Main Flask Application Entry Point
Fruit Grading System Backend
"""
from flask import Flask, g
from flask_cors import CORS
import pymongo
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MONGO_CONNECTION_STRING'] = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    app.config['DB_NAME'] = os.getenv('DB_NAME', 'fruit_grading')
    app.config['STORED_DATASET_PATH'] = os.getenv('STORED_DATASET_PATH')
    app.config['ORIGINAL_DATASET_PATH'] = os.getenv('ORIGINAL_DATASET_PATH')
    app.config['PROCESSED_DATASET_PATH'] = os.getenv('PROCESSED_DATASET_PATH')
    app.config['NUM_OF_CAMERAS'] = int(os.getenv('NUM_OF_CAMERAS', 4))
    app.config['CAMERA_FPS'] = int(os.getenv('CAMERA_FPS', 30))
    app.config['BATCH_SIZE'] = int(os.getenv('BATCH_SIZE', 128))
    
    # Enable CORS
    CORS(app)
    
    # Initialize database connection
    try:
        app.mongo_client = pymongo.MongoClient(
            app.config['MONGO_CONNECTION_STRING'],
            serverSelectionTimeoutMS=5000
        )
        app.mongo_client.server_info()
        print(f" Connected to MongoDB: {app.config['DB_NAME']}")
    except Exception as e:
        print(f" Failed to connect to MongoDB: {e}")
        app.mongo_client = None
    
    # Register blueprints
    from routes.user_dashboard import user_dashboard_bp
    from routes.admin_dashboard import admin_dashboard_bp
    from routes.camera_monitor import cameras_bp
    from routes.processing import processing_bp
    from routes.results import results_bp
    from routes.settings import settings_bp
    from routes.add_fruit import add_fruit_bp
    from routes.health import health_bp
    
    app.register_blueprint(user_dashboard_bp, url_prefix='/api/user')
    app.register_blueprint(admin_dashboard_bp, url_prefix='/api/admin')
    app.register_blueprint(cameras_bp, url_prefix='/api/cameras')
    app.register_blueprint(processing_bp, url_prefix='/api/pipeline')
    app.register_blueprint(results_bp, url_prefix='/api/results')
    app.register_blueprint(settings_bp, url_prefix='/api/settings')
    app.register_blueprint(add_fruit_bp, url_prefix='/api/fruit')
    app.register_blueprint(health_bp, url_prefix='/api')
    
    @app.teardown_appcontext
    def close_db(error):
        """Close database connection"""
        g.pop('db', None)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)