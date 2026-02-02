"""Flask Application with Pre-startup Testing"""
from flask import Flask, g
from flask_cors import CORS
import pymongo
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tests.test_main import TestOrchestrator


def preload_models():
    """Preload ML models into memory for faster inference."""
    try:
        from cnn.pre_trained_feature_map import load_model as load_cnn
        load_cnn()

        from cnn.fully_connected_layer import load_model as load_classifier
        model_path = os.path.join(os.getenv('MODEL_DIR', 'saved_models'), 'fruit_classifier.pkl')
        if os.path.exists(model_path):
            load_classifier(model_path)
    except Exception as e:
        print(f"Model preloading failed: {e}")


def run_tests(mode='full', verbose=True):
    """Run test suite with specified mode."""
    orchestrator = TestOrchestrator()
    if mode == 'critical':
        return orchestrator.run_critical_only(verbose=verbose)
    elif mode == 'full':
        return orchestrator.run_all(verbose=verbose, stop_on_failure=True)
    elif isinstance(mode, int) and 1 <= mode <= 5:
        success = orchestrator.run_phase(mode, verbose=verbose)
        orchestrator._print_summary(success, mode=f"phase-{mode}")
        return success
    return False


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)

    app.config['MONGO_CONNECTION_STRING'] = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    app.config['DB_NAME'] = os.getenv('DB_NAME', 'fruit_grading')
    app.config['STORED_DATASET_PATH'] = os.getenv('STORED_DATASET_PATH')
    app.config['ORIGINAL_DATASET_PATH'] = os.getenv('ORIGINAL_DATASET_PATH')
    app.config['PROCESSED_DATASET_PATH'] = os.getenv('PROCESSED_DATASET_PATH')
    app.config['NUM_OF_CAMERAS'] = int(os.getenv('NUM_OF_CAMERAS', 4))
    app.config['CAMERA_FPS'] = int(os.getenv('CAMERA_FPS', 30))
    app.config['BATCH_SIZE'] = int(os.getenv('BATCH_SIZE', 128))

    CORS(app)

    try:
        app.mongo_client = pymongo.MongoClient(
            app.config['MONGO_CONNECTION_STRING'],
            serverSelectionTimeoutMS=5000
        )
        app.mongo_client.server_info()
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        app.mongo_client = None

    from routes.user_dashboard import user_dashboard_bp
    from routes.admin_dashboard import admin_dashboard_bp
    from routes.camera_monitor import cameras_bp
    from routes.processing import processing_bp
    from routes.results import results_bp
    from routes.settings import settings_bp
    from routes.add_fruit import add_fruit_bp

    app.register_blueprint(user_dashboard_bp, url_prefix='/api/user')
    app.register_blueprint(admin_dashboard_bp, url_prefix='/api/admin')
    app.register_blueprint(cameras_bp, url_prefix='/api/cameras')
    app.register_blueprint(processing_bp, url_prefix='/api/pipeline')
    app.register_blueprint(results_bp, url_prefix='/api/results')
    app.register_blueprint(settings_bp, url_prefix='/api/settings')
    app.register_blueprint(add_fruit_bp, url_prefix='/api/fruit')

    @app.teardown_appcontext
    def close_db(error):
        g.pop('db', None)

    preload_models()
    return app


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(description='Flask Backend Server')
    parser.add_argument('--skip-tests', action='store_true', help='Skip all tests')
    parser.add_argument('--critical-test', action='store_true', help='Run critical tests only')
    parser.add_argument('--continue-on-test-failure', action='store_true', help='Start even if tests fail')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')

    args = parser.parse_args()
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

    if not is_reloader_process:
        preload_models()

    if not args.skip_tests and not is_reloader_process:
        test_mode = 'critical' if args.critical_test else 'full'
        tests_passed = run_tests(mode=test_mode, verbose=True)

        if not tests_passed and not args.continue_on_test_failure:
            sys.exit(1)

    app = create_app()
    app.run(host=args.host, port=args.port, debug=not args.no_debug, use_reloader=False)


if __name__ == '__main__':
    main()