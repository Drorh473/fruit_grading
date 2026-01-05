"""
Flask App with Integrated Testing
Uses comprehensive pre-pipeline test validation
"""
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


def run_tests(mode='full', verbose=True):
    """
    Run test suite before server startup
    
    Args:
        mode: 'full', 'critical', or phase number (1-5)
        verbose: Output verbosity
    
    Returns:
        bool: True if tests passed
    """
    orchestrator = TestOrchestrator()
    
    if mode == 'critical':
        print("\nRunning critical tests only...")
        return orchestrator.run_critical_only(verbose=verbose)
    elif mode == 'full':
        print("\nRunning full test suite...")
        return orchestrator.run_all(verbose=verbose, stop_on_failure=True)
    elif isinstance(mode, int) and 1 <= mode <= 5:
        print(f"\nRunning Phase {mode} only...")
        success = orchestrator.run_phase(mode, verbose=verbose)
        orchestrator._print_summary(success, mode=f"phase-{mode}")
        return success
    else:
        print(f"Invalid mode: {mode}")
        return False


def create_app():
    """Create Flask app"""
    app = Flask(__name__)
    
    # Config
    app.config['MONGO_CONNECTION_STRING'] = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    app.config['DB_NAME'] = os.getenv('DB_NAME', 'fruit_grading')
    app.config['STORED_DATASET_PATH'] = os.getenv('STORED_DATASET_PATH')
    app.config['ORIGINAL_DATASET_PATH'] = os.getenv('ORIGINAL_DATASET_PATH')
    app.config['PROCESSED_DATASET_PATH'] = os.getenv('PROCESSED_DATASET_PATH')
    app.config['NUM_OF_CAMERAS'] = int(os.getenv('NUM_OF_CAMERAS', 4))
    app.config['CAMERA_FPS'] = int(os.getenv('CAMERA_FPS', 30))
    app.config['BATCH_SIZE'] = int(os.getenv('BATCH_SIZE', 128))
    
    CORS(app)
    
    # MongoDB
    try:
        app.mongo_client = pymongo.MongoClient(
            app.config['MONGO_CONNECTION_STRING'],
            serverSelectionTimeoutMS=5000
        )
        app.mongo_client.server_info()
        print(f"MongoDB connected: {app.config['DB_NAME']}")
    except Exception as e:
        print(f"MongoDB failed: {e}")
        app.mongo_client = None
    
    # Register routes
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
    
    return app


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Start Flask Backend with Testing')
    parser.add_argument('--skip-tests', action='store_true', help='Skip all tests')
    parser.add_argument('--critical-test', action='store_true', help='Run critical tests only')
    parser.add_argument('--continue-on-test-failure', action='store_true', help='Start even if tests fail')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    args = parser.parse_args()
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    if not args.skip_tests and not is_reloader_process:
        print("\n" + "="*70)
        print("PRE-STARTUP VALIDATION")
        print("="*70)
        
        if args.critical_test:
            test_mode = 'critical'
        else:
            test_mode = 'full'
        
        tests_passed = run_tests(mode=test_mode, verbose=True)
        
        if not tests_passed:
            print("\nTests failed!")
            if not args.continue_on_test_failure:
                print("Server will NOT start")
                sys.exit(1)
        else:
            print("\nAll tests passed")
    elif is_reloader_process:
        print("\n[Reloader] Skipping tests on restart...")
    
    # Start server
    print("\n" + "="*70)
    print("STARTING FLASK SERVER")
    print("="*70)
    
    app = create_app()
    
    print(f"\nServer: {args.host}:{args.port}")
    print(f"Debug: {not args.no_debug}")
    print(f"URL: http://localhost:{args.port}\n")
    
    app.run(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == '__main__':
    main()