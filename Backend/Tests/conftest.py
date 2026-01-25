import sys
import os
import pytest
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from PIL import Image
from flask import Flask

# CRITICAL: Set test environment BEFORE any other imports
os.environ['TESTING'] = 'true'
os.environ['DB_NAME'] = 'test_fruit_grading'

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add Tests directory to path
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import test configuration (this also loads .env.test)
from Tests.test_config import TestConfig, ensure_test_environment

# Ensure test environment is active
ensure_test_environment()

def _verify_not_production_db(db_name):
    """Safety check to prevent accidental production database access"""
    production_names = ['fruit_grading', 'fruit_rotation_db', 'production', 'prod']
    if db_name.lower() in [n.lower() for n in production_names]:
        raise RuntimeError(
            f"CRITICAL: Attempted to access production database '{db_name}' in tests! "
            "Check your test configuration."
        )


# ==================== Flask Application Fixtures ====================

@pytest.fixture
def app():
    """Create Flask app for testing with all blueprints registered"""
    # Verify test mode
    ensure_test_environment()
    
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Set test configuration - ALWAYS use test database
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    app.config['DB_NAME'] = TestConfig.TEST_DB_NAME  # Force test DB
    app.config['MONGO_CONNECTION_STRING'] = TestConfig.MONGO_CONNECTION_STRING
    
    # Verify not production
    _verify_not_production_db(app.config['DB_NAME'])
    
    # Handle directory paths with fallback
    app.config['STORED_DATASET_PATH'] = str(TestConfig.TEMP_STORED_DATASET_DIR)
    app.config['ORIGINAL_DATASET_PATH'] = str(TestConfig.TEMP_ORIGINAL_DATASET_DIR)
    app.config['PROCESSED_DATASET_PATH'] = str(TestConfig.TEMP_PROCESSED_DATASET_DIR)
    
    app.config['CAMERA_FPS'] = 30
    app.config['NUM_OF_CAMERAS'] = 4
    app.config['BATCH_SIZE'] = 32
    app.config['MODEL_DIR'] = '/tmp/test_models'
    
    try:
        app.mongo_client = MongoClient(
            app.config['MONGO_CONNECTION_STRING'],
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        app.mongo_client.server_info()
        print(f"[Test] MongoDB connected to TEST database: {app.config['DB_NAME']}")
    except Exception as e:
        print(f"[Test] MongoDB connection failed: {e}")
        app.mongo_client = None
    
    # Try to import and register blueprints (with error handling)
    blueprint_imports = [
        ('routes.add_fruit', 'add_fruit_bp', '/api/fruits'),
        ('routes.camera_monitor', 'cameras_bp', '/api/cameras'),
        ('routes.processing', 'processing_bp', '/api/pipeline'),
        ('routes.results', 'results_bp', '/api/results'),
        ('routes.settings', 'settings_bp', '/api/settings'),
        ('routes.user_dashboard', 'user_dashboard_bp', '/api/user'),
        ('routes.admin_dashboard', 'admin_dashboard_bp', '/api/admin'),
    ]
    
    for module_name, bp_name, url_prefix in blueprint_imports:
        try:
            module = __import__(module_name, fromlist=[bp_name])
            bp = getattr(module, bp_name)
            app.register_blueprint(bp, url_prefix=url_prefix)
        except (ImportError, AttributeError) as e:
            print(f"[Test] Warning: Could not import {bp_name}: {e}")
    
    yield app

    # Cleanup: Drop test database collections
    if hasattr(app, 'mongo_client') and app.mongo_client:
        try:
            db = app.mongo_client[app.config['DB_NAME']]
            # Only clean if it's the test database
            if app.config['DB_NAME'] == TestConfig.TEST_DB_NAME:
                db.images.delete_many({})
                db.classification_results.delete_many({})
                db.test_images.delete_many({})
        except Exception as e:
            print(f"[Test] Cleanup warning: {e}")
        finally:
            app.mongo_client.close()


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


# ==================== Database Fixtures ====================

@pytest.fixture(scope='session')
def mongo_client():
    """Provide MongoDB client for testing"""
    ensure_test_environment()

    connection_string = TestConfig.MONGO_CONNECTION_STRING
    # Use shorter timeout for faster failure detection
    client = MongoClient(
        connection_string,
        serverSelectionTimeoutMS=5000,  # 5 second timeout
        connectTimeoutMS=5000
    )

    # Verify connection immediately
    try:
        client.admin.command('ping')
        print(f"[Test] MongoDB connected successfully")
    except Exception as e:
        print(f"[Test] MongoDB connection failed: {e}")
        raise

    yield client
    
    # Session cleanup: drop entire test database
    try:
        _verify_not_production_db(TestConfig.TEST_DB_NAME)
        client.drop_database(TestConfig.TEST_DB_NAME)
        print(f"[Test Session] Dropped test database: {TestConfig.TEST_DB_NAME}")
    except Exception as e:
        print(f"[Test Session] Database cleanup warning: {e}")
    finally:
        client.close()


@pytest.fixture
def test_db(mongo_client):
    """Provide test database - automatically cleaned after each test"""
    db_name = TestConfig.TEST_DB_NAME
    _verify_not_production_db(db_name)
    
    db = mongo_client[db_name]
    
    yield db
    
    # Cleanup after each test
    try:
        mongo_client.drop_database(db_name)
    except Exception as e:
        print(f"[Test] Database cleanup warning: {e}")


@pytest.fixture
def test_collection(test_db):
    """Provide test collection - automatically cleaned after each test"""
    collection_name = TestConfig.TEST_COLLECTION_NAME
    collection = test_db[collection_name]
    
    yield collection
    
    # Cleanup
    try:
        collection.drop()
    except Exception as e:
        print(f"[Test] Collection cleanup warning: {e}")


# ==================== Image Fixtures ====================

@pytest.fixture
def sample_image():
    """Provide sample numpy image array"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_metadata():
    """Provide sample image metadata for testing"""
    return {
        'fruit_type': 'market',
        'object_id': 'obj0001',
        'camera_id': 1,
        'camera_angle': 'Front View',
        'set_type': 'train',
        'file_path': '/test/path/image.jpg',
        'file_name': 'image.jpg',
        'timestamp': '2024-01-01 12:00:00'
    }


@pytest.fixture
def sample_image_documents():
    """Provide multiple sample image documents for testing"""
    camera_angles = TestConfig.CAMERA_ANGLES
    # Create documents for multiple objects (3 objects with 4 cameras each = 12 images)
    documents = []
    for obj_num in range(3):  # 3 different objects
        for cam_id in range(4):  # 4 cameras per object
            documents.append({
                'fruit_type': 'market',
                'object_id': f'obj{obj_num:04d}',  # obj0000, obj0001, obj0002
                'camera_id': cam_id,
                'camera_angle': camera_angles[cam_id],
                'set_type': 'train',
                'file_path': f'/test/path/obj{obj_num:04d}_cam{cam_id}.jpg',
                'file_name': f'obj{obj_num:04d}_cam{cam_id}.jpg',
                'timestamp': '2024-01-01 12:00:00'
            })
    return documents


@pytest.fixture
def valid_image_path(tmp_path):
    """Provide path to a valid test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_path = tmp_path / 'valid_image.jpg'
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def corrupted_image_path(tmp_path):
    """Provide path to a corrupted image file"""
    img_path = tmp_path / 'corrupted_image.jpg'
    img_path.write_bytes(b'not an image')
    return str(img_path)


@pytest.fixture
def multiple_valid_images(tmp_path):
    """Provide multiple valid test images for camera testing"""
    image_paths = []
    for i in range(4):
        img = Image.new('RGB', (224, 224), color=['red', 'green', 'blue', 'yellow'][i])
        img_path = tmp_path / f'camera_{i}.jpg'
        img.save(img_path)
        image_paths.append(str(img_path))
    return image_paths


# ==================== Model Fixtures ====================

@pytest.fixture
def mock_model_parameters():
    """Provide mock neural network parameters"""
    input_dim = 100
    hidden_dim = 16
    num_classes = 3
    
    return {
        'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.random.randn(hidden_dim, num_classes) * 0.01,
        'b2': np.zeros((1, num_classes))
    }


# ==================== Environment Fixtures ====================

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Setup test environment before all tests"""
    # Ensure test mode
    ensure_test_environment()
    
    # Create temp directories
    TestConfig.create_temp_dirs()
    
    print(f"\n[Test Session] Started with database: {TestConfig.TEST_DB_NAME}")
    
    yield
    
    # Cleanup temp directories
    TestConfig.cleanup_temp_dirs()
    print(f"\n[Test Session] Completed and cleaned up")


@pytest.fixture
def temp_test_dir(tmp_path):
    """Provide temporary directory for test files"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Reset pipeline state before each test"""
    import time
    try:
        from Backend.utils.shared_state import pipeline_state
        # Wait for any running pipeline to finish
        max_wait = 5  # seconds
        wait_interval = 0.1
        waited = 0
        while pipeline_state.is_running() and waited < max_wait:
            time.sleep(wait_interval)
            waited += wait_interval
        pipeline_state.reset_pipeline()
    except (ImportError, AttributeError):
        pass

    yield

    # Cleanup after test
    try:
        from Backend.utils.shared_state import pipeline_state
        # Wait for any running pipeline to finish
        max_wait = 5  # seconds
        wait_interval = 0.1
        waited = 0
        while pipeline_state.is_running() and waited < max_wait:
            time.sleep(wait_interval)
            waited += wait_interval
        pipeline_state.reset_pipeline()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def cleanup_uploaded_files():
    """Cleanup uploaded files after each test"""
    yield
    
    # Cleanup temporary uploaded files
    try:
        import shutil
        upload_dir = project_root / 'uploads'
        if upload_dir.exists():
            for item in upload_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    except Exception:
        pass  # Silent cleanup


@pytest.fixture(autouse=True)
def enforce_test_database(monkeypatch):
    """
    Automatically patch environment variables for EVERY test
    to ensure test database is always used
    """
    monkeypatch.setenv('DB_NAME', TestConfig.TEST_DB_NAME)
    monkeypatch.setenv('TESTING', 'true')
    
    # Also patch os.getenv to return test values for critical vars
    original_getenv = os.getenv
    
    def patched_getenv(key, default=None):
        if key == 'DB_NAME':
            return TestConfig.TEST_DB_NAME
        if key == 'TESTING':
            return 'true'
        return original_getenv(key, default)
    
    monkeypatch.setattr(os, 'getenv', patched_getenv)


# ==================== API Testing Fixtures ====================

@pytest.fixture
def mock_classification_result():
    """Provide mock classification result"""
    return {
        'object_id': 'test_fruit_001',
        'fruit_type': 'premium',
        'confidence': 0.95,
        'timestamp': '2024-01-01 12:00:00',
        'image_count': 4
    }


@pytest.fixture
def mock_dashboard_stats():
    """Provide mock dashboard statistics"""
    return {
        'totalToday': 100,
        'marketCount': 25,
        'standardCount': 35,
        'premiumCount': 30,
        'rejectCount': 10
    }


@pytest.fixture
def mock_model_metadata():
    """Provide mock model metadata"""
    return {
        'performance': {
            'test_accuracy': 0.92,
            'train_accuracy': 0.95,
            'test_loss': 0.25,
            'train_loss': 0.18
        },
        'dataset_info': {
            'total_objects': 500,
            'train_objects': 400,
            'test_objects': 100
        },
        'timestamp': '2024-01-01T12:00:00',
        'hyperparameters': {
            'hidden_dim': 16,
            'epochs': 100,
            'learning_rate': 0.0005,
            'lambda_reg': 0.001
        }
    }


@pytest.fixture
def api_headers():
    """Provide standard API request headers"""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


# ==================== Database Seeding Fixtures ====================

@pytest.fixture
def seed_test_data(test_collection):
    """Seed database with test data"""
    from datetime import datetime
    
    camera_angles = TestConfig.CAMERA_ANGLES
    
    test_data = [
        {
            'object_id': f'test_obj_{i:03d}',
            'fruit_type': ['market', 'standard', 'premium', 'reject'][i % 4],
            'camera_id': i % 4,
            'angle': camera_angles[i % 4],
            'timestamp': datetime.now(),
            'confidence': 0.85 + (i % 10) * 0.01,
            'image_path': f'/test/path/obj_{i:03d}.jpg',
            'batch_id': f'batch_{i // 10}'
        }
        for i in range(20)
    ]
    
    test_collection.insert_many(test_data)
    yield test_collection
    test_collection.drop()