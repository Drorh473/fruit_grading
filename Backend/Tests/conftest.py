"""
Pytest configuration and fixtures for test suite
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from PIL import Image
from flask import Flask


# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Add Tests directory to path
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


# Import test configuration
from test_config import TestConfig


# ==================== Flask Application Fixtures ====================

@pytest.fixture
def app():
    """Create Flask app for testing with all blueprints registered"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Set test configuration - Use getattr with defaults to avoid AttributeError
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    app.config['DB_NAME'] = getattr(TestConfig, 'TEST_DB_NAME', 'test_fruit_grading')
    app.config['MONGO_CONNECTION_STRING'] = getattr(TestConfig, 'MONGO_CONNECTION_STRING', 'mongodb://localhost:27017')
    
    # Handle directory paths with fallback
    app.config['STORED_DATASET_PATH'] = str(getattr(TestConfig, 'TEMP_STORED_DATASET_DIR', Path('/tmp/stored_dataset')))
    app.config['ORIGINAL_DATASET_PATH'] = str(getattr(TestConfig, 'TEMP_ORIGINAL_DATASET_DIR', Path('/tmp/original_dataset')))
    app.config['PROCESSED_DATASET_PATH'] = str(getattr(TestConfig, 'TEMP_PROCESSED_DATASET_DIR', Path('/tmp/processed_dataset')))
    
    app.config['CAMERA_FPS'] = 30
    app.config['NUM_OF_CAMERAS'] = 4
    app.config['BATCH_SIZE'] = 32
    app.config['MODEL_DIR'] = 'saved_models'
    
    # Try to import and register blueprints (with error handling)
    try:
        from routes.add_fruit import add_fruit_bp
        app.register_blueprint(add_fruit_bp, url_prefix='/api/fruits')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import add_fruit_bp: {e}")
    
    try:
        from routes.camera_monitor import cameras_bp
        app.register_blueprint(cameras_bp, url_prefix='/api/cameras')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import cameras_bp: {e}")
    
    try:
        from routes.processing import processing_bp
        app.register_blueprint(processing_bp, url_prefix='/api/pipeline')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import processing_bp: {e}")
    
    try:
        from routes.results import results_bp
        app.register_blueprint(results_bp, url_prefix='/api/results')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import results_bp: {e}")
    
    try:
        from routes.settings import settings_bp
        app.register_blueprint(settings_bp, url_prefix='/api/settings')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import settings_bp: {e}")
    
    try:
        from routes.user_dashboard import user_dashboard_bp
        app.register_blueprint(user_dashboard_bp, url_prefix='/api/user')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import user_dashboard_bp: {e}")
    
    try:
        from routes.admin_dashboard import admin_dashboard_bp
        app.register_blueprint(admin_dashboard_bp, url_prefix='/api/admin')
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import admin_dashboard_bp: {e}")
    
    return app


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


# ==================== Database Fixtures ====================

@pytest.fixture(scope='session')
def mongo_client():
    """Provide MongoDB client for testing"""
    connection_string = getattr(TestConfig, 'MONGO_CONNECTION_STRING', 'mongodb://localhost:27017')
    client = MongoClient(connection_string)
    yield client
    client.close()


@pytest.fixture
def test_db(mongo_client):
    """Provide test database"""
    db_name = getattr(TestConfig, 'TEST_DB_NAME', 'test_fruit_grading')
    db = mongo_client[db_name]
    yield db
    mongo_client.drop_database(db_name)


@pytest.fixture
def test_collection(test_db):
    """Provide test collection"""
    collection_name = getattr(TestConfig, 'TEST_COLLECTION_NAME', 'images')
    collection = test_db[collection_name]
    yield collection
    collection.drop()


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
    camera_angles = getattr(TestConfig, 'CAMERA_ANGLES', ['Front View', 'Right View', 'Back View', 'Left View'])
    return [
        {
            'fruit_type': 'market',
            'object_id': 'obj0001',
            'camera_id': i,
            'camera_angle': camera_angles[i],
            'set_type': 'train',
            'file_path': f'/test/path/image_{i}.jpg',
            'file_name': f'image_{i}.jpg',
            'timestamp': '2024-01-01 12:00:00'
        }
        for i in range(4)
    ]


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
    # Check if TestConfig has create_temp_dirs method
    if hasattr(TestConfig, 'create_temp_dirs'):
        TestConfig.create_temp_dirs()
    yield
    # Check if TestConfig has cleanup_temp_dirs method
    if hasattr(TestConfig, 'cleanup_temp_dirs'):
        TestConfig.cleanup_temp_dirs()


@pytest.fixture
def temp_test_dir(tmp_path):
    """Provide temporary directory for test files"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Reset pipeline state before each test"""
    try:
        from shared_state import pipeline_state
        pipeline_state.reset_pipeline()
    except (ImportError, AttributeError):
        pass
    
    yield
    
    # Cleanup after test
    try:
        from shared_state import pipeline_state
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
    
    camera_angles = getattr(TestConfig, 'CAMERA_ANGLES', ['Front View', 'Right View', 'Back View', 'Left View'])
    
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
