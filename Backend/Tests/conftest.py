"""
Pytest Fixtures
Shared fixtures for all tests
"""
import pytest
import pymongo
import shutil
import numpy as np
import cv2
from pathlib import Path
from test_config import TestConfig


# ==================== Session Fixtures ====================

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Set up test environment before all tests"""
    TestConfig.create_temp_dirs()
    yield
    TestConfig.cleanup_temp_dirs()


@pytest.fixture(scope='session')
def mongo_client():
    """MongoDB client for testing"""
    client = pymongo.MongoClient(
        TestConfig.MONGO_CONNECTION_STRING,
        serverSelectionTimeoutMS=5000
    )
    try:
        client.server_info()
        yield client
    finally:
        client.close()


# ==================== Database Fixtures ====================

@pytest.fixture(scope='function')
def test_db(mongo_client):
    """Provide clean test database for each test"""
    db = mongo_client[TestConfig.TEST_DB_NAME]
    yield db
    # Cleanup: drop database after test
    mongo_client.drop_database(TestConfig.TEST_DB_NAME)


@pytest.fixture(scope='function')
def test_collection(test_db):
    """Provide clean test collection"""
    collection = test_db[TestConfig.TEST_COLLECTION_NAME]
    # Create indexes
    collection.create_index('fruit_type')
    collection.create_index('object_id')
    collection.create_index('set_type')
    collection.create_index('camera_id')
    yield collection
    # Cleanup: clear collection after test
    collection.delete_many({})


@pytest.fixture
def sample_image_metadata():
    """Sample image metadata for testing"""
    return {
        "path": "/test/path/image.png",
        "fruit_type": "market",
        "object_id": "obj0001",
        "camera_id": 0,
        "timestamp": "2025-01-01T00:00:00",
        "width": 224,
        "height": 224,
        "color": 3,
        "set_type": "training",
        "category": "A"
    }


@pytest.fixture
def sample_image_documents():
    """Multiple sample documents for testing"""
    documents = []
    for i in range(20):
        fruit_type = TestConfig.FRUIT_TYPES[i % 3]
        camera_id = i % TestConfig.NUM_OF_CAMERAS
        set_type = "training" if i < 12 else "testing"
        
        documents.append({
            "path": f"/test/path/{fruit_type}_obj{i:04d}_cam{camera_id}.png",
            "processed_path": f"/test/processed/{fruit_type}_obj{i:04d}_cam{camera_id}.png",
            "fruit_type": fruit_type,
            "object_id": f"obj{i:04d}",
            "camera_id": camera_id,
            "timestamp": f"2025-01-01T00:{i:02d}:00",
            "width": 224,
            "height": 224,
            "color": 3,
            "set_type": set_type,
            "category": "A"
        })
    
    return documents


# ==================== Image Fixtures ====================

@pytest.fixture
def sample_image():
    """Generate a sample RGB image"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Generate a sample grayscale image"""
    return np.random.randint(0, 255, (224, 224), dtype=np.uint8)


@pytest.fixture
def sample_batch_images():
    """Generate a batch of sample images"""
    batch_size = TestConfig.BATCH_SIZE
    return np.random.rand(batch_size, 224, 224, 3).astype(np.float32)


@pytest.fixture
def corrupted_image_path(tmp_path):
    """Create a corrupted image file"""
    path = tmp_path / "corrupted.png"
    # Write invalid PNG data
    with open(path, 'wb') as f:
        f.write(b'INVALID_PNG_DATA')
    return path


@pytest.fixture
def valid_image_path(tmp_path, sample_image):
    """Create a valid image file"""
    path = tmp_path / "valid.png"
    cv2.imwrite(str(path), sample_image)
    return path


# ==================== Model Fixtures ====================

@pytest.fixture
def sample_features():
    """Sample feature vector"""
    return np.random.rand(TestConfig.FEATURE_DIM).astype(np.float32)


@pytest.fixture
def sample_feature_batch():
    """Batch of feature vectors"""
    batch_size = 10
    return np.random.rand(batch_size, TestConfig.FEATURE_DIM).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Sample labels for classification"""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


@pytest.fixture
def mock_model_parameters():
    """Mock neural network parameters"""
    from cnn.fully_connected_layer import initialize_parameters
    return initialize_parameters(
        TestConfig.FEATURE_DIM,
        TestConfig.HIDDEN_DIM,
        TestConfig.NUM_CLASSES
    )


# ==================== Flask App Fixtures ====================

@pytest.fixture
def flask_app():
    """Flask application for API testing"""
    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    app.config['DB_NAME'] = TestConfig.TEST_DB_NAME
    return app


@pytest.fixture
def client(flask_app):
    """Flask test client"""
    return flask_app.test_client()


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test_token'
    }


# ==================== Utility Fixtures ====================

@pytest.fixture
def mock_camera_data():
    """Mock camera status data"""
    cameras = []
    for i in range(TestConfig.NUM_OF_CAMERAS):
        cameras.append({
            'id': i,
            'name': f'Camera {i}',
            'status': True,
            'angle': TestConfig.CAMERA_ANGLES[i],
            'fps': TestConfig.CAMERA_FPS,
            'resolution': '224x224',
            'health': 'healthy'
        })
    return cameras


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()