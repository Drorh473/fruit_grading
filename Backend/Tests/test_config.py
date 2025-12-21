"""
Test Configuration
Centralized configuration for all tests
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load test environment
env_path = Path(__file__).parent.parent / '.env.test'
load_dotenv(dotenv_path=env_path)


class TestConfig:
    """Test configuration constants"""
    
    # Database
    MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    TEST_DB_NAME = os.getenv('DB_NAME', 'test_fruit_grading')
    TEST_COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'test_images')
    
    # Paths
    BASE_DIR = Path(__file__).parent
    FIXTURES_DIR = BASE_DIR / 'fixtures'
    IMAGES_DIR = FIXTURES_DIR / 'images'
    VALID_IMAGES_DIR = IMAGES_DIR / 'valid'
    EDGE_CASES_DIR = IMAGES_DIR / 'edge_cases'
    DATABASE_FIXTURES_DIR = FIXTURES_DIR / 'database'
    MODEL_FIXTURES_DIR = FIXTURES_DIR / 'models'
    
    # Temporary directories
    TEMP_DIR = Path('/tmp/fruit_grading_tests')
    TEMP_PROCESSED_DIR = TEMP_DIR / 'processed'
    TEMP_FEATURES_DIR = TEMP_DIR / 'features'
    
    # Camera configuration
    NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS', 4))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', 30))
    
    # Processing configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
    MAX_DIMENSION = int(os.getenv('MAX_DIMENSION', 224))
    
    # Model configuration
    FEATURE_DIM = 50176  # 7 * 7 * 1024
    HIDDEN_DIM = 16
    NUM_CLASSES = 3
    
    # Test data
    FRUIT_TYPES = ['market', 'standard', 'premium']
    CAMERA_ANGLES = ['Front View', 'Right View', 'Back View', 'Left View']
    
    # Performance benchmarks
    MAX_PREPROCESSING_TIME = 0.1  # seconds per image
    MAX_FEATURE_EXTRACTION_TIME = 0.05  # seconds per image
    MIN_BATCH_THROUGHPUT = 10  # images per second
    
    @classmethod
    def create_temp_dirs(cls):
        """Create temporary directories for testing"""
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def cleanup_temp_dirs(cls):
        """Clean up temporary directories after testing"""
        import shutil
        if cls.TEMP_DIR.exists():
            shutil.rmtree(cls.TEMP_DIR)