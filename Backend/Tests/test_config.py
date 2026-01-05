"""
Test Configuration
Centralized configuration for all tests with proper database isolation
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# CRITICAL: Set test environment BEFORE any other imports
# This must happen first to prevent production modules from loading production env
os.environ['TESTING'] = 'true'
os.environ['DB_NAME'] = 'test_fruit_grading'

# Find the test environment file
# Try multiple locations for flexibility
_current_dir = Path(__file__).parent
_possible_env_paths = [
    _current_dir / '.env.test',           # Same directory as test_config.py
    _current_dir.parent / '.env.test',    # Parent directory
    _current_dir / '_env.test',           # Alternative naming
    Path('.') / '.env.test',              # Current working directory
]

_env_loaded = False
for env_path in _possible_env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)  # override=True is KEY!
        print(f"[TestConfig] Loaded test environment from: {env_path}")
        _env_loaded = True
        break

if not _env_loaded:
    print(f"[TestConfig] WARNING: No .env.test file found! Using defaults.")
    print(f"[TestConfig] Searched locations: {[str(p) for p in _possible_env_paths]}")

# Force test database name (safety net)
os.environ['DB_NAME'] = 'test_fruit_grading'


class TestConfig:
    """Test configuration constants with database isolation"""
    
    # Database - ALWAYS use test database
    MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    TEST_DB_NAME = 'test_fruit_grading'  # Hardcoded for safety
    TEST_COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'test_images')
    
    # Verify we're NOT using production database
    _db_name_from_env = os.getenv('DB_NAME', '')
    if _db_name_from_env == 'fruit_grading':
        raise RuntimeError(
            "CRITICAL: Test configuration loaded production DB_NAME 'fruit_grading'! "
            "Check your .env.test file and ensure it's being loaded correctly."
        )
    
    # Paths - Use temporary directories for test isolation
    BASE_DIR = Path(__file__).parent
    FIXTURES_DIR = BASE_DIR / 'fixtures'
    IMAGES_DIR = FIXTURES_DIR / 'images'
    VALID_IMAGES_DIR = IMAGES_DIR / 'valid'
    EDGE_CASES_DIR = IMAGES_DIR / 'edge_cases'
    DATABASE_FIXTURES_DIR = FIXTURES_DIR / 'database'
    MODEL_FIXTURES_DIR = FIXTURES_DIR / 'models'
    
    # Temporary directories - Use /tmp for complete isolation
    TEMP_DIR = Path('/tmp/fruit_grading_tests')
    TEMP_PROCESSED_DIR = TEMP_DIR / 'processed'
    TEMP_FEATURES_DIR = TEMP_DIR / 'features'
    TEMP_STORED_DATASET_DIR = TEMP_DIR / 'stored_dataset'
    TEMP_ORIGINAL_DATASET_DIR = TEMP_DIR / 'original_dataset'
    TEMP_PROCESSED_DATASET_DIR = TEMP_DIR / 'processed_dataset'
    
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
        cls.TEMP_STORED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_ORIGINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def cleanup_temp_dirs(cls):
        """Clean up temporary directories after testing"""
        import shutil
        if cls.TEMP_DIR.exists():
            shutil.rmtree(cls.TEMP_DIR)
    
    @classmethod
    def get_test_db_name(cls):
        """Always return test database name"""
        return cls.TEST_DB_NAME
    
    @classmethod
    def verify_test_mode(cls):
        """Verify we're running in test mode - call this before any DB operations"""
        current_db = os.getenv('DB_NAME', '')
        if current_db == 'fruit_grading':
            raise RuntimeError("Tests attempting to use production database!")
        return True


def ensure_test_environment():
    """
    Call this function at the start of test modules to ensure
    test environment is properly configured.
    
    Usage:
        from test_config import ensure_test_environment
        ensure_test_environment()
    """
    # Force test database
    os.environ['DB_NAME'] = 'test_fruit_grading'
    os.environ['TESTING'] = 'true'
    
    # Verify configuration
    TestConfig.verify_test_mode()
    
    return True