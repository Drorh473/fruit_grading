"""
Database Configuration Module
Centralized database configuration with automatic test detection

All modules should import database settings from here instead of
loading .env directly. This ensures test isolation.

Usage:
    from db_config import get_db_config, get_mongo_client, get_collection
    
    config = get_db_config()
    client = get_mongo_client()
    collection = get_collection('images')
"""
import os
from pathlib import Path
from functools import lru_cache
import pymongo


def is_test_mode():
    """Check if we're running in test mode"""
    # Check multiple indicators
    indicators = [
        os.getenv('TESTING', '').lower() == 'true',
        os.getenv('PYTEST_CURRENT_TEST') is not None,
        'pytest' in os.getenv('_', ''),
        os.getenv('DB_NAME', '') == 'test_fruit_grading',
    ]
    return any(indicators)


def _load_environment():
    """Load the appropriate environment file"""
    from dotenv import load_dotenv
    
    # Determine which env file to load
    if is_test_mode():
        # In test mode, try to load .env.test
        possible_paths = [
            Path(__file__).parent / 'Tests' / '.env.test',
            Path(__file__).parent / '.env.test',
            Path('.') / '.env.test',
            Path(__file__).parent / 'Tests' / '_env.test',
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                print(f"[db_config] Loaded TEST environment from: {env_path}")
                return 'test'
        
        # No test env file found, but we're in test mode
        # Set defaults for test database
        os.environ.setdefault('DB_NAME', 'test_fruit_grading')
        print("[db_config] No .env.test found, using test defaults")
        return 'test_default'
    else:
        # Production mode
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"[db_config] Loaded production environment from: {env_path}")
        return 'production'


# Load environment on module import
_env_mode = _load_environment()


class DatabaseConfig:
    """Database configuration class with test-awareness"""
    
    # Production database name (for safety checks)
    PRODUCTION_DB_NAME = 'fruit_grading'
    TEST_DB_NAME = 'test_fruit_grading'
    
    def __init__(self):
        self._test_mode = is_test_mode()
        
    @property
    def connection_string(self):
        return os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
    
    @property
    def db_name(self):
        """Get database name - ALWAYS returns test DB in test mode"""
        if self._test_mode:
            return self.TEST_DB_NAME
        return os.getenv('DB_NAME', self.PRODUCTION_DB_NAME)
    
    @property
    def collection_name(self):
        if self._test_mode:
            return os.getenv('COLLECTION_NAME', 'test_images')
        return os.getenv('COLLECTION_NAME', 'images')
    
    @property
    def is_test_mode(self):
        return self._test_mode
    
    def get_path(self, key, default=None):
        """Get a path configuration value"""
        return os.getenv(key, default)
    
    def get_int(self, key, default=0):
        """Get an integer configuration value"""
        return int(os.getenv(key, default))
    
    def verify_safe_for_production(self):
        """Verify we're not accidentally targeting production in test mode"""
        if self._test_mode and self.db_name == self.PRODUCTION_DB_NAME:
            raise RuntimeError(
                f"SAFETY VIOLATION: Test mode is active but database name is "
                f"'{self.PRODUCTION_DB_NAME}'. This would corrupt production data!"
            )
        return True


# Singleton configuration instance
_config = None


def get_db_config():
    """Get the database configuration singleton"""
    global _config
    if _config is None:
        _config = DatabaseConfig()
    return _config


@lru_cache(maxsize=1)
def get_mongo_client():
    """Get a MongoDB client (cached)"""
    config = get_db_config()
    config.verify_safe_for_production()
    
    client = pymongo.MongoClient(
        config.connection_string,
        serverSelectionTimeoutMS=5000,
        maxPoolSize=50
    )
    
    # Verify connection
    try:
        client.server_info()
    except Exception as e:
        print(f"[db_config] MongoDB connection failed: {e}")
        raise
    
    return client


def get_database():
    """Get the database instance"""
    config = get_db_config()
    client = get_mongo_client()
    return client[config.db_name]


def get_collection(collection_name=None):
    """Get a collection from the database"""
    config = get_db_config()
    db = get_database()
    
    if collection_name is None:
        collection_name = config.collection_name
    
    return db[collection_name]


def close_connections():
    """Close all database connections"""
    global _config
    get_mongo_client.cache_clear()
    _config = None


# Convenience exports for common configuration values
def get_stored_dataset_path():
    return os.getenv('STORED_DATASET_PATH')


def get_original_dataset_path():
    return os.getenv('ORIGINAL_DATASET_PATH')


def get_processed_dataset_path():
    return os.getenv('PROCESSED_DATASET_PATH')


def get_model_dir():
    return os.getenv('MODEL_DIR', 'saved_models')


def get_num_cameras():
    return int(os.getenv('NUM_OF_CAMERAS', 4))


def get_batch_size():
    return int(os.getenv('BATCH_SIZE', 32))


# Safety check on module load
if is_test_mode():
    print(f"[db_config] TEST MODE ACTIVE - Using database: {get_db_config().db_name}")