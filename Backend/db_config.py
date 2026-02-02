"""Database configuration with automatic test detection."""
import os
from pathlib import Path
from functools import lru_cache
import pymongo


def is_test_mode():
    """Check if running in test mode."""
    indicators = [
        os.getenv('TESTING', '').lower() == 'true',
        os.getenv('PYTEST_CURRENT_TEST') is not None,
        'pytest' in os.getenv('_', ''),
        os.getenv('DB_NAME', '') == 'test_fruit_grading',
    ]
    return any(indicators)


def _load_environment():
    """Load the appropriate environment file."""
    from dotenv import load_dotenv

    if is_test_mode():
        possible_paths = [
            Path(__file__).parent / 'Tests' / '.env.test',
            Path(__file__).parent / '.env.test',
            Path('.') / '.env.test',
            Path(__file__).parent / 'Tests' / '_env.test',
        ]

        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                return 'test'

        os.environ.setdefault('DB_NAME', 'test_fruit_grading')
        return 'test_default'
    else:
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        return 'production'


_env_mode = _load_environment()


class DatabaseConfig:
    """Database configuration with test-awareness."""

    PRODUCTION_DB_NAME = 'fruit_grading'
    TEST_DB_NAME = 'test_fruit_grading'

    def __init__(self):
        self._test_mode = is_test_mode()

    @property
    def connection_string(self):
        return os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')

    @property
    def db_name(self):
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
        return os.getenv(key, default)

    def get_int(self, key, default=0):
        return int(os.getenv(key, default))

    def verify_safe_for_production(self):
        """Prevent test mode from targeting production database."""
        if self._test_mode and self.db_name == self.PRODUCTION_DB_NAME:
            raise RuntimeError("Test mode cannot target production database")
        return True


_config = None


def get_db_config():
    """Get database configuration singleton."""
    global _config
    if _config is None:
        _config = DatabaseConfig()
    return _config


@lru_cache(maxsize=1)
def get_mongo_client():
    """Get cached MongoDB client."""
    config = get_db_config()
    config.verify_safe_for_production()

    client = pymongo.MongoClient(
        config.connection_string,
        serverSelectionTimeoutMS=5000,
        maxPoolSize=50
    )
    client.server_info()
    return client


def get_database():
    """Get database instance."""
    config = get_db_config()
    client = get_mongo_client()
    return client[config.db_name]


def get_collection(collection_name=None):
    """Get collection from database."""
    config = get_db_config()
    db = get_database()
    if collection_name is None:
        collection_name = config.collection_name
    return db[collection_name]


def close_connections():
    """Close all database connections."""
    global _config
    get_mongo_client.cache_clear()
    _config = None


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