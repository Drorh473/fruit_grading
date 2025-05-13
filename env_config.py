import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Database settings
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')

# Dataset paths
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')

# Camera settings
CAMERA_FPS = int(os.getenv('CAMERA_FPS', 30))
CAMERA_RANDOM_ORDER = os.getenv('CAMERA_RANDOM_ORDER', 'true').lower() == 'true'

# Model settings
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
DEFAULT_MODEL_VARIANT = os.getenv('DEFAULT_MODEL_VARIANT', '1.0x')

def get_db_config():
    """Return database configuration as a dictionary"""
    return {
        'connection_string': MONGO_CONNECTION_STRING,
        'db_name': DB_NAME
    }

def get_dataset_paths():
    """Return dataset paths as a dictionary"""
    return {
        'processed': PROCESSED_DATASET_PATH,
        'original': ORIGINAL_DATASET_PATH
    }

def get_camera_config():
    """Return camera configuration as a dictionary"""
    return {
        'fps': CAMERA_FPS,
        'random_order': CAMERA_RANDOM_ORDER
    }

def get_model_config():
    """Return model configuration as a dictionary"""
    return {
        'model_dir': MODEL_DIR,
        'default_variant': DEFAULT_MODEL_VARIANT
    }