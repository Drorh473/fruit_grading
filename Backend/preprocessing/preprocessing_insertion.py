import os
import sys
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from preprocessing.preprocessing_from_db import (
    process_image,
    set_generator,
    PROCESSED_DATASET_PATH,
    STORED_DATASET_PATH
)

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def validate_folder_structure(folder_path):
    """Validate folder contains angle_0, angle_1, etc. with images."""
    if not os.path.exists(folder_path):
        return False, f"Folder does not exist: {folder_path}"
    
    if not os.path.isdir(folder_path):
        return False, f"Path is not a directory: {folder_path}"
    
    # Check for angle_ or an_ directories
    valid_prefixes = ("angle_", "an_")
    angle_dirs = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d)) and d.lower().startswith(valid_prefixes)]
    
    if not angle_dirs:
        return False, "No angle directories found (expected: angle_0, an_0, etc.)"
    
    has_images = False
    for angle_dir in angle_dirs:
        angle_path = os.path.join(folder_path, angle_dir)
        images = [f for f in os.listdir(angle_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            has_images = True
            break
    
    if not has_images:
        return False, "No images found in angle directories"
    
    return True, None


def preprocess_image_wrapper(args):
    """Wrapper to unpack 3 arguments for process_image."""
    # FIX: Unpack 3 arguments instead of 2
    source_path, output_dir, metadata = args
    
    try:
        # FIX: Call process_image with 3 separate arguments, not a tuple
        file_id, output_path, error = process_image(source_path, output_dir, metadata)
        return file_id, output_path, error
    except Exception as e:
        # Fallback error handling
        try:
            filename = os.path.basename(source_path)
            doc_id = filename.split('.')[0]
        except:
            doc_id = "unknown"
        return doc_id, None, f"Wrapper Error: {e}"


def preprocess_images_batch(image_docs, dummy_ids):
    process_args = []

    for i, doc in enumerate(image_docs):
        # Use 'testing' if set_type is empty or missing
        set_type = doc.get('set_type') or 'testing'
        meta = {
            'set_type': set_type,
            'camera_id': doc.get('camera_id'),
            '_id': dummy_ids[i],
            'object_id': dummy_ids[i]
        }
        process_args.append((doc['stored_path'], PROCESSED_DATASET_PATH, meta))

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # FIX: Use preprocess_image_wrapper instead of process_image
        results = list(executor.map(preprocess_image_wrapper, process_args))
        
    return results



def create_generator_for_object(images):
    """Create data generator for a single object's images."""
    image_paths = [img['processed_path'] for img in images if 'processed_path' in img]
    
    metadata_dict = {
        img['processed_path']: {
            'fruit_type': img.get('fruit_type', 'unknown'),
            'object_id': img.get('object_id'),
            'camera_id': img['camera_id'],
            'timestamp': img['timestamp']
        }
        for img in images if 'processed_path' in img
    }
    
    generator, _, count = set_generator(image_paths, metadata_dict)
    
    return generator, count
