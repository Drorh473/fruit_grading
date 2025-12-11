import os
import sys
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool

# Add project to path
PROJECT_DIR = '/mnt/project'
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
    
    angle_dirs = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("angle_")]
    
    if not angle_dirs:
        return False, "No angle directories found (expected: angle_0, angle_1, etc.)"
    
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
    """Wrapper for existing process_image function."""
    source_path, output_dir = args
    
    try:
        file_id, output_path, error = process_image((source_path, output_dir))
        return file_id, output_path, error
    except Exception as e:
        parts = source_path.split(os.sep)
        filename = parts[-1]
        doc_id = filename.split('.')[0]
        return doc_id, None, f"Error: {e}"


def preprocess_images_batch(image_data, inserted_ids):
    """Preprocess batch of images using existing process_image function."""
    preprocess_args = []
    
    for i, doc_id in enumerate(inserted_ids):
        img_data = image_data[i]
        _, ext = os.path.splitext(img_data['path'])
        stored_path = os.path.join(
            STORED_DATASET_PATH,
            img_data['set_type'],
            f"camera_{img_data['camera_id']}",
            f"{str(doc_id)}{ext}"
        )
        
        preprocess_args.append((stored_path, PROCESSED_DATASET_PATH))
    
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        preprocess_results = list(tqdm(
            pool.imap(preprocess_image_wrapper, preprocess_args),
            total=len(preprocess_args),
            desc="Preprocessing images"
        ))
    
    return preprocess_results


def create_generator_for_object(images):
    """Create data generator for a single object's images."""
    image_paths = [img['processed_path'] for img in images if 'processed_path' in img]
    
    metadata_dict = {
        img['processed_path']: {
            'fruit_type': img['fruit_type'],
            'object_id': img['object_id'],
            'camera_id': img['camera_id'],
            'timestamp': img['timestamp']
        }
        for img in images if 'processed_path' in img
    }
    
    generator, _, count = set_generator(image_paths, metadata_dict)
    
    return generator, count