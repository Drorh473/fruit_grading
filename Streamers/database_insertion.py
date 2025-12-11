import os
import sys
import time
import pymongo
import shutil
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocessing import Pool

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from Streamers.database_creation import (
    copy_image_file,
    split_data,
    MONGODB_CONNECTION_STRING,
    STORED_DATASET_PATH,
    NUM_OF_CAMERAS,
    FPS
)

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME', 'fruit_grading')


def get_next_object_id(db_name=None, collection_name="images"):
    """Get next available object ID by querying database for max ID."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    object_ids = collection.distinct("object_id")
    client.close()
    
    if not object_ids:
        return "obj0001"
    
    max_num = 0
    for obj_id in object_ids:
        if obj_id.startswith("obj"):
            try:
                num = int(obj_id[3:])
                max_num = max(max_num, num)
            except ValueError:
                continue
    
    next_num = max_num + 1
    return f"obj{next_num:04d}"


def collect_images_metadata(folder_path, db_name=None, collection_name="images"):
    """Collect image metadata from angle directories with auto-incremented object ID."""
    image_data = []
    
    object_id = get_next_object_id(db_name, collection_name)
    print(f"Assigned object ID: {object_id}")
    
    sequence_start = datetime.now()
    frame_duration = timedelta(seconds=1.0/FPS)
    frame_count = 0
    
    angle_dirs = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("angle_")]
    
    if not angle_dirs:
        return []
    
    angle_dirs.sort()
    print(f"Found {len(angle_dirs)} angle directories")
    
    for angle_id in angle_dirs:
        angle_dir = os.path.join(folder_path, angle_id)
        
        try:
            camera_id = int(angle_id.split('_')[-1])
        except ValueError:
            camera_id = 0
        
        if camera_id >= NUM_OF_CAMERAS:
            camera_id = 0
        
        images = sorted([f for f in os.listdir(angle_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in images:
            img_path = os.path.join(angle_dir, img_file)
            timestamp = sequence_start + (frame_count * frame_duration)
            frame_count += 1
            
            img_data = {
                "path": img_path,
                "fruit_type": "unknown",
                "object_id": object_id,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "frame_number": frame_count,
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": "",
                "added_date": datetime.now()
            }
            
            image_data.append(img_data)
    
    return image_data


def insert_images_to_db(image_data, db_name=None, collection_name="images"):
    """Insert image metadata into database and return inserted IDs."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    result = collection.insert_many(image_data, ordered=False)
    inserted_ids = result.inserted_ids
    
    client.close()
    
    return inserted_ids


def copy_images_to_stored(image_data, inserted_ids):
    """Copy images to stored dataset using existing copy_image_file function."""
    copy_args = []
    for i, doc_id in enumerate(inserted_ids):
        img_data = image_data[i]
        copy_args.append((
            str(doc_id),
            img_data['path'],
            img_data['camera_id'],
            img_data['set_type'],
            STORED_DATASET_PATH
        ))
    
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        copy_results = list(tqdm(
            pool.imap(copy_image_file, copy_args),
            total=len(copy_args),
            desc="Copying images"
        ))
    
    return copy_results


def update_stored_paths(image_data, inserted_ids, copy_results, db_name=None, collection_name="images"):
    """Update database with stored dataset paths."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    update_ops = []
    success_count = 0
    
    for i, success in enumerate(copy_results):
        if success:
            doc_id = str(inserted_ids[i])
            img_data = image_data[i]
            
            _, ext = os.path.splitext(img_data['path'])
            dest_path = os.path.join(
                STORED_DATASET_PATH,
                img_data['set_type'],
                f"camera_{img_data['camera_id']}",
                f"{doc_id}{ext}"
            )
            
            update_ops.append(pymongo.UpdateOne(
                {'_id': ObjectId(doc_id)},
                {'$set': {'path': dest_path}}
            ))
            success_count += 1
    
    if update_ops:
        collection.bulk_write(update_ops, ordered=False)
    
    client.close()
    
    return success_count


def update_preprocessed_paths(preprocess_results, db_name=None, collection_name="images"):
    """Update database with preprocessed dataset paths."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    preprocess_ops = []
    for doc_id, processed_path, error in preprocess_results:
        if processed_path:
            preprocess_ops.append(pymongo.UpdateOne(
                {'_id': ObjectId(doc_id)},
                {'$set': {'processed_path': processed_path}}
            ))
    
    if preprocess_ops:
        collection.bulk_write(preprocess_ops, ordered=False)
    
    client.close()
    
    return len(preprocess_ops)


def split_training_testing(db_name=None, collection_name="images"):
    """Split new images into training (66%) and testing (34%) sets."""
    if db_name is None:
        db_name = DB_NAME
    
    split_data(db_name, collection_name, 66, 34)


def get_images_by_object(object_id, db_name=None, collection_name="images"):
    """Retrieve all images for a specific object ID."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    images = list(collection.find({"object_id": object_id}))
    
    client.close()
    
    return images


def update_fruit_type(object_id, fruit_type, confidence=None, db_name=None, collection_name="images"):
    """Update fruit type and inference results for an object."""
    if db_name is None:
        db_name = DB_NAME
    
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    update_data = {
        "fruit_type": fruit_type,
        "inference_result": {
            "predicted_type": fruit_type,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
    }
    
    result = collection.update_many(
        {"object_id": object_id},
        {"$set": update_data}
    )
    
    client.close()
    
    return result.modified_count