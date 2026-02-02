import os
import time
import random
import shutil
import pymongo
from PIL import Image
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from bson.objectid import ObjectId
from datetime import datetime, timedelta

env_path = Path(__file__).parent.parent / '.env'
if not env_path.exists():
    env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not MONGODB_CONNECTION_STRING:
    raise ValueError("MongoDB connection string not found in .env file")

ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH', "./data")
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH', "./processed_fruits")
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS', 4))
FPS = int(os.getenv('FPS', 30))

def create_database(db_name=os.getenv('DB_NAME'), collection_name="images"):
    """Create MongoDB database for storing fruit image metadata."""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    collection.create_index("fruit_type")
    collection.create_index("object_id")
    collection.create_index("set_type")
    collection.create_index("camera_id")
    return db_name, collection_name

def collect_images(dataset_path=ORIGINAL_DATASET_PATH):
    """Collect all images from the dataset with their metadata."""
    start_time = time.time()
    image_data = []
    width = int(os.getenv('IMAGE_SIZE_W'))
    height = int(os.getenv('IMAGE_SIZE_H'))

    if not dataset_path or not os.path.exists(dataset_path):
        return []

    try:
        categories = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]
    except (FileNotFoundError, PermissionError):
        return []
    
    for category in categories:
        category_dir = os.path.join(dataset_path, category)
        sequence_start = datetime.now()
        frame_duration = timedelta(seconds=1.0/FPS)

        object_dirs = []
        for item in os.listdir(category_dir):
            if os.path.isdir(os.path.join(category_dir, item)) and item.startswith("obj"):
                object_dirs.append(item)
        
        if not object_dirs:
            continue

        frame_count = 0
        for obj_id in object_dirs:
            obj_dir = os.path.join(category_dir, obj_id)
            angle_dirs = []
            for item in os.listdir(obj_dir):
                angle_path = os.path.join(obj_dir, item)
                if os.path.isdir(angle_path):
                    angle_dirs.append(item)
            
            for angle_id in angle_dirs:
                angle_dir = os.path.join(obj_dir, angle_id)
                camera_id = int(angle_id[-1]) if angle_id[-1].isdigit() else 1

                images = []
                for file in os.listdir(angle_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images.append(file)
                
                for img_file in images:
                    img_path = os.path.join(angle_dir, img_file)
                    timestamp = sequence_start + (frame_count * frame_duration)
                    frame_count += 1

                    img_data = {
                        "path": img_path,
                        "fruit_type": category,
                        "object_id": f"{obj_id}_{category}",
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "frame_number": frame_count,
                        "width": width,
                        "height": height,
                        "color": 3,
                        "set_type": ""
                    }
                    
                    image_data.append(img_data)

    return image_data

def store_in_database(image_data, db_name, collection_name):
    """Store image metadata in MongoDB."""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    batch_size = 1000
    for i in tqdm(range(0, len(image_data), batch_size), desc="Inserting batches"):
        batch = image_data[i:i+batch_size]
        collection.insert_many(batch, ordered=False)

    return db_name, collection_name

def split_data(db_name, collection_name, training_percentage=66, testing_percentage=34):
    """Split data into training and testing sets with stratified sampling."""
    if training_percentage + testing_percentage != 100:
        raise ValueError("Training and testing percentages must sum to 100")

    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    all_docs = list(collection.find({}, {"_id": 1, "fruit_type": 1, "object_id": 1}))

    fruit_type_objects = {}
    for doc in all_docs:
        fruit_type = doc.get('fruit_type', 'unknown')
        obj_id = doc['object_id']
        
        if fruit_type not in fruit_type_objects:
            fruit_type_objects[fruit_type] = {}
        if obj_id not in fruit_type_objects[fruit_type]:
            fruit_type_objects[fruit_type][obj_id] = []
        fruit_type_objects[fruit_type][obj_id].append(doc["_id"])
    
    # Print object counts per fruit_type
    print("\nObjects per fruit_type:")
    for ft, objs in fruit_type_objects.items():
        print(f"  {ft}: {len(objs)} objects")
    
    training_ids = []
    testing_ids = []

    for fruit_type, ft_objects in fruit_type_objects.items():
        obj_ids = list(ft_objects.keys())
        random.shuffle(obj_ids)
        
        split_point = max(1, int(len(obj_ids) * (training_percentage / 100)))
        
        train_objs = obj_ids[:split_point]
        test_objs = obj_ids[split_point:]
        
        for obj_id in train_objs:
            training_ids.extend(ft_objects[obj_id])
        for obj_id in test_objs:
            testing_ids.extend(ft_objects[obj_id])

    if training_ids:
        collection.update_many(
            {"_id": {"$in": training_ids}},
            {"$set": {"set_type": "training"}}
        )
    
    if testing_ids:
        collection.update_many(
            {"_id": {"$in": testing_ids}},
            {"$set": {"set_type": "testing"}}
        )

    return db_name, collection_name

def copy_image_file(args):
    """Copy image file to output directory organized by camera_id."""
    doc_id, path, camera_id, set_type, output_dir = args
    target_dir = os.path.join(output_dir, set_type, f"camera_{camera_id}")
    os.makedirs(target_dir, exist_ok=True)
    
    _, ext = os.path.splitext(path)
    dest_path = os.path.join(target_dir, f"{doc_id}{ext}")
    
    try:
        shutil.copy2(path, dest_path)
        return True
    except Exception:
        return False


def create_directory_structure(db_name, collection_name, output_dir=STORED_DATASET_PATH):
    """Create directory structure and copy files organized by camera_id."""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)

    for set_type in ["training", "testing"]:
        for camera_id in range(0, NUM_OF_CAMERAS):
            os.makedirs(os.path.join(output_dir, set_type, f"camera_{camera_id}"), exist_ok=True)

    all_images = list(collection.find({}, {
        "_id": 1,
        "path": 1,
        "camera_id": 1,
        "set_type": 1
    }))

    copy_args = [
        (str(doc["_id"]), doc["path"], doc["camera_id"], 
         doc["set_type"], output_dir)
        for doc in all_images if "set_type" in doc and "camera_id" in doc
    ]

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_image_file, copy_args),
            total=len(copy_args),
            desc="Copying files"
        ))

    return output_dir

def update_data_directory(db_name, collection_name, data_dir=STORED_DATASET_PATH):
    """Scan directory structure and update path field in database entries."""
    client = pymongo.MongoClient(
        MONGODB_CONNECTION_STRING,
        maxPoolSize=50,
        retryWrites=True
    )
    db = client[db_name]
    collection = db[collection_name]

    files_processed = 0
    updates_made = 0
    bulk_operations = []
    MAX_BULK_SIZE = 500

    for set_type in ['training', 'testing']:
        set_type_dir = os.path.join(data_dir, set_type)
        if not os.path.exists(set_type_dir):
            continue

        for camera_folder in os.listdir(set_type_dir):
            camera_path = os.path.join(set_type_dir, camera_folder)
            if not os.path.isdir(camera_path):
                continue

            for filename in os.listdir(camera_path):
                if not filename.lower().endswith('.jpg'):
                    continue
                
                try:
                    image_path = os.path.join(camera_path, filename)
                    image_id = os.path.splitext(filename)[0]
                    object_id = ObjectId(image_id)

                    bulk_operations.append(
                        pymongo.UpdateOne(
                            {"_id": object_id},
                            {"$set": {"path": image_path}}
                        )
                    )

                    if len(bulk_operations) >= MAX_BULK_SIZE:
                        result = collection.bulk_write(bulk_operations, ordered=False)
                        updates_made += result.modified_count
                        bulk_operations = []

                    files_processed += 1

                except Exception:
                    pass

    if bulk_operations:
        try:
            result = collection.bulk_write(bulk_operations, ordered=False)
            updates_made += result.modified_count
        except Exception:
            pass

    client.close()
    
    
def print_summary(db_name, collection_name):
    """Print summary statistics from the database."""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]

    fruit_type_pipeline = [
        {"$group": {"_id": "$fruit_type", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    list(collection.aggregate(fruit_type_pipeline))

    camera_pipeline = [
        {"$group": {"_id": "$camera_id", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    list(collection.aggregate(camera_pipeline))

    set_type_pipeline = [
        {"$group": {"_id": "$set_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    list(collection.aggregate(set_type_pipeline))

    client.close()

def process_dataset(dataset_path=ORIGINAL_DATASET_PATH, db_name=os.getenv('DB_NAME'), collection_name="images"):
    """Process the full dataset pipeline."""
    processed_path = STORED_DATASET_PATH
    flag_file = os.path.join(processed_path, ".processing_complete")

    if os.path.exists(flag_file):
        return db_name, collection_name

    db_name, collection_name = create_database(db_name, collection_name)

    image_data = collect_images(dataset_path)
    store_in_database(image_data, db_name, collection_name)
    split_data(db_name, collection_name, 66, 34)
    create_directory_structure(db_name, collection_name, STORED_DATASET_PATH)
    update_data_directory(db_name, collection_name, STORED_DATASET_PATH)
    print_summary(db_name, collection_name)

    os.makedirs(os.path.dirname(flag_file), exist_ok=True)
    with open(flag_file, 'w') as f:
        f.write(f"Processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return db_name, collection_name