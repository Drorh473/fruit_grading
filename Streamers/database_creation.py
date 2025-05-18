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
import re
from datetime import datetime, timedelta

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Get MongoDB connection string from .env
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not MONGODB_CONNECTION_STRING:
    raise ValueError("MongoDB connection string not found in .env file")

# Get dataset paths from .env
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH', "./data")
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH', "./processed_fruits")
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS', 4))  # Default to 4 cameras
FPS = int(os.getenv('FPS', 30))  # Frames per second

def create_database(db_name=os.getenv('DB_NAME', "fruit_rotation_db"), collection_name="images"):
    """Create MongoDB database for storing fruit image metadata"""
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Create indexes for faster queries
    collection.create_index("fruit_type")
    collection.create_index("object_id")      # Full object ID (e.g., obj0001)
    collection.create_index("set_type")       # Training/testing
    collection.create_index("camera_id")      # Added index for camera_id
    
    print(f"Connected to MongoDB database: {db_name}, collection: {collection_name}")
    return db_name, collection_name

def collect_images(dataset_path=ORIGINAL_DATASET_PATH):
    """Collect all images from the dataset with their metadata"""
    start_time = time.time()
    image_data = []
    
    # Get fruit types (top-level directories in dataset path)
    fruit_types = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    for fruit_type in fruit_types:
        fruit_dir = os.path.join(dataset_path, fruit_type)
        
        # Generate timestamp for this fruit type
        sequence_start = datetime.now()
        frame_duration = timedelta(seconds=1.0/FPS)
        
        # Find all object directories (like obj0000, obj0001, etc.)
        object_dirs = []
        for item in os.listdir(fruit_dir):
            if os.path.isdir(os.path.join(fruit_dir, item)) and item.startswith("obj"):
                object_dirs.append(item)
        
        if not object_dirs:
            print(f"No object directories found for {fruit_type}, skipping")
            continue
            
        print(f"Found {len(object_dirs)} objects for {fruit_type}")
        
        # Process each object directory
        frame_count = 0
        for obj_id in object_dirs:
            obj_dir = os.path.join(fruit_dir, obj_id)
            
            # Assign a random camera_id (between 1 and 4) for this object
            camera_id = random.randint(1, NUM_OF_CAMERAS)
            
            # Find all images for this object
            images = []
            for file in os.listdir(obj_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    images.append(file)
            
            # Extract orientation from image names (im1.png, im2.png, etc.)
            for img_file in images:
                # Extract orientation number
                orientation_match = re.search(r'im(\d+)', img_file)
                orientation = int(orientation_match.group(1)) if orientation_match else 0
                
                img_path = os.path.join(obj_dir, img_file)
                
                # Generate timestamp for this frame
                timestamp = sequence_start + (frame_count * frame_duration)
                frame_count += 1
                
                # Convert timestamp to ISO format string for MongoDB storage
                timestamp_str = timestamp.isoformat()
                
                # Get image dimensions
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    color_channels = 3 if img.mode == 'RGB' else 1
                    img.close()
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
                
                # Create image metadata
                img_data = {
                    "path": img_path,
                    "fruit_type": fruit_type,
                    "object_id": obj_id,      # Full object ID (e.g., "obj0001")
                    "orientation": orientation,
                    "camera_id": camera_id,   # Added camera_id field
                    "timestamp": timestamp_str,  # Store as string for MongoDB compatibility
                    "frame_number": frame_count,
                    "width": width,
                    "height": height,
                    "color": color_channels,
                    "set_type": "",           # Will be set later
                    "category": ""            # Left empty as requested
                }
                
                image_data.append(img_data)
    
    print(f"Found {len(image_data)} images in {time.time() - start_time:.2f} seconds")
    return image_data

def store_in_database(image_data, db_name, collection_name):
    """Store image metadata in MongoDB"""
    start_time = time.time()
    print(f"Storing {len(image_data)} images in database...")
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Insert in batches
    batch_size = 1000
    for i in tqdm(range(0, len(image_data), batch_size), desc="Inserting batches"):
        batch = image_data[i:i+batch_size]
        collection.insert_many(batch, ordered=False)
    
    print(f"Stored {len(image_data)} images in {time.time() - start_time:.2f} seconds")
    return db_name, collection_name

def split_data(db_name, collection_name, training_percentage=80, testing_percentage=20):
    """Split data into training and testing sets (80/20 split)"""
    start_time = time.time()
    
    # Verify percentages
    if training_percentage + testing_percentage != 100:
        raise ValueError("Training and testing percentages must sum to 100")
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Group by fruit_type and object_id to ensure object-level split
    groups = {}
    
    # Get all documents
    all_docs = list(collection.find({}, {"_id": 1, "fruit_type": 1, "object_id": 1}))
    
    # Group documents by fruit_type and object_id
    for doc in all_docs:
        group_key = f"{doc['fruit_type']}_{doc['object_id']}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(doc["_id"])
    
    # Shuffle the group keys to randomize the split
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    # Calculate split point for groups
    training_groups_count = int(len(group_keys) * (training_percentage / 100))
    
    # Assign groups to training or testing
    training_ids = []
    testing_ids = []
    
    for i, key in enumerate(group_keys):
        if i < training_groups_count:
            training_ids.extend(groups[key])
        else:
            testing_ids.extend(groups[key])
    
    # Update database for training set
    if training_ids:
        collection.update_many(
            {"_id": {"$in": training_ids}},
            {"$set": {"set_type": "training"}}
        )
    
    # Update database for testing set
    if testing_ids:
        collection.update_many(
            {"_id": {"$in": testing_ids}},
            {"$set": {"set_type": "testing"}}
        )
    
    # Verify the split
    training_count = collection.count_documents({"set_type": "training"})
    testing_count = collection.count_documents({"set_type": "testing"})
    
    print("Data split results:")
    print(f"  training: {training_count} images ({training_percentage}%)")
    print(f"  testing: {testing_count} images ({testing_percentage}%)")
    print(f"Split completed in {time.time() - start_time:.2f} seconds")
    
    return db_name, collection_name

def copy_image_file(args):
    """Copy image file to the output directory"""
    doc_id, path, fruit_type, object_id, set_type, output_dir = args
    
    # Create target directory: output_dir/set_type/fruit_type/object_id
    target_dir = os.path.join(output_dir, set_type, fruit_type, object_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Get file extension
    _, ext = os.path.splitext(path)
    
    # Create destination path
    dest_path = os.path.join(target_dir, f"{doc_id}{ext}")
    
    try:
        shutil.copy2(path, dest_path)
        return True
    except Exception as e:
        print(f"Error copying {path}: {e}")
        return False

def create_directory_structure(db_name, collection_name, output_dir=STORED_DATASET_PATH):
    """Create directory structure and copy files"""
    start_time = time.time()
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # If directory exists, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create main directories - only training and testing
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    
    # Get all images with metadata
    all_images = list(collection.find({}, {
        "_id": 1,
        "path": 1,
        "fruit_type": 1,
        "object_id": 1,
        "set_type": 1
    }))
    
    # Prepare arguments for parallel copying
    copy_args = [
        (str(doc["_id"]), doc["path"], doc["fruit_type"], doc["object_id"], 
         doc["set_type"], output_dir)
        for doc in all_images if "set_type" in doc
    ]
    
    # Use multiprocessing for copying
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_image_file, copy_args),
            total=len(copy_args),
            desc="Copying files"
        ))
    
    success_count = sum(1 for r in results if r)
    print(f"Successfully copied {success_count} of {len(copy_args)} images in {time.time() - start_time:.2f} seconds")
    
    return output_dir

def update_data_directory(db_name, collection_name, data_dir=STORED_DATASET_PATH):
    """
    Scan the structured directory and update only the path field in database entries.
    """
    start_time = time.time()
    
    # Connect to MongoDB
    client = pymongo.MongoClient(
        MONGODB_CONNECTION_STRING,
        maxPoolSize=50,
        retryWrites=True
    )
    db = client[db_name]
    collection = db[collection_name]
    
    # Counters for statistics
    files_processed = 0
    updates_made = 0
    errors = 0
    
    # Use bulk operations for better performance
    bulk_operations = []
    MAX_BULK_SIZE = 500
    
    print(f"Starting directory scan at {data_dir}...")
    
    for set_type in ['training', 'testing']:
        set_type_dir = os.path.join(data_dir, set_type)
        if not os.path.exists(set_type_dir):
            print(f"Warning: {set_type_dir} does not exist, skipping.")
            continue
        
        for category in os.listdir(set_type_dir):
            category_path = os.path.join(set_type_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            for obj in os.listdir(category_path):
                obj_path = os.path.join(category_path, obj)
                if not os.path.isdir(obj_path):
                    continue
                
                for filename in os.listdir(obj_path):
                    if not filename.lower().endswith('.png'):
                        continue
                    
                    files_processed += 1
                    
                    try:
                        #full path to the image file
                        image_path = os.path.join(obj_path, filename)
                        
                        # Extract ID from filename only (not full path)
                        image_id = os.path.splitext(filename)[0]
                        
                        # Debug print for first few files
                        if files_processed <= 3:
                            print(f"Debug: filename={filename}, image_id={image_id}, path={image_path}")
                        
                        # Try with string ID first (most likely case for your setup)
                        object_id = ObjectId(image_id)
                
                        # Create update operation - updating ONLY the path
                        bulk_operations.append(
                            pymongo.UpdateOne(
                                {"_id": object_id},
                                {"$set": {"path": image_path}}
                            )
                        )
                    
                        # Execute bulk operation when batch size is reached
                        if len(bulk_operations) >= MAX_BULK_SIZE:
                            result = collection.bulk_write(bulk_operations, ordered=False)
                            updates_made += result.modified_count
                            bulk_operations = []
                            
                            print(f"Processed {files_processed} files, updated {updates_made} records so far...")
                            
                    except Exception as e:
                        errors += 1
                        if errors < 10:
                            print(f"Error processing {filename}: {e}")
                        elif errors == 10:
                            print("Too many errors, suppressing further error messages...")
        
    # Process any remaining operations
    if bulk_operations:
        try:
            result = collection.bulk_write(bulk_operations, ordered=False)
            updates_made += result.modified_count
        except Exception as e:
            print(f"Error in final bulk update: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\nDirectory scan complete in {elapsed_time:.2f} seconds")
    print(f"Files processed: {files_processed}")
    print(f"Database records updated: {updates_made}")
    print(f"Errors encountered: {errors}")
    
    client.close()
    
    
def print_summary(db_name, collection_name):
    """Print summary statistics from the database"""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Count by fruit type
    fruit_pipeline = [
        {"$group": {"_id": "$fruit_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    fruit_counts = list(collection.aggregate(fruit_pipeline))
    
    # Count by camera_id
    camera_pipeline = [
        {"$group": {"_id": "$camera_id", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    camera_counts = list(collection.aggregate(camera_pipeline))
    
    # Count by set_type
    set_type_pipeline = [
        {"$group": {"_id": "$set_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    set_type_counts = list(collection.aggregate(set_type_pipeline))
    
    print("\nDatabase Summary:")
    print("=================")
    
    print("\nFruit Types:")
    for item in fruit_counts:
        print(f"  {item['_id']}: {item['count']} images")
    
    print("\nCamera Distribution:")
    for item in camera_counts:
        print(f"  Camera {item['_id']}: {item['count']} images")
    
    print("\nData Splits:")
    for item in set_type_counts:
        print(f"  {item['_id']}: {item['count']} images")
    
    client.close()

def process_dataset(dataset_path=ORIGINAL_DATASET_PATH, db_name=os.getenv('DB_NAME', "fruit_rotation_db"), collection_name="images"):
    """Process the full dataset pipeline"""
    # Create flag file path
    processed_path = STORED_DATASET_PATH
    flag_file = os.path.join(processed_path, ".processing_complete")
    
    # Check if already processed
    if os.path.exists(flag_file):
        print(f"Dataset already processed. Using existing data.")
        return db_name, collection_name
    
    # Process from scratch
    print("Processing dataset from scratch...")
    start_time = time.time()
    
    # Create database
    db_name, collection_name = create_database(db_name, collection_name)
    
    # Collect and store images
    image_data = collect_images(dataset_path)
    store_in_database(image_data, db_name, collection_name)
    
    # Split data - 80% training, 20% testing
    split_data(db_name, collection_name, 80, 20)
    
    # Create directory structure
    create_directory_structure(db_name, collection_name, STORED_DATASET_PATH)
    
    # Update paths
    update_data_directory(db_name, collection_name, STORED_DATASET_PATH)
    
    # Print database summary
    print_summary(db_name, collection_name)
    
    # Create flag file
    os.makedirs(os.path.dirname(flag_file), exist_ok=True)
    with open(flag_file, 'w') as f:
        f.write(f"Processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Dataset processing complete in {time.time() - start_time:.2f} seconds")
    return db_name, collection_name

if __name__ == "__main__":
    # Use environment variables or command line
    if not ORIGINAL_DATASET_PATH:
        print("Warning: ORIGINAL_DATASET_PATH not set in .env file")
        dataset_path = input("Please enter the path to the dataset: ")
    else:
        dataset_path = ORIGINAL_DATASET_PATH
    
    db_name, collection_name = process_dataset(dataset_path)