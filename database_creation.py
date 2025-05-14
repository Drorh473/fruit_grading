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

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Get MongoDB connection string from .env
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not MONGODB_CONNECTION_STRING:
    raise ValueError("MongoDB connection string not found in .env file")

# Get dataset path from .env
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
#number of cameras
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS'))
def create_database(db_name=os.getenv('DB_NAME', "fruit_grading"), collection_name="images"):
    """Create MongoDB database for storing image metadata"""
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Create indexes for faster queries
    collection.create_index("category")
    collection.create_index("set_type")
    collection.create_index("condition")
    print(f"Connected to MongoDB database: {db_name}, collection: {collection_name}")
    return db_name, collection_name

def collect_images(dataset_path=ORIGINAL_DATASET_PATH):
    """Collect all images from Mendeley dataset"""
    # Track execution time
    start_time = time.time()
    image_paths = []
    
    # Walk through all directories in the dataset ignoring subdirectories 
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    
    # Report execution time
    print(f"Found {len(image_paths)} images in {time.time() - start_time:.2f} seconds")
    return image_paths

def process_single_image(path):
    """Process a single image and return its metadata"""
    try:
        # Extract category from path seperating by /
        parts = path.split(os.sep)
        category = parts[-3]  # The directory name containing the image
        name = parts[-2]
        if category == "Mixed Qualit_Fruits":
            return None
        # Get image dimensions
        img = Image.open(path)
        width, height = img.size
        img.close()  # Close immediately after getting dimensions
        camera = random.randint(1, NUM_OF_CAMERAS)
        # Create metadata dictionary
        metadata = {
            "path": path,
            "name": name,
            "category": category,
            "width": width,
            "height": height,
            "set_type": "",  # Will be set in split_data
            "camera": camera,
            "color":3
        }
        
        return metadata
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def organize_images(image_paths):
    """Organize images with metadata"""
    # Use parallel processing
    start_time = time.time()
    
    # Determine optimal number of processes (don't use all cores)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Processing images with {num_processes} parallel processes...")
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, image_paths),
            total=len(image_paths),
            desc="Organizing images"
        ))
    
    # Filter out None values (failed processing)
    organized_images = [r for r in results if r is not None]
    
    print(f"Organized {len(organized_images)} images in {time.time() - start_time:.2f} seconds")
    return organized_images

def store_in_database(organized_images, db_name, collection_name):
    """Store image metadata in MongoDB"""
    start_time = time.time()
    print("Starting database storage...")
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Batch insert
    print(f"Inserting {len(organized_images)} records into database...")
    
    # Use bulk insert with ordered=False for better performance
    if organized_images:
        # Insert in batches to avoid memory issues with very large datasets
        batch_size = 1000
        for i in tqdm(range(0, len(organized_images), batch_size), desc="Inserting batches"):
            batch = organized_images[i:i+batch_size]
            collection.insert_many(batch, ordered=False)
    
    print(f"Stored {len(organized_images)} images in {time.time() - start_time:.2f} seconds")
    return db_name, collection_name

def split_data(db_name, collection_name, training_percentage=80, validation_percentage=10, testing_percentage=10):
    """Split data into training and testing sets"""
    # Track performance
    start_time = time.time()
    
    if training_percentage + testing_percentage + validation_percentage != 100:
        raise ValueError("Percentages must sum to 100")
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Get all categories to ensure balanced splits
    categories = collection.distinct("category")
    
    # For each category, split data proportionally
    for category in categories:
        # Get all document IDs for this category
        if category == "Mixed Qualit_fruits":
            continue
        category_docs = list(collection.find({"category": category}, {"_id": 1}))
        ids = [doc["_id"] for doc in category_docs]
        
        # Shuffle IDs randomly to avoid ordering 
        random.shuffle(ids)
        
        # Calculate split point
        training_count = int(len(ids) * (training_percentage / 100))
        # Assign to training set
        training_ids = ids[:training_count]
        if training_ids:
            collection.update_many(
                {"_id": {"$in": training_ids}},
                {"$set": {"set_type": "training"}}
            )
        validation_count = int(len(ids) * (validation_percentage / 100))
        validation_ids=ids[training_count:training_count+validation_count]
        if validation_ids:
            collection.update_many(
                {"_id": {"$in": validation_ids}},
                {"$set": {"set_type": "validation"}}
            )
        # Assign to testing set
        testing_ids = ids[training_count+validation_count:]
        if testing_ids:
            collection.update_many(
                {"_id": {"$in": testing_ids}},
                {"$set": {"set_type": "testing"}}
            )
    
    # Verify the split
    training_count = collection.count_documents({"set_type": "training"})
    testing_count = collection.count_documents({"set_type": "testing"})
    val_count = collection.count_documents({"set_type": "validation"})
    print("Data split results:")
    print(f"  training: {training_count} images")
    print(f"  testing: {testing_count} images")
    print(f"  validation: {val_count} images")
    print(f"Split completed in {time.time() - start_time:.2f} seconds")
    return db_name, collection_name
def copy_image_file(args):
    doc_id, path, name, category, set_type, camera, color, output_dir = args
    category_dir = os.path.join(output_dir, set_type, category)
    dest_path = os.path.join(category_dir, f"{doc_id}.jpg")
    try:
        shutil.copy2(path, dest_path)
        return True
    except Exception as e:
        print(f"Error copying {path}: {e}")
        return False

def create_data_directory_structure(db_name, collection_name, output_dir=STORED_DATASET_PATH):
    """Create directory structure and copy files for training/testing/validation"""
    start_time = time.time()
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    # if this dir already exists delete all the dirs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # make new dirs for testing, training and validation
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Get all images with their metadata
    all_images = list(collection.find({}, {
        "_id": 1, 
        "path": 1, 
        "name": 1, 
        "category": 1,  
        "set_type": 1,
        "camera": 1,
        "color":1
    }))
    
    # Create necessary directories first (one-time operation)
    categories = set()
    for doc in all_images:
        categories.add((doc["set_type"], doc["category"]))
    
    for set_type, category in categories:
        category_dir = os.path.join(output_dir, set_type, category)
        os.makedirs(category_dir, exist_ok=True)
    
    # Add output_dir to each args tuple
    all_images_with_output = [(
        str(doc["_id"]), 
        doc["path"], 
        doc["name"], 
        doc["category"],  
        doc["set_type"],
        doc["camera"],
        doc["color"], 
        output_dir
    ) for doc in all_images]
    
    # Use multiprocessing for file copying
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_image_file, all_images_with_output),
            total=len(all_images_with_output),
            desc="Creating directory structure"
        ))
    
    success_count = sum(1 for r in results if r)
    print(f"Successfully copied {success_count} of {len(all_images)} images in {time.time() - start_time:.2f} seconds")
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
    
    for set_type in ['training', 'testing', 'validation']:
        set_type_dir = os.path.join(data_dir, set_type)
        if not os.path.exists(set_type_dir):
            print(f"Warning: {set_type_dir} does not exist, skipping.")
            continue
        
        for category in os.listdir(set_type_dir):
            category_path = os.path.join(set_type_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            for filename in os.listdir(category_path):
                if not filename.lower().endswith('.jpg'):
                    continue
                    
                files_processed += 1
                
                try:
                    # Correct path construction - full path to the image file
                    image_path = os.path.join(category_path, filename)
                    
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

def print_database_summary(db_name, collection_name):
    """Print summary statistics from the database"""    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Count by category
    category_pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    category_counts = list(collection.aggregate(category_pipeline))
    
    # Count by set_type
    set_type_pipeline = [
        {"$group": {"_id": "$set_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    set_type_counts = list(collection.aggregate(set_type_pipeline))
    
    print("\nDatabase Summary:")
    print("=================")
    
    print("\nCategories:")
    for item in category_counts:
        print(f"  {item['_id']}: {item['count']} images")
    
    print("\nData Splits:")
    for item in set_type_counts:
        print(f"  {item['_id']}: {item['count']} images")


def process_dataset(dataset_path=ORIGINAL_DATASET_PATH, db_name=os.getenv('DB_NAME', "fruit_grading"), collection_name="images"):
    """Process the dataset if not already done"""
    # Create flag file path in local directory
    processed_path = STORED_DATASET_PATH
    flag_file = os.path.join(processed_path, ".processing_complete")
    
    # Check if processing is already complete
    if os.path.exists(flag_file):
        print(f"Dataset already processed. Using existing data.")
        return db_name, collection_name
    
    # If not processed, do the full pipeline
    print("Processing dataset from scratch...")
    start_time = time.time()
    
    # Create database
    db_name, collection_name = create_database(db_name, collection_name)
    
    # Collect images
    image_paths = collect_images(dataset_path)
    
    # Organize images
    organized_images = organize_images(image_paths)
    
    # Store in database
    db_name, collection_name = store_in_database(organized_images, db_name, collection_name)
    
    # Split data
    db_name, collection_name = split_data(db_name, collection_name)
    
    # Create directory structure
    create_data_directory_structure(db_name, collection_name, STORED_DATASET_PATH)
    
    update_data_directory(db_name, collection_name,STORED_DATASET_PATH)
    
    # Print summary
    print_database_summary(db_name, collection_name)
    
    # Create directory for flag file if it doesn't exist
    os.makedirs(os.path.dirname(flag_file), exist_ok=True)
    
    # Create flag file to indicate processing is complete
    with open(flag_file, 'w') as f:
        f.write(f"Processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Dataset processing complete in {time.time() - start_time:.2f} seconds")
    return db_name, collection_name

if __name__ == "__main__":  # Fixed the __main__ check
    # Use environment variables for dataset path
    if not ORIGINAL_DATASET_PATH:
        print("Warning: ORIGINAL_DATASET_PATH not set in .env file")
        dataset_path = input("Please enter the path to the original dataset: ")
    else:
        dataset_path = ORIGINAL_DATASET_PATH
    
    db_name, collection_name = process_dataset(dataset_path)