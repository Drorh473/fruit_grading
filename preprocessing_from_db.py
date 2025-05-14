import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from dotenv import load_dotenv
import pymongo
import time
from bson.objectid import ObjectId
# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
# Get MongoDB connection string from .env
MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not MONGODB_CONNECTION_STRING:
    raise ValueError("MongoDB connection string not found in .env file")
# Get dataset paths from environment
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS'))
IMAGE_SIZE = os.getenv('IMAGE_SIZE')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))


def custom_preprocessing(image, save_path=None):
    """Apply preprocessing steps from the article:
    1. Gaussian filtering for noise removal
    2. CLAHE for histogram equalization
    
    Args:
        image: Input image to process
        save_path: If provided, save the processed image to this path
    """
    # Make sure image is in uint8 format for OpenCV
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    max_dim = 224  # Maximum dimension size
    if h != max_dim or w != max_dim:
        # Calculate the ratio to resize
        ratio = max_dim / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))
    
    # Save original image for comparison
    original = image.copy()
    
    # 1. Apply Gaussian filter with kernel size 3x3 and sigma=0.01
    blurred = cv2.GaussianBlur(original, (3, 3), 0.001)
    
    # 2. Apply CLAHE for histogram equalization (on grayscale version)
    # Convert to Lab color space (L channel is the lightness)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Save the processed image if a path is provided
    if save_path:
        # Convert to BGR for OpenCV saving
        save_img = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, save_img)
    
    # Normalize to [0,1] range for the neural network
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced

# Define this at the module level (outside any other function)
def process_image(args):
    image_path, output_dir = args  # Now getting output_dir from args
    try:
        # Parse path components
        parts = image_path.split(os.sep)
        image_type = parts[-4] if len(parts) > 3 else "unknown"
        quality_type = parts[-3] if len(parts) > 2 else "unknown"
        fruit_type = parts[-2] if len(parts) > 1 else "unknown"
        filename = parts[-1]
        
        # Construct output path
        output_subdir = os.path.join(output_dir, quality_type, image_type, fruit_type)
        output_path = os.path.join(output_subdir, filename)
        
        # Ensure output directory exists
        os.makedirs(output_subdir, exist_ok=True)
        
        # Read and process image
        img = cv2.imread(image_path)
        if img is None:
            return None, None, f"Could not read {image_path}"
            
        # Convert to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Apply preprocessing and save
        custom_preprocessing(img, save_path=output_path)
        
        # Extract document ID from filename
        file_id = os.path.splitext(filename)[0]
        
        return file_id, output_path, None  # Success
    except Exception as e:
        return None, None, f"Error processing {image_path}: {e}"

def preprocess_and_save_dataset(sequanceofcameras, output_dir=None, db_name=os.getenv('DB_NAME', "fruit_grading"), collection_name="images"):
    """Preprocess all images in the sequence of cameras and save them to the output directory"""
    output_dir = output_dir or PROCESSED_DATASET_PATH
    print(f"Starting preprocessing and saving images from sequence of cameras to {output_dir}...")
    
    # Count total images for progress tracking
    total_images = sum(len(camera) for camera in sequanceofcameras if camera)
    if total_images == 0:
        print("No images to process.")
        return output_dir
    
    # Process each camera's images
    all_results = []
    for cam_idx, camera in enumerate(sequanceofcameras):
        if not camera:
            continue
            
        start_time = time.time()
        print(f"Processing camera {cam_idx} with {len(camera)} images...")
        
        # Create args with camera number and output_dir for tracking
        process_args = [(path, output_dir) for path in camera]  # Added output_dir to args
        
        # Use multiprocessing for image processing
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_image, process_args),
                total=len(camera),
                desc=f"Camera {cam_idx}"
            ))
        
        # Collect successful results for bulk database update
        updates = []
        errors = 0
        for file_id, output_path, error in results:
            if error:
                errors += 1
                if errors <= 10:  # Limit error output
                    print(error)
                elif errors == 11:
                    print("Too many errors, suppressing further messages...")
            elif file_id and output_path:
                try:
                    object_id = ObjectId(file_id)
                    updates.append(pymongo.UpdateOne(
                        {'_id': object_id},
                        {'$set': {'processed_path': output_path}}
                    ))
                except Exception as e:
                    print(f"Error with ID {file_id}: {e}")
        
        # Bulk update database in batches
        if updates:
            client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
            db = client[db_name]
            collection = db[collection_name]
            
            # Update in batches of 500
            batch_size = 500
            update_count = 0
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                try:
                    result = collection.bulk_write(batch, ordered=False)
                    update_count += result.modified_count
                except Exception as e:
                    print(f"Database error during batch update: {e}")
            
            client.close()
            
            print(f"Updated {update_count} database records for camera {cam_idx}")
        
        elapsed_time = time.time() - start_time
        print(f"Camera {cam_idx} processing completed in {elapsed_time:.2f} seconds")
        
        # Save results for reporting
        all_results.extend(results)
    
    # Final statistics
    success_count = sum(1 for r in all_results if r[0] is not None)
    print(f"Preprocessing complete: {success_count}/{total_images} images successfully processed")
    print(f"All processed images saved to {output_dir}")


def load_dataset_split_by_Camera(db_name=os.getenv('DB_NAME'), collection_name="images"):
    #make an array of sequence cameras
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    sequenceofcamera = [[] for _ in range(NUM_OF_CAMERAS+1)]  # Create list of empty lists
    for image in collection.find():
        camera_id = image.get('camera')
        if 1 <= camera_id <= NUM_OF_CAMERAS:
            sequenceofcamera[camera_id].append(image.get('path'))
    return sequenceofcamera

def set_generator(set_type, db_name=os.getenv('DB_NAME'), collection_name="images", show_progress=True):
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        # Get documents for this set_type
        documents = list(collection.find({"set_type": set_type}))
        
        if not documents:
            print(f"No documents found for set_type: {set_type}")
            return None, {}, 0
        
        # Get categories and create class indices
        categories = sorted(list(set(doc["category"] for doc in documents if "category" in doc)))
        class_indices = {category: i for i, category in enumerate(categories)}
        
        print(f"Found {len(documents)} images for '{set_type}' with {len(categories)} categories")
        
        # Function to generate batches
        def batch_generator():
            # Create indices for the documents
            indices = list(range(len(documents)))
            
            # Shuffle for training and validation
            if set_type in ['training', 'validation']:
                np.random.shuffle(indices)
            
            # Calculate number of batches
            num_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
            
            # Create iterator with optional progress bar
            batch_range = range(num_batches)
            if show_progress:
                batch_range = tqdm(batch_range, desc=f"{set_type} batches")
            
            # Generate batches
            for batch_idx in batch_range:
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(documents))
                batch_indices = indices[start_idx:end_idx]
                
                # We'll collect valid images and labels first
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    doc = documents[idx]
                    
                    # Use processed_path if it exists, otherwise use regular path
                    img_path = doc.get("processed_path", doc.get("path"))
                    if not img_path or "category" not in doc:
                        continue
                    
                    try:
                        # Load the image
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (224, 224))
                        
                        if img is None:
                            continue
                        
                        # Convert to RGB (OpenCV loads as BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Create one-hot label
                        class_idx = class_indices[doc["category"]]
                        label = np.zeros(len(categories))
                        label[class_idx] = 1.0
                        
                        # Add to batch lists
                        batch_images.append(img)
                        batch_labels.append(label)
                    
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                
                # Skip empty batches
                if not batch_images:
                    continue
                
                # Convert lists to arrays
                batch_x = np.array(batch_images , dtype=np.uint8)
                batch_y = np.array(batch_labels , dtype=np.uint8)
                
                yield batch_x, batch_y
        
        # Add metadata to the generator function
        batch_generator.samples = len(documents)
        batch_generator.num_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_generator.class_indices = class_indices
        
        return batch_generator, class_indices, len(documents)
    
    except Exception as e:
        print(f"Error retrieving set type '{set_type}': {e}")
        return None, {}, 0
    
    finally:
        # Close the MongoDB connection
        client.close()
    
def load_dataset_with_preprocessing(sequenceofcamera):
    """
    Load dataset with custom preprocessing
    
    Args:
        sequanceofcamera: array of arrays containing the images paths
    """
    if not os.path.isdir(PROCESSED_DATASET_PATH):
        preprocess_and_save_dataset(sequenceofcamera, PROCESSED_DATASET_PATH)
    train_gen, class_indices, train_count = set_generator("training")
    test_gen,_, test_count = set_generator("testing")
    validation_gen, _, validation_count = set_generator("validation")
        
      # Create a dictionary of counts
    counts = {
        "training": train_count,
        "validation": validation_count,
        "testing": test_count
    }
    
    # Print summary
    print("\nDatasets summary:")
    print(f"  Classes: {list(class_indices.keys())}")
    print(f"  Training: {train_count} images")
    print(f"  Validation: {validation_count} images")
    print(f"  Testing: {test_count} images")
    
    # Return all the generators and related information
    return train_gen, validation_gen, test_gen, class_indices, counts

def preprocessing_from_db():
    # Check if PROCESSED_DATASET_PATH is set in .env
    if not PROCESSED_DATASET_PATH:
        print("Warning: PROCESSED_DATASET_PATH not set in .env file. Using default 'processed_dataset'.")
        processed_dir = "processed_dataset"
    else:
        processed_dir = PROCESSED_DATASET_PATH
        
    sequance = load_dataset_split_by_Camera()
    
    train_gen, val_gen, test_gen, class_indices, counts = load_dataset_with_preprocessing(sequance)
    
    return train_gen, val_gen, test_gen
