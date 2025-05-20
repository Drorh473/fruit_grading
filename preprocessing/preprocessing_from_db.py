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
DB_NAME = os.getenv('DB_NAME', 'fruit_rotation_db')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
IMAGE_SIZE = (224, 224)
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS'))

def custom_preprocessing(image, save_path=None):
    """Apply preprocessing steps:
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

def process_image(args):
    """Process a single image and save the preprocessed version"""
    image_path, camera_num, output_dir = args
    try:
        # Parse path components
        parts = image_path.split(os.sep)
        
        # Extract components from the path
        filename = parts[-1]
        object_id = parts[-2]
        fruit_type = parts[-3]
        set_type = parts[-4]
        
        # Construct output path with same structure
        if set_type and fruit_type and object_id:
            output_subdir = os.path.join(output_dir, set_type, fruit_type, object_id)
        else:
            # Fallback if path structure not recognized
            output_subdir = os.path.join(output_dir, "unknown")
        
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
        file_id = filename.split(".")[0]
        
        return file_id , output_path , None  # Success
    except Exception as e:
        return None, None , f"Error processing {image_path}: {e}"


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
        process_args = [(path, cam_idx, output_dir) for path in camera]  # Added output_dir to args
        
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


def load_dataset_split_by_camera(db_name=os.getenv('DB_NAME'), collection_name="images"):
    #make an array of sequence cameras
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    sequenceofcamera = [[] for _ in range(NUM_OF_CAMERAS+1)]  # Create list of empty lists
    for image in collection.find():
        camera_id = image.get('camera_id')
        if 1 <= camera_id <= NUM_OF_CAMERAS:
            sequenceofcamera[camera_id].append(image.get('path'))
    return sequenceofcamera

def set_generator(set_type, db_name=DB_NAME, collection_name="images", show_progress=True):
    """Create a batch generator for the specified set type"""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        # Get documents for this set_type
        documents = list(collection.find({"set_type": set_type}))
        
        if not documents:
            print(f"No documents found for set_type: {set_type}")
            return None, {}, 0
        
        print(f"Found {len(documents)} images for '{set_type}'")
        
        # Function to generate batches
        def batch_generator():
            # Create indices for the documents
            indices = list(range(len(documents)))
            
            # Shuffle for training
            if set_type == 'training':
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
                
                # We'll collect valid images
                batch_images = []
                batch_orientations = []
                
                for idx in batch_indices:
                    doc = documents[idx]
                    
                    # Use processed_path if it exists, otherwise use regular path
                    img_path = doc.get("processed_path", doc.get("path"))
                    if not img_path:
                        continue
                    
                    try:
                        # Load the image
                        img = cv2.imread(img_path)
                        
                        if img is None:
                            continue
                        
                        # Resize to standardized size
                        img = cv2.resize(img, IMAGE_SIZE)
                        
                        # Convert to RGB (OpenCV loads as BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Get orientation
                        orientation = doc.get("orientation", 0)
                        
                        # Add to batch lists
                        batch_images.append(img)
                        batch_orientations.append(orientation)
                    
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                
                # Skip empty batches
                if not batch_images:
                    continue
                
                # Convert lists to arrays
                batch_x = np.array(batch_images)
                batch_y = np.array(batch_orientations)
                
                yield batch_x, batch_y
        
        # Add metadata to the generator function
        batch_generator.samples = len(documents)
        batch_generator.num_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        
        return batch_generator, {}, len(documents)
    
    except Exception as e:
        print(f"Error retrieving set type '{set_type}': {e}")
        return None, {}, 0
    
    finally:
        # Close the MongoDB connection
        client.close()

def load_dataset_with_preprocessing():
    """
    Load dataset with custom preprocessing
    
    Returns:
        train_gen, test_gen: Generators for training and testing data
    """
    # Check if PROCESSED_DATASET_PATH is set in .env
    if not PROCESSED_DATASET_PATH:
        print("Warning: PROCESSED_DATASET_PATH not set in .env file. Using default 'processed_dataset'.")
        processed_dir = "processed_dataset"
    else:
        processed_dir = PROCESSED_DATASET_PATH
    
    # Get image paths by set type

    sequencecamera = load_dataset_split_by_camera()
    # Preprocess images if needed
    if not os.path.isdir(processed_dir):
        preprocess_and_save_dataset(sequencecamera, processed_dir)
    
    # Create generators for training and testing
    train_gen, _, train_count = set_generator("training")
    test_gen, _, test_count = set_generator("testing")
    
    # Get counts
    counts = {
        "training": train_count,
        "testing": test_count
    }
    
    # Print summary
    print("\nDatasets summary:")
    print(f"  Training: {train_count} images")
    print(f"  Testing: {test_count} images")
    
    # Return generators and counts
    return train_gen, test_gen

def preprocessing_from_db():
    # Load dataset with preprocessing
    train_gen, test_gen= load_dataset_with_preprocessing()
    
    # Return generators
    return train_gen, test_gen

def main():
    """Main function to demonstrate preprocessing and data loading"""
    print("=== Fruit Dataset Preprocessing and Loading ===\n")
    
    # Check if PROCESSED_DATASET_PATH is set in .env
    if not PROCESSED_DATASET_PATH:
        print("Warning: PROCESSED_DATASET_PATH not set in .env file. Using default 'processed_dataset'.")
        processed_dir = "processed_dataset"
    else:
        processed_dir = PROCESSED_DATASET_PATH
        print(f"Using processed dataset directory: {processed_dir}")
    
    # Step 1: Load dataset paths
    print("\nStep 1: Loading dataset paths from database...")
    sequence_of_sets = load_dataset_split_by_camera()
    
    # Step 2: Preprocess images if needed
    if not os.path.isdir(processed_dir):
        print("\nStep 2: Preprocessing images and saving to disk...")
        preprocess_and_save_dataset(sequence_of_sets, processed_dir)
    else:
        print(f"\nStep 2: Skipping preprocessing as directory already exists: {processed_dir}")
    
    # Step 3: Create dataset generators
    print("\nStep 3: Creating dataset generators...")
    train_gen, test_gen = preprocessing_from_db()
    
    # Step 4: Display some example images
    if train_gen:
        print("\nStep 4: Displaying sample images from training set...")
        try:
            # Get a batch from the training generator
            batch_x, batch_y = next(train_gen())
            
            # Display a few images with their orientations
            plt.figure(figsize=(15, 5))
            for i in range(min(5, len(batch_x))):
                plt.subplot(1, 5, i+1)
                plt.imshow(batch_x[i])
                plt.title(f"Orientation: {batch_y[i]}")
                plt.axis('off')
            
            plt.tight_layout()
            
            # Create MODEL_DIR if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save figure
            save_path = os.path.join(MODEL_DIR, "sample_training_images.png")
            plt.savefig(save_path)
            print(f"Sample images saved to: {save_path}")
            
            # Show plot
            plt.show()
            
        except Exception as e:
            print(f"Error displaying sample images: {e}")
    else:
        print("No training generator available.")
    
    # Step 5: Display dataset statistics
    print("\nStep 5: Dataset statistics")
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db["images"]
    
    # Get counts by set type
    training_count = collection.count_documents({"set_type": "training"})
    testing_count = collection.count_documents({"set_type": "testing"})
    
    # Get counts by fruit type
    fruit_pipeline = [
        {"$group": {"_id": "$fruit_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]  
    # Print statistics
    print(f"Total images: {training_count + testing_count}")
    print(f"Training images: {training_count}")
    print(f"Testing images: {testing_count}")
    
    # Close MongoDB connection
    client.close()
    
    print("\n=== Preprocessing and data loading complete ===")

if __name__ == "__main__":
    main()