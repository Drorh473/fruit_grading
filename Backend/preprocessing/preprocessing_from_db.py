import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from dotenv import load_dotenv
import pymongo
import time
from bson.objectid import ObjectId
import random

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
    max_dim = 224 
    if h != max_dim or w != max_dim:
        # Calculate the ratio to resize
        ratio = max_dim / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))
    
    # Save original image for comparison
    original = image.copy()
    
    # 1. Apply Gaussian filter with kernel size 3x3 and sigma=0.001
    blurred = cv2.GaussianBlur(original, (3, 3), 0.001)
    
    # 2. Apply CLAHE for histogram equalization
    # Convert to Lab color space
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


def apply_augmentation(image):
    """
    Apply random augmentations to an image for data augmentation.
    This function applies a combination of transformations to increase dataset diversity.

    Args:
        image: Input image (numpy array, RGB, uint8 or float32)

    Returns:
        augmented: Augmented image in the same format as input
    """
    # Convert to uint8 if needed for OpenCV operations
    is_float = image.dtype == np.float32 and image.max() <= 1.0
    if is_float:
        image = (image * 255).astype(np.uint8)

    augmented = image.copy()
    h, w = augmented.shape[:2]

    # 1. Random Rotation (-30 to +30 degrees) - MORE AGGRESSIVE
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)  # Doubled range
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)

    # 2. Random Horizontal Flip
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # 3. Random Vertical Flip (less common for fruits, but can help)
    if random.random() > 0.7:
        augmented = cv2.flip(augmented, 0)

    # 4. Random Brightness Adjustment - MORE AGGRESSIVE
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.5, 1.5)  # Wider range
        augmented = np.clip(augmented.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    # 5. Random Contrast Adjustment - MORE AGGRESSIVE
    if random.random() > 0.5:
        contrast_factor = random.uniform(0.6, 1.4)  # Wider range
        mean = np.mean(augmented, axis=(0, 1), keepdims=True)
        augmented = np.clip((augmented - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

    # 6. Random Gaussian Noise
    if random.random() > 0.6:
        noise = np.random.normal(0, random.uniform(5, 15), augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7. Random Color Jitter (Hue, Saturation adjustments) - MORE AGGRESSIVE
    if random.random() > 0.5:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust hue - WIDER RANGE
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-20, 20), 0, 179)

        # Adjust saturation - WIDER RANGE
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)

        # Adjust value - WIDER RANGE
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)

        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 8. Random Slight Zoom (crop and resize)
    if random.random() > 0.6:
        zoom_factor = random.uniform(0.85, 0.95)
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        y1 = random.randint(0, h - new_h)
        x1 = random.randint(0, w - new_w)
        cropped = augmented[y1:y1+new_h, x1:x1+new_w]
        augmented = cv2.resize(cropped, (w, h))

    # Convert back to float32 [0,1] if input was float
    if is_float:
        augmented = augmented.astype(np.float32) / 255.0

    return augmented


def process_image(image_path, output_dir, metadata):
    """Process a single image and save the preprocessed version"""
    try:
        # Parse path components
        set_type = metadata.get('set_type')
        camera_id = metadata.get('camera_id')
        file_id = str(metadata.get('_id'))
        
        # Get file extension from original path
        _, ext = os.path.splitext(image_path)
        if not ext:
            ext = '.jpg' 
        
        # Construct output path with same structure
        if set_type and camera_id is not None:
            output_subdir = os.path.join(output_dir, set_type, f"camera_{camera_id}")
        else:
            # Fallback if metadata incomplete
            output_subdir = os.path.join(output_dir, "unknown")
        
        output_path = os.path.join(output_subdir, f"{file_id}{ext}")
        
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
        
        return file_id , output_path , None
    except Exception as e:
        return None, None , f"Error processing {image_path}: {e}"


def preprocess_and_save_dataset(sequanceofcameras, output_dir=None, db_name=os.getenv('DB_NAME', "fruit_grading"), collection_name="images"):
    """
    Preprocess all images and return organized paths for training and testing
    Returns:
        training_paths: List of paths for training images
        testing_paths: List of paths for testing images
        metadata_dict: Dictionary mapping paths to metadata
    """
    output_dir = output_dir or PROCESSED_DATASET_PATH
    print(f"Starting preprocessing and saving images from sequence of cameras to {output_dir}...")
    
    # Connect to database to get set_type info
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    
    # Get all documents to know which images belong to training/testing
    all_docs = list(collection.find())
    
    # Create lookup: original_path -> (set_type, metadata)
    path_info = {}
    for doc in all_docs:
        original_path = doc.get('path')
        path_info[original_path] = {
            'set_type': doc.get('set_type'),
            'fruit_type': doc.get('fruit_type'),
            'object_id': doc.get('object_id'),
            'camera_id': doc.get('camera_id'),
            'timestamp': doc.get('timestamp'),
            '_id': doc.get('_id')
        }
    
    client.close()
    
    # Count total images for progress tracking
    total_images = sum(len(camera) for camera in sequanceofcameras if camera)
    if total_images == 0:
        print("No images to process.")
        return [], [], {}
    
    # Storage for results
    training_paths = []
    testing_paths = []
    metadata_dict = {}
    
    # Process each camera's images
    all_results = []
    for cam_idx, camera in enumerate(sequanceofcameras):
        if not camera:
            continue
            
        start_time = time.time()
        print(f"Processing camera {cam_idx} with {len(camera)} images...")
        
        # Create args with metadata for each image
        process_args = []
        for path in camera:
            metadata = path_info.get(path, {})
            process_args.append((path, output_dir, metadata))
        
        # Use multiprocessing for image processing
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(process_image, process_args),
                total=len(camera),
                desc=f"Camera {cam_idx}"
            ))
        
        # Collect successful results for bulk database update
        updates = []
        errors = 0
        
        for i, (file_id, output_path, error) in enumerate(results):
            if error:
                errors += 1
                if errors <= 10:
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
                    
                    # Get original path and its metadata
                    original_path = camera[i]
                    if original_path in path_info:
                        info = path_info[original_path]
                        set_type = info['set_type']
                        
                        # Add to appropriate list
                        if set_type == 'training':
                            training_paths.append(output_path)
                        elif set_type == 'testing':
                            testing_paths.append(output_path)
                        
                        # Store metadata
                        metadata_dict[output_path] = {
                            'fruit_type': info['fruit_type'],
                            'object_id': info['object_id'],
                            'camera_id': info['camera_id'],
                            'timestamp': info['timestamp']
                        }
                    
                except Exception as e:
                    print(f"Error with ID {file_id}: {e}")
        
        # Bulk update database in batches
        if updates:
            client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
            db = client[db_name]
            collection = db[collection_name]
            
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
        
        all_results.extend(results)
    
    # Final statistics
    success_count = sum(1 for r in all_results if r[0] is not None)
    print(f"Preprocessing complete: {success_count}/{total_images} images successfully processed")
    print(f"All processed images saved to {output_dir}")
    print(f"Training images: {len(training_paths)}")
    print(f"Testing images: {len(testing_paths)}")
    
    return training_paths, testing_paths, metadata_dict


def load_dataset_split_by_camera(db_name=os.getenv('DB_NAME'), collection_name="images"):
    # Make an array of sequence cameras
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    sequenceofcamera = [[] for _ in range(0,NUM_OF_CAMERAS)]
    for image in collection.find():
        camera_id = image.get('camera_id')
        if 0 <= camera_id <= NUM_OF_CAMERAS-1:
            sequenceofcamera[camera_id].append(image.get('path'))
    return sequenceofcamera

def set_generator(image_paths, metadata_dict, augment=False, augment_multiplier=3):
    """
    Create a batch generator for the given image paths
    Args:
        image_paths: List of image paths
        metadata_dict: Dictionary mapping paths to metadata
        augment: Whether to apply data augmentation (default False)
        augment_multiplier: How many augmented versions per image (default 3)
    Returns:
        batch_generator: Generator function
        metadata: Empty dict
        count: Number of images (multiplied by augment_multiplier if augmenting)
    """

    if not image_paths:
        print(f"No images provided")
        return None, {}, 0

    effective_count = len(image_paths) * augment_multiplier if augment else len(image_paths)
    print(f"Creating generator for {len(image_paths)} images" +
          (f" (x{augment_multiplier} augmentation = {effective_count} total)" if augment else ""))

    # Function to generate batches
    def batch_generator():

        # Create indices for the paths
        if augment:
            # Create expanded indices: each image gets augment_multiplier versions
            # Format: (image_index, augmentation_version)
            indices = [(i, aug_idx) for i in range(len(image_paths))
                      for aug_idx in range(augment_multiplier)]
        else:
            # Just image indices
            indices = [(i, 0) for i in range(len(image_paths))]

        # Shuffle
        np.random.shuffle(indices)

        # Calculate number of batches
        effective_count = len(indices)
        num_batches = (effective_count + BATCH_SIZE - 1) // BATCH_SIZE

        # Generate batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, effective_count)
            batch_indices = indices[start_idx:end_idx]

            # Collect valid images
            batch_images = []
            batch_metadata = []

            files_not_found = 0
            files_failed_load = 0

            for img_idx, aug_version in batch_indices:
                img_path = image_paths[img_idx]

                # Check if file exists
                if not os.path.exists(img_path):
                    files_not_found += 1
                    if files_not_found <= 3:  # Show first 3
                        print(f"[Generator] File not found: {img_path}")
                    continue

                try:
                    # Load the image
                    img = cv2.imread(img_path)

                    if img is None:
                        files_failed_load += 1
                        if files_failed_load <= 3:
                            print(f"[Generator] Failed to load: {img_path}")
                        continue

                    # Resize to standardized size
                    img = cv2.resize(img, IMAGE_SIZE)

                    # Convert to RGB (OpenCV loads as BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Apply augmentation if enabled and not the original version (aug_version=0)
                    if augment and aug_version > 0:
                        img = apply_augmentation(img)

                    # Add to batch lists
                    batch_images.append(img)
                    img_data_dict = metadata_dict.get(img_path, {})
                    batch_metadata.append(img_data_dict)
                except Exception as e:
                    print(f"[Generator] Error loading {img_path}: {e}")
            
            if files_not_found > 0:
                print(f"[Generator] {files_not_found} files not found in this batch")
            if files_failed_load > 0:
                print(f"[Generator] {files_failed_load} files failed to load in this batch")
            
            # Skip empty batches
            if not batch_images:
                print(f"[Generator] Batch {batch_idx + 1} is empty, skipping")
                continue
            
            # Convert lists to arrays
            batch_x = np.array(batch_images)
            batch_y = np.array(batch_metadata)
            yield batch_x, batch_y

    # Add metadata to the generator function
    batch_generator.samples = effective_count
    batch_generator.num_batches = (effective_count + BATCH_SIZE - 1) // BATCH_SIZE

    return batch_generator, {}, effective_count

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
    
    # Load sequence of cameras
    print("Loading dataset split by camera...")
    sequence_camera = load_dataset_split_by_camera()
    
    # Check if preprocessing is needed
    if not os.path.isdir(processed_dir):
        # Preprocess images and get organized paths
        print("Preprocessing images...")
        training_paths, testing_paths, metadata_dict = preprocess_and_save_dataset(
            sequence_camera, processed_dir
        )
    else:
        # Load from database if already preprocessed
        print("Loading preprocessed paths from database...")
        client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db["images"]
        
        # Get training paths
        training_docs = list(collection.find({"set_type": "training"}))
        training_paths = [doc.get('processed_path') or doc.get('path') 
                         for doc in training_docs]
        
        # Get testing paths
        testing_docs = list(collection.find({"set_type": "testing"}))
        testing_paths = [doc.get('processed_path') or doc.get('path') 
                        for doc in testing_docs]
        
        # Create metadata dict
        metadata_dict = {}
        for doc in training_docs + testing_docs:
            path = doc.get('processed_path') or doc.get('path')
            metadata_dict[path] = {
                'fruit_type': doc.get('fruit_type'),
                'object_id': doc.get('object_id'),
                'camera_id': doc.get('camera_id'),
                'timestamp': doc.get('timestamp')
            }
        
        client.close()
    
    # Create generators
    print("\nCreating generators...")
    # Training generator WITH augmentation (5x multiplier for more diversity)
    train_gen, _, train_count = set_generator(training_paths, metadata_dict,
                                              augment=True, augment_multiplier=5)
    # Testing generator WITHOUT augmentation
    test_gen, _, test_count = set_generator(testing_paths, metadata_dict,
                                            augment=False)

    # Print summary
    print("\nDatasets summary:")
    print(f"  Training: {train_count} images (with augmentation)")
    print(f"  Testing: {test_count} images (no augmentation)")

    # Return generators
    return train_gen, test_gen