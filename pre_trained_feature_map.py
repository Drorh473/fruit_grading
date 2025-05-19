import os
import numpy as np
import torch
import glob
import time
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import pymongo

# Import the trained CNN
from pre_trained_cnn import load_model, extract_features


# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Get configuration from .env
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
DB_NAME = os.getenv('DB_NAME', 'fruit_rotation_db')

def get_image_time_from_db(image_path):
    """
    Get the timestamp/orientation information for an image from the database
    
    Args:
        image_path: Path to the image
        
    Returns:
        Timestamp or orientation value (integer)
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db["images"]
        
        # Extract object_id from the filename
        filename = os.path.basename(image_path)
        file_id = os.path.splitext(filename)[0]
        
        # Get the document from MongoDB
        document = collection.find_one({"_id": file_id})
        if document:
            # Extract orientation/timestamp
            time = document.get("timestamp", 0)
            return time
        
        # Default fallback
        return 0
        
    except Exception as e:
        print(f"Error getting time from DB for {image_path}: {e}")
        return 0
    finally:
        if 'client' in locals():
            client.close()


def batch_extract_features(input_dir, file_extensions='.png'):
    """
    Extract features for all images in a directory and return them as a dictionary
    
    Args:
        input_dir: Directory containing images
        file_extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
        use_custom_preprocessing: Whether to use custom preprocessing
        
    Returns:
        Dictionary mapping image paths to their feature maps
    """    
    # Load model once for all images
    _, feature_extractor, device = load_model(pretrained=True)
    
    # Get all image files in the directory
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(input_dir, f'*{file_extensions}')))
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    input_dir_paths = input_dir.split(os.sep)
    fruit = input_dir_paths[-2]
    obj_num = input_dir_paths[-1]
    
    print(f"Found {len(image_paths)} images to process from {fruit} in {obj_num}")
    
    # Dictionary to store results
    processing_times = []
    feature_map = {}
    # Process each image
    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            # Time the feature extraction
            start_time = time.time()
            
            # Load image and extract features
            # For batch processing, we load the image in extract_features to avoid duplication
            with torch.no_grad():
                img = torch.zeros(1, 3, 224, 224).to(device)  # Dummy tensor for initial load
                features_tensor = feature_extractor(img)  # Dummy forward pass
                
            # Extract features with custom preprocessing
            features = extract_features(image_path, use_custom_preprocessing=True)
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Get time value from database
            time_value = get_image_time_from_db(image_path)
            
            # Reshape features to include time dimension
            features = np.transpose(features, (0, 2, 3, 1))[0]
            
            feature_map[image_path] = {
                'featuremap': features,  
                'timestamp' : time_value ,          
            }  
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Print processing time statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        total_time = sum(processing_times)
        print(f"\nProcessing complete. Extracted features for {len(feature_map)} images")
        print(f"Average processing time per image: {avg_time:.4f} seconds")
        print(f"Minimum processing time: {min_time:.4f} seconds")
        print(f"Maximum processing time: {max_time:.4f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")
    
    return feature_map

def group_features_by_object(feature_maps):
    """
    Group feature maps by object_id for temporal processing
    
    Args:
        feature_maps: Dictionary of feature maps by image path
        
    Returns:
        Dictionary with object_ids as keys and lists of features sorted by time as values
    """
    # Dictionary to group features by object
    object_features = {}
    
    # Process all feature maps
    for image_path , data in feature_maps.items():
        # Extract object_id from the path
        path_parts = image_path.split(os.sep)
        
        # Look for object_id in the path
        fruit_name = path_parts[-3]
        object_id = path_parts[-2]
        final_object_name = fruit_name + object_id

        # Initialize list for this object if not already present
        if final_object_name not in object_features:
            object_features[final_object_name] = []
        
        # Add features with time value to the list
        object_features[final_object_name] = ({
            'features': data['featuremap'],
            'timestamp': data['timestamp'],
        })
    
    return object_features

def get_all_feature_maps(dataset_type='training', fruit_type=None):
    """
    Extract features for all images of a specific dataset type and optionally a specific fruit type
    
    Args:
        dataset_type: 'training' or 'testing'
        fruit_type: Optional filter for a specific fruit type (e.g., 'mandarins')
        
    Returns:
        Dictionary mapping image paths to their feature maps
    """
    # Build the input directory path
    if fruit_type:
        input_dir = os.path.join(PROCESSED_DATASET_PATH, dataset_type, fruit_type)
    else:
        input_dir = os.path.join(PROCESSED_DATASET_PATH, dataset_type)
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        print(f"Available directories in {os.path.dirname(input_dir)}:")
        for item in os.listdir(os.path.dirname(input_dir)):
            if os.path.isdir(os.path.join(os.path.dirname(input_dir), item)):
                print(f"  - {item}")
        return {}
    feature_maps = {}
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        for obj in os.listdir(category_path):
            obj_path = os.path.join(category_path, obj)
            if not os.path.isdir(obj_path):
                continue
            feature_maps.update(batch_extract_features(obj_path))
    # Group features by object_id
    print("Grouping features by object...")
    object_features = group_features_by_object(feature_maps)
    
    return object_features

def flatten_features(feature_map):
    """
    Apply flattening to feature maps
    
    Args:
        feature_map: Dictionary with feature maps
        
    Returns:
        Dictionary with flattened features
    """
    # Create new dictionary to store results
    flattened_features = {}
    
    for obj_id, data in feature_map.items():
        # Get the feature data
        feature_data = data['features']
        
        # Get shape information (h,w,c)
        h, w, c = feature_data.shape
        
        # Flatten the features from (h,w,c) to a 1D vector (h*w*c)
        flattened_data = np.reshape(feature_data, (h * w * c))
        
        # Store the flattened data
        flattened_features[obj_id] = {
            'features': flattened_data,
            'timestamp': data['timestamp']
        }
    
    return flattened_features

def temporal_pooling(flattened_features):
    """
    Pool feature vectors across time frames for each object using AveragePooling1D.
    
    Args:
        flattened_features: {obj_id: {'features': array, 'timestamp': value}}
    Returns:
        {obj_id: pooled_feature_vector}
    """
    pooled_vector = {} 
    grouped_features = {}

    for obj_id, data in flattened_features.items():
         # Initialize list for new objects
        if obj_id not in grouped_features:
            grouped_features[obj_id] = []

        # Add features to list
        features = data['features']
        grouped_features[obj_id].append(features)
    
    # Average features for each object
    for obj_id, features_list in grouped_features.items():
        if len(features_list) == 1:
            # Single frame - no averaging needed
            pooled_vector[obj_id] = features_list[0]
        else:
            # Multiple frames - stack and average across time dimension
            stacked_features = np.stack(features_list)
            # axis=0 means average across frames dimension
            # (num_frames, feature_dim) -> (feature_dim,)
            pooled_vector[obj_id] = np.mean(stacked_features, axis=0)

    return pooled_vector

def main():
    """
    Main function to extract features for all images and organize them by time
    
    Returns:
        Dictionary of temporal feature maps grouped by object_id,
        with features in shape (t, h, w, c) where t is time/orientation
    """
    # Extract features for training dataset
    training_features = get_all_feature_maps(dataset_type='training')
    
    # Extract features for testing dataset
    testing_features = get_all_feature_maps(dataset_type='testing')
    
    # Combine results
    all_features = {**training_features, **testing_features}
    
    print(f"\nTotal temporal feature maps created: {len(all_features)}")
    print(f"  Training: {len(training_features)}")
    print(f"  Testing: {len(testing_features)}")
    
    flatten_all_features = flatten_features(all_features)
    return flatten_all_features

if __name__ == "__main__":
    # Extract features for all images
    feature_maps_after_flattening = main()
    print("\nFlattening result: ", feature_maps_after_flattening)
    pooled_vector = temporal_pooling(feature_maps_after_flattening)
    print("\nPooling result: ", pooled_vector)