import os
import numpy as np
import torch
import glob
import time
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import pymongo
from bson.objectid import ObjectId

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

def get_image_time_and_camera_from_db(image_path):
    """
    Get the timestamp and camera_id information for an image from the database
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (timestamp, camera_id) or (None, None) if not found 
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db["images"]
        
        # Extract object_id from the filename
        filename = os.path.basename(image_path)
        file_id = os.path.splitext(filename)[0]

        # Convert to ObjectId 
        object_id = ObjectId(file_id) 

        # Get the document from MongoDB
        document = collection.find_one({"_id": object_id})
        if document:
            # Extract timestamp and camera_id
            timestamp = document.get("timestamp")
            camera_id = document.get("camera_id")
            return timestamp, camera_id
        
        # Default fallback
        print(f"No document found for image: {image_path}")
        return None, None
        
    except Exception as e:
        print(f"Error getting time from DB for {image_path}: {e}")
        return None, None

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
            
            # Check if file exists before processing
            if not os.path.isfile(image_path):
                print(f"File not found: {image_path}")
                continue

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
            time_value, camera_id = get_image_time_and_camera_from_db(image_path)
            
            # Reshape features to include time dimension
            features = np.transpose(features, (0, 2, 3, 1))[0]
            
            feature_map[image_path] = {
                'featuremap': features,  
                'timestamp' : time_value,  
                'camera_id': camera_id         
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
        camera_id = data.get('camera_id')

         # Skip entries without camera_id
        if camera_id is None:
            print(f"Warning: No camera_id for {image_path}, skipping")
            continue
        
        # Create a unique key combining fruit type, object_id, and camera_id
        final_object_key = f"{fruit_name}{object_id}_{camera_id}"

        # Initialize list for this object if not already present
        if final_object_key not in object_features:
            object_features[final_object_key] = {
                'features': data['featuremap'],
                'timestamp': data['timestamp'],
                'camera_id': camera_id,
                'base_obj': f"{fruit_name}{object_id}"
            }
    
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
            'timestamp': data['timestamp'],
            'camera_id': data['camera_id'],
            'base_obj': data['base_obj']
        }
    
    return flattened_features

def temporal_pooling(flattened_features):
    """
    Pool feature vectors across time frames for each object using AveragePooling1D.
    
    Args:
        flattened_features: Dictionary {obj_id: {'features': array, 'timestamp': value, 'camera_id': id}}
    Returns:
       Dictionary {obj_id: pooled_feature_vector}
    """
    # Dictionary to store the pooled feature vectors
    pooled_vector = {} 
    grouped_features = {}

    for obj_key, data in flattened_features.items():
         # Initialize list for new objects
        if obj_key not in grouped_features:
            grouped_features[obj_key] = []

        # Add this feature and its timestamp to the list
        grouped_features[obj_key].append({
            'features': data['features'],
            'timestamp': data['timestamp'],
            'camera_id': data['camera_id'],
            'base_obj': data['base_obj']
        })
    
    # Average features for each object
    for obj_key, features_list in grouped_features.items():
        # If we have timestamp information, sort features by timestamp
        if features_list[0]['timestamp'] is not None:
            try:
                features_list.sort(key=lambda x: x['timestamp'])
            except Exception as e:
                print(f"Warning: Could not sort features by timestamp for {obj_id}: {e}")

        # Extract just the feature arrays
        feature_arrays = [item['features'] for item in features_list]

        if len(feature_arrays) == 1:
            # Single frame - no averaging needed
            # Store only the feature vector
            pooled_vector[obj_key] = feature_arrays[0]

        else:
            # Multiple frames - stack and average across time dimension
            stacked_features = np.stack(feature_arrays)
            # axis=0 means average across frames dimension
            # (num_frames, feature_dim) -> (feature_dim,)
            pooled_vector[obj_key] = np.mean(stacked_features, axis=0)

    return pooled_vector

def multi_view_fusion(pooled_vectors):
    """
    Combine feature vectors from multiple camera views for each object.
    
    Args:
        pooled_vectors: Dictionary {obj_id: feature_array} containing pooled feature vectors
    Returns:
        Dictionary {object_id: fused_feature_vector}
    """
    # Dictionary to store the fused feature vectors
    fused_vectors = {}
    grouped_vectors = {}
    
    # Extract base object IDs from the combined fruit_type and object_id
    for obj_key, feature_vector in pooled_vectors.items():
        # Extract base object ID and camera ID from obj_key
        # Format is "fruit_typeobj_id_camera_id"
        parts = obj_key.split('_')
        if len(parts) >= 2:
            # Last part is camera_id, everything before is the base object
            base_obj = parts[0]
        else:
            # If no underscore, use the whole key
            base_obj = obj_key
        
        # Initialize list for this base object if not already present
        if base_obj not in grouped_vectors:
            grouped_vectors[base_obj] = []
        
        # Add this feature vector to the list
        grouped_vectors[base_obj].append(feature_vector)
    
    # Apply fusion method to each group
    for base_obj, vectors in grouped_vectors.items():
        # Extract the object number for consistent output naming
        obj_pos = base_obj.find('obj')
        if obj_pos != -1:
            obj_num = base_obj[obj_pos:]  # e.g., 'obj0003'
        else:
            obj_num = base_obj
            
        # Concatenate all feature vectors
        if len(vectors) > 1:
            # If we have multiple vectors, concatenate them
            fused_vector = np.concatenate(vectors)
        else:
            # If we have only one vector, use it as is
            fused_vector = vectors[0]
            
        # Store the fused vector with the base object ID
        fused_vectors[base_obj] = fused_vector
    
    # Print some statistics about the fusion
    print(f"\nMulti-view fusion complete using concateante method")
    print(f"Number of base objects after fusion: {len(fused_vectors)}")
    
    # Calculate and print the dimensionality of fused vectors
    if fused_vectors:
        # Get dimension of first fused vector
        sample_dim = next(iter(fused_vectors.values())).shape[0]
        print(f"Fused vector dimensionality: {sample_dim}")
        
        # Get average number of cameras per object
        avg_cameras = sum(len(v) for v in grouped_vectors.values()) / len(grouped_vectors)
        print(f"Average cameras per object: {avg_cameras:.2f}")
        
    return fused_vectors

def main():
    """
    Main function to extract features for all images and organize them by time
    
    Returns:
        Dictionary of temporal feature maps grouped by object_id,
        with features in shape (t, h, w, c) where t is time
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
    
    # Flatten features
    flatten_all_features = flatten_features(all_features)

    # Temporal pooling 
    pooled_vectors = temporal_pooling(flatten_all_features)
    print("\nPooling result (sample):")
    if pooled_vectors:
        for key in list(pooled_vectors.keys())[:3]:  # Show first 3 items
            if isinstance(pooled_vectors[key], dict):
                print(f"{key}: shape {pooled_vectors[key]['features'].shape}, camera_id: {pooled_vectors[key]['camera_id']}")
            else:
                print(f"{key}: shape {pooled_vectors[key].shape}")
    
    # Mult-view fusion
    fused_vectors = multi_view_fusion(pooled_vectors)
    print("\nFusion result (sample):")
    if fused_vectors:
        for key in list(fused_vectors.keys())[:3]:  # Show first 3 items
            print(f"{key}: shape {fused_vectors[key].shape}")

    return fused_vectors

if __name__ == "__main__":
    fused_feature_vectors = main()