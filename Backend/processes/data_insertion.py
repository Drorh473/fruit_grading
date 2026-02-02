import os
import sys
import time
import json
import shutil
import re
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Streamers.database_insertion import (
    collect_images_metadata,
    copy_images_to_stored,
)

from preprocessing.preprocessing_insertion import (
    preprocess_images_batch,
    create_generator_for_object
)

from cnn.feature_map_insertion import (
    extract_and_fuse_features,
    get_feature_vector
)

from cnn.fully_connected_layer import load_model, predict
from cnn.fine_tune_classifier import predict_multiple_images, clear_model_cache

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_DIR, 'saved_models'))
METADATA_FILE_PATH = os.path.join(PROJECT_DIR, 'saved_models', 'dashboard_metadata.json')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH', os.path.join(PROJECT_DIR))

def update_dashboard_metadata(predicted_type, image_count):
    if not os.path.exists(METADATA_FILE_PATH):
        print(f"Warning: Metadata file not found at {METADATA_FILE_PATH}")
        return

    try:
        with open(METADATA_FILE_PATH, 'r') as f:
            data = json.load(f)

        # Update dataset info
        if 'dataset_info' in data:
            # Update totals
            data['dataset_info']['total_objects'] = data['dataset_info'].get('total_objects', 0) + 1
            data['dataset_info']['total_images'] = data['dataset_info'].get('total_images', 0) + image_count

            # Update split counts (new data goes to testing/inference)
            data['dataset_info']['testing_count'] = data['dataset_info'].get('testing_count', 0) + 1

            # Update class distribution
            if 'class_distribution' in data['dataset_info']:
                if predicted_type in data['dataset_info']['class_distribution']:
                    dist = data['dataset_info']['class_distribution'][predicted_type]
                    dist['total'] = dist.get('total', 0) + 1
                    dist['test'] = dist.get('test', 0) + 1
                else:
                    # Add new class if not exists
                    data['dataset_info']['class_distribution'][predicted_type] = {
                        'train': 0,
                        'test': 1,
                        'total': 1
                    }

        # Note: We do NOT update the model's performance metrics (avg_confidence, test_accuracy, etc.)
        # These values represent the trained model's performance and should remain constant.
        # The confidence parameter is only used for individual prediction results, not to update model metrics.

        # Update timestamp
        from datetime import datetime
        data['timestamp'] = datetime.now().isoformat()

        with open(METADATA_FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)

        print("Dashboard metadata updated successfully.")

    except Exception as e:
        print(f"Failed to update dashboard metadata: {e}")
        import traceback
        traceback.print_exc()

def get_next_object_id(db_name, collection_name, predicted_type):
    """Get next available object ID by checking the original dataset folder."""
    fruit_type_dir = os.path.join(ORIGINAL_DATASET_PATH, predicted_type)

    max_num = 0

    # Check if the fruit type directory exists
    if os.path.exists(fruit_type_dir) and os.path.isdir(fruit_type_dir):
        # List all directories in the fruit type folder
        folders = [d for d in os.listdir(fruit_type_dir)
                   if os.path.isdir(os.path.join(fruit_type_dir, d))]

        print(f"[DEBUG] Found {len(folders)} folders in {fruit_type_dir}")

        for folder_name in folders:
            # Parse folder names starting with "obj"
            if folder_name.startswith("obj"):
                numbers = re.findall(r'\d+', folder_name)
                if numbers:
                    num = int(numbers[0])
                    # print(f"[DEBUG] Folder: {folder_name} -> parsed number: {num}")
                    max_num = max(max_num, num)
    else:
        print(f"[DEBUG] Directory does not exist: {fruit_type_dir}")

    # Calculate next number
    next_num = max_num + 1
    
    print(f"[DEBUG] Max number found: {max_num}, next will be: {next_num}")
    
    # Return the formatted object ID
    return f"obj_{next_num}_{predicted_type}"


def run_classification_model_fine_tuned(image_paths):
    """
    Run fine-tuned model on image paths and return prediction.
    Uses multi-view fusion for better accuracy.
    """
    fine_tuned_model_path = os.path.join(MODEL_DIR, 'fruit_classifier_finetuned.pth')

    if not os.path.exists(fine_tuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {fine_tuned_model_path}")

    try:
        predicted_type, confidence = predict_multiple_images(image_paths, fine_tuned_model_path)

        if predicted_type is None:
            raise ValueError("Prediction returned None - no valid images")

        return predicted_type, float(confidence)

    except Exception as e:
        print(f"Fine-tuned Model Execution Failed: {e}")
        raise e


def run_classification_model(feature_vector):
    """Run trained model on feature vector and return prediction (legacy approach)."""
    label_mapping = {'market': 0, 'standard': 1, 'premium': 2}
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Critical Error: Model file not found at {model_path}")

    try:
        # Load the trained model and metadata
        params, model_info = load_model(model_path)

        # Prepare input
        X = feature_vector.astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 1. Apply Scaler FIRST (StandardScaler expects 50176 features)
        scaler = model_info.get('scaler', None)
        if scaler is not None:
            try:
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == X.shape[1]:
                    X = scaler.transform(X)
                elif not hasattr(scaler, 'n_features_in_'):
                    X = scaler.transform(X)
            except Exception as e:
                pass  # Continue without scaler if it fails

        # 2. Apply Dimensionality Reduction (PCA)
        expected_dim = model_info.get('input_dim')
        if expected_dim and X.shape[1] != expected_dim:
            reducer = model_info.get('pca') or model_info.get('reducer')
            if reducer:
                X = reducer.transform(X)
            else:
                raise ValueError(f"Model expects {expected_dim} features but got {X.shape[1]}, and no PCA/Reducer was found.")

        # 3. Make prediction
        predictions, probabilities = predict(X, params)

        predicted_class = predictions[0]
        confidence = probabilities[0, predicted_class]
        predicted_type = reverse_mapping.get(predicted_class, 'unknown')

        return predicted_type, float(confidence)

    except Exception as e:
        print(f"Model Execution Failed: {e}")
        raise e
    
def process_new_fruit_folder(folder_path, db_name=None, collection_name="images", run_tests=False):
    start_time = time.time()
    if db_name is None: db_name = DB_NAME
    
    print("\nFruit Grading Pipeline")
    print("----------------------\n")
    
    print("Step 1: Collecting metadata")
    try:
        image_docs = collect_images_metadata(folder_path, db_name, collection_name)
        if not image_docs:
            print("Error: No images found.")
            return None
    except Exception as e:
        print(f"Metadata collection failed: {e}")
        return None
        
    temp_object_id = f"temp_{int(time.time())}"
    for doc in image_docs:
        doc['object_id'] = temp_object_id
        
    print(f"Processing temporary object: {temp_object_id} ({len(image_docs)} images)")
    
     # Step 2: Copy Original Images to Storage
    print("\nStep 2: Copying original images")
    try:
        # Pass None for IDs -> function will generate temp IDs
        copy_results = copy_images_to_stored(image_docs, None)
        
        success_count = 0
        if copy_results:
            for i, doc in enumerate(image_docs):
                # Unpack tuple (success, path)
                if i < len(copy_results):
                    success, new_path = copy_results[i]
                    
                    if success and new_path:
                        doc['stored_path'] = new_path
                        success_count += 1
                    else:
                        print(f"Warning: Failed to copy image {doc.get('original_filename')}")
        
        print(f"Copied {success_count} images")
        
        if success_count == 0:
            print("Error: Failed to copy any images.")
            return None

    except Exception as e:
        print(f"Copy step failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("Step 3: Preprocessing")
    try:
        dummy_ids = [f"{temp_object_id}_{i}" for i in range(len(image_docs))]
        preprocess_results = preprocess_images_batch(image_docs, dummy_ids)

        preprocessed_count = 0
        for i, doc in enumerate(image_docs):
            res = preprocess_results[i]
            if res[1]:
                doc['processed_path'] = res[1]
                preprocessed_count += 1

        print(f"  Preprocessed {preprocessed_count} images")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None

    print("Step 4: Classification")
    try:
        valid_imgs = [d for d in image_docs if 'processed_path' in d]
        image_paths = [d['processed_path'] for d in valid_imgs]

        if len(image_paths) == 0:
            print("Error: No valid images for classification.")
            return None

        # Check which model is available
        fine_tuned_model_path = os.path.join(MODEL_DIR, 'fruit_classifier_finetuned.pth')
        legacy_model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')

        if os.path.exists(fine_tuned_model_path):
            # Use fine-tuned model (preferred)
            print("  Using fine-tuned ShuffleNet model")
            predicted_type, confidence = run_classification_model_fine_tuned(image_paths)
        elif os.path.exists(legacy_model_path):
            # Fall back to legacy feature extraction + small NN
            print("  Using legacy feature extraction model")
            metadata_dict = {
                d['processed_path']: {
                    'camera_id': d['camera_id'],
                    'timestamp': d['timestamp'],
                    'object_id': temp_object_id
                } for d in valid_imgs
            }

            from preprocessing.preprocessing_from_db import set_generator
            generator, _, count = set_generator(image_paths, metadata_dict)

            if count == 0:
                print("Error: No valid images for generator.")
                return None

            fused_features = extract_and_fuse_features(generator)
            feature_vector = get_feature_vector(fused_features, temp_object_id)

            if feature_vector is None:
                print("Feature extraction returned None.")
                return None

            predicted_type, confidence = run_classification_model(feature_vector)
        else:
            raise FileNotFoundError("No trained model found. Please run build_model.py first.")

        print(f"  Result: {predicted_type} ({confidence:.2%})")

    except Exception as e:
        print(f"Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    elapsed = time.time() - start_time

    # Generate final object ID
    final_object_id = get_next_object_id(db_name, collection_name, predicted_type)
    print(f"Final Object ID: {final_object_id}")

    print("\nStep 5: Copying to Original Folder")
    try:
        obj_num = final_object_id.split('_')[1]
        folder_name = f"obj_{obj_num}"

        # 1. Define and Create the Target Folder
        object_folder = os.path.join(ORIGINAL_DATASET_PATH, predicted_type, folder_name)
        print(f"[DEBUG] Ensuring target folder exists at: {object_folder}")
        os.makedirs(object_folder, exist_ok=True)

        copied_count = 0

        for doc in image_docs:
            # FIX: Try the 'stored_path' (safe local copy) first!
            source_path = doc.get('stored_path')
            
            # If stored path is missing or broken, try the original upload path
            if not source_path or not os.path.exists(source_path):
                # print(f"[DEBUG] Stored path missing, trying original path...")
                source_path = doc.get('path')

            # Final validation
            if not source_path or not os.path.exists(source_path):
                print(f"[ERROR] Skipping image {doc.get('original_filename')}: Source file not found in stored OR temp path.")
                continue

            # 2. Prepare Destination
            camera_id = doc.get('camera_id', 0)
            angle_folder = os.path.join(object_folder, f"an_{camera_id}")
            os.makedirs(angle_folder, exist_ok=True)

            original_filename = doc.get('original_filename') or os.path.basename(source_path)
            dest_path = os.path.join(angle_folder, original_filename)

            # 3. Copy
            try:
                shutil.copy2(source_path, dest_path)
                doc['original_path'] = dest_path
                copied_count += 1
            except Exception as copy_err:
                print(f"[ERROR] Copy failed for {original_filename}: {copy_err}")

        print(f"Copied {copied_count} images to {object_folder}")
        
    except Exception as e:
        print(f"Warning: Failed to copy to original folder: {e}")

    print("\nStep 6: Saving to Database")
    try:
        client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        db = client[db_name]
        collection = db[collection_name]
        
        for doc in image_docs:
            doc['object_id'] = final_object_id
            doc['fruit_type'] = predicted_type
            doc['confidence'] = confidence
            doc['dataset_type'] = 'test'
            
        result = collection.insert_many(image_docs)
        print(f"Successfully inserted {len(result.inserted_ids)} records")
        
    except Exception as e:
        print(f"Database insertion failed: {e}")
        return None
    
    print("\nStep 7: Updating Dashboard Metadata")
    update_dashboard_metadata(predicted_type, len(image_docs))
    return {
        'object_id': final_object_id,
        'predicted_type': predicted_type,
        'confidence': confidence,
        'images_count': len(image_docs),
        'processing_time': elapsed
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process fruit folder')
    parser.add_argument('folder_path', help='Path to folder')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--db', default=None, help='Database name')
    
    args = parser.parse_args()
    
    result = process_new_fruit_folder(
        args.folder_path,
        db_name=args.db,
        run_tests=args.test
    )
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
