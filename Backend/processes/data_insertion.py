"""Data insertion pipeline for new fruit classification."""
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

from Streamers.database_insertion import collect_images_metadata, copy_images_to_stored
from preprocessing.preprocessing_insertion import preprocess_images_batch
from cnn.feature_map_insertion import extract_and_fuse_features, get_feature_vector
from cnn.fully_connected_layer import load_model, predict
from cnn.fine_tune_classifier import predict_multiple_images, clear_model_cache

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_DIR, 'saved_models'))
METADATA_FILE_PATH = os.path.join(PROJECT_DIR, 'saved_models', 'dashboard_metadata.json')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH', os.path.join(PROJECT_DIR))

def update_dashboard_metadata(predicted_type, image_count):
    """Update dashboard metadata with new classification results."""
    if not os.path.exists(METADATA_FILE_PATH):
        return

    try:
        with open(METADATA_FILE_PATH, 'r') as f:
            data = json.load(f)

        if 'dataset_info' in data:
            data['dataset_info']['total_objects'] = data['dataset_info'].get('total_objects', 0) + 1
            data['dataset_info']['total_images'] = data['dataset_info'].get('total_images', 0) + image_count
            data['dataset_info']['testing_count'] = data['dataset_info'].get('testing_count', 0) + 1

            if 'class_distribution' in data['dataset_info']:
                if predicted_type in data['dataset_info']['class_distribution']:
                    dist = data['dataset_info']['class_distribution'][predicted_type]
                    dist['total'] = dist.get('total', 0) + 1
                    dist['test'] = dist.get('test', 0) + 1
                else:
                    data['dataset_info']['class_distribution'][predicted_type] = {
                        'train': 0, 'test': 1, 'total': 1
                    }

        from datetime import datetime
        data['timestamp'] = datetime.now().isoformat()

        with open(METADATA_FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception:
        pass

def get_next_object_id(db_name, collection_name, predicted_type):
    """Get next available object ID by checking the original dataset folder."""
    fruit_type_dir = os.path.join(ORIGINAL_DATASET_PATH, predicted_type)
    max_num = 0

    if os.path.exists(fruit_type_dir) and os.path.isdir(fruit_type_dir):
        folders = [d for d in os.listdir(fruit_type_dir)
                   if os.path.isdir(os.path.join(fruit_type_dir, d))]

        for folder_name in folders:
            if folder_name.startswith("obj"):
                numbers = re.findall(r'\d+', folder_name)
                if numbers:
                    num = int(numbers[0])
                    max_num = max(max_num, num)

    next_num = max_num + 1
    return f"obj_{next_num}_{predicted_type}"


def run_classification_model_fine_tuned(image_paths):
    """Run fine-tuned model on image paths and return prediction."""
    fine_tuned_model_path = os.path.join(MODEL_DIR, 'fruit_classifier_finetuned.pth')

    if not os.path.exists(fine_tuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {fine_tuned_model_path}")

    predicted_type, confidence = predict_multiple_images(image_paths, fine_tuned_model_path)

    if predicted_type is None:
        raise ValueError("Prediction returned None - no valid images")

    return predicted_type, float(confidence)


def run_classification_model(feature_vector):
    """Run trained model on feature vector and return prediction (legacy approach)."""
    label_mapping = {'market': 0, 'standard': 1, 'premium': 2}
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    params, model_info = load_model(model_path)

    X = feature_vector.astype(np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    scaler = model_info.get('scaler', None)
    if scaler is not None:
        try:
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == X.shape[1]:
                X = scaler.transform(X)
            elif not hasattr(scaler, 'n_features_in_'):
                X = scaler.transform(X)
        except Exception:
            pass

    expected_dim = model_info.get('input_dim')
    if expected_dim and X.shape[1] != expected_dim:
        reducer = model_info.get('pca') or model_info.get('reducer')
        if reducer:
            X = reducer.transform(X)
        else:
            raise ValueError(f"Model expects {expected_dim} features but got {X.shape[1]}")

    predictions, probabilities = predict(X, params)

    predicted_class = predictions[0]
    confidence = probabilities[0, predicted_class]
    predicted_type = reverse_mapping.get(predicted_class, 'unknown')

    return predicted_type, float(confidence)
    
def process_new_fruit_folder(folder_path, db_name=None, collection_name="images", run_tests=False):
    """Process new fruit folder: collect metadata, preprocess, classify, and save."""
    start_time = time.time()
    if db_name is None:
        db_name = DB_NAME

    try:
        image_docs = collect_images_metadata(folder_path, db_name, collection_name)
        if not image_docs:
            return None
    except Exception:
        return None

    temp_object_id = f"temp_{int(time.time())}"
    for doc in image_docs:
        doc['object_id'] = temp_object_id

    try:
        copy_results = copy_images_to_stored(image_docs, None)

        success_count = 0
        if copy_results:
            for i, doc in enumerate(image_docs):
                if i < len(copy_results):
                    success, new_path = copy_results[i]
                    if success and new_path:
                        doc['stored_path'] = new_path
                        success_count += 1

        if success_count == 0:
            return None

    except Exception:
        return None

    try:
        dummy_ids = [f"{temp_object_id}_{i}" for i in range(len(image_docs))]
        preprocess_results = preprocess_images_batch(image_docs, dummy_ids)

        for i, doc in enumerate(image_docs):
            res = preprocess_results[i]
            if res[1]:
                doc['processed_path'] = res[1]
    except Exception:
        return None

    try:
        valid_imgs = [d for d in image_docs if 'processed_path' in d]
        image_paths = [d['processed_path'] for d in valid_imgs]

        if len(image_paths) == 0:
            return None

        fine_tuned_model_path = os.path.join(MODEL_DIR, 'fruit_classifier_finetuned.pth')
        legacy_model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')

        if os.path.exists(fine_tuned_model_path):
            predicted_type, confidence = run_classification_model_fine_tuned(image_paths)
        elif os.path.exists(legacy_model_path):
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
                return None

            fused_features = extract_and_fuse_features(generator)
            feature_vector = get_feature_vector(fused_features, temp_object_id)

            if feature_vector is None:
                return None

            predicted_type, confidence = run_classification_model(feature_vector)
        else:
            raise FileNotFoundError("No trained model found. Please run build_model.py first.")

    except Exception:
        return None

    elapsed = time.time() - start_time
    final_object_id = get_next_object_id(db_name, collection_name, predicted_type)

    try:
        obj_num = final_object_id.split('_')[1]
        folder_name = f"obj_{obj_num}"
        object_folder = os.path.join(ORIGINAL_DATASET_PATH, predicted_type, folder_name)
        os.makedirs(object_folder, exist_ok=True)

        for doc in image_docs:
            source_path = doc.get('stored_path')
            if not source_path or not os.path.exists(source_path):
                source_path = doc.get('path')

            if not source_path or not os.path.exists(source_path):
                continue

            camera_id = doc.get('camera_id', 0)
            angle_folder = os.path.join(object_folder, f"an_{camera_id}")
            os.makedirs(angle_folder, exist_ok=True)

            original_filename = doc.get('original_filename') or os.path.basename(source_path)
            dest_path = os.path.join(angle_folder, original_filename)

            try:
                shutil.copy2(source_path, dest_path)
                doc['original_path'] = dest_path
            except Exception:
                pass

    except Exception:
        pass

    try:
        client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        db = client[db_name]
        collection = db[collection_name]

        for doc in image_docs:
            doc['object_id'] = final_object_id
            doc['fruit_type'] = predicted_type
            doc['confidence'] = confidence
            doc['dataset_type'] = 'test'

        collection.insert_many(image_docs)

    except Exception:
        return None

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

    result = process_new_fruit_folder(args.folder_path, db_name=args.db, run_tests=args.test)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
