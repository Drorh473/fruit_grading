import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Import from our new modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Streamers.database_insertion import (
    collect_images_metadata,
    insert_images_to_db,
    copy_images_to_stored,
    update_stored_paths,
    update_preprocessed_paths,
    split_training_testing,
    get_images_by_object,
    update_fruit_type
)
from preprocessing.preprocessing_insertion import (
    validate_folder_structure,
    preprocess_images_batch,
    create_generator_for_object
)
from cnn.feature_map_insertion import (
    extract_and_fuse_features,
    get_feature_vector
)
from cnn.fully_connected_layer import load_model, predict
import numpy as np

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME', 'fruit_grading')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
PROJECT_ROOT = Path(__file__).parent.parent


def run_quick_tests():
    """Run quick test suite before processing"""
    print("\nRunning quick test suite")
    
    # Path to run_all_tests.py
    test_script = os.path.join(PROJECT_ROOT, 'run_all_tests.py')
    
    if not os.path.exists(test_script):
        print("Warning: Test script not found")
        print("Skipping tests")
        return True
    
    try:
        # Run the comprehensive test suite with quick mode
        result = subprocess.run(
            [sys.executable, test_script, '--quick'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        # Return success based on exit code
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Tests timed out")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_classification_model(feature_vector):
    """Run trained model on feature vector and return prediction."""
    # Label mapping (same as training)
    label_mapping = {
        'market': 0,
        'standard': 1,
        'premium': 2
    }
    
    # Reverse mapping for decoding predictions
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Path to saved model
    model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
    
    if not os.path.exists(model_path):
        print("Warning: Model not found, using mock prediction")
        return "market", 0.33
    
    try:
        # Load the trained model
        params, model_info = load_model(model_path)
        
        # Prepare input (add batch dimension)
        X = feature_vector.reshape(1, -1).astype(np.float32)
        
        # Make prediction
        predictions, probabilities = predict(X, params)
        
        # Get predicted class and confidence
        predicted_class = predictions[0]
        confidence = probabilities[0, predicted_class]
        
        # Decode to fruit type
        predicted_type = reverse_mapping.get(predicted_class, 'unknown')
        
        print(f"Model: {model_info['input_dim']} -> {model_info['hidden_dim']} -> {model_info['num_classes']}")
        print("Probabilities:")
        for fruit_type, class_idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            prob = probabilities[0, class_idx]
            print(f"  {fruit_type}: {prob:.1%}")
        
        return predicted_type, float(confidence)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using mock prediction")
        return "market", 0.33


def process_new_fruit_folder(folder_path, db_name=None, collection_name="images", run_tests=False):
    """Complete pipeline: validate, add to DB, preprocess, extract features, and classify.
    
    Args:
        folder_path: Path to folder containing angle directories
        db_name: Database name (defaults to env variable)
        collection_name: Collection name
        run_tests: If True, run quick test suite before processing
    
    Returns:
        Dictionary with processing results or None on failure
    """
    start_time = time.time()
    
    if db_name is None:
        db_name = DB_NAME
    
    print("\nFruit Grading Pipeline")
    print("----------------------\n")
    
    # Step 0: Run tests (optional)
    if run_tests:
        test_success = run_quick_tests()
        if not test_success:
            print("\nWarning: Some tests failed. Continue? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Pipeline aborted")
                return None
    
    # Step 1: Validate folder structure
    print("Step 1: Validating folder")
    is_valid, error_msg = validate_folder_structure(folder_path)
    if not is_valid:
        print(f"Error: {error_msg}")
        return None
    print("Folder valid")
    
    # Step 2: Collect metadata
    print("\nStep 2: Collecting metadata")
    image_data = collect_images_metadata(folder_path, db_name, collection_name)
    if not image_data:
        print("No images found")
        return None
    object_id = image_data[0]['object_id']
    print(f"Collected {len(image_data)} images for {object_id}")
    
    # Step 3: Insert into database
    print("\nStep 3: Inserting to database")
    inserted_ids = insert_images_to_db(image_data, db_name, collection_name)
    print(f"Inserted {len(inserted_ids)} records")
    
    # Step 4: Copy to stored dataset
    print("\nStep 4: Copying images")
    copy_results = copy_images_to_stored(image_data, inserted_ids)
    success_count = update_stored_paths(image_data, inserted_ids, copy_results, db_name, collection_name)
    print(f"Copied {success_count} images")
    
    # Step 5: Preprocess images
    print("\nStep 5: Preprocessing")
    preprocess_results = preprocess_images_batch(image_data, inserted_ids)
    preprocessed_count = update_preprocessed_paths(preprocess_results, db_name, collection_name)
    print(f"Preprocessed {preprocessed_count} images")
    
    # Step 6: Split into training/testing
    print("\nStep 6: Splitting data")
    split_training_testing(db_name, collection_name)
    print("Split complete")
    
    # Step 7: Get images and create generator
    print("\nStep 7: Loading images")
    images = get_images_by_object(object_id, db_name, collection_name)
    generator, count = create_generator_for_object(images)
    print(f"Generator created with {count} images")
    
    # Step 8: Extract and fuse features
    print("\nStep 8: Extracting features")
    fused_features = extract_and_fuse_features(generator)
    feature_vector = get_feature_vector(fused_features, object_id)
    
    if feature_vector is None:
        print("Failed to extract features")
        return None
    print(f"Features extracted (dim: {feature_vector.shape[0]})")
    
    # Step 9: Run classification model
    print("\nStep 9: Running classifier")
    predicted_type, confidence = run_classification_model(feature_vector)
    print("Classification complete")
    
    # Step 10: Update database with results
    print("\nStep 10: Updating database")
    updated_count = update_fruit_type(object_id, predicted_type, confidence, db_name, collection_name)
    print(f"Updated {updated_count} records")
    
    # Print final assessment
    elapsed = time.time() - start_time
    
    print("\nResults")
    print("-------")
    print(f"Object ID: {object_id}")
    print(f"Type: {predicted_type}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Images: {len(images)}")
    print(f"Time: {elapsed:.1f}s\n")
    
    return {
        'object_id': object_id,
        'predicted_type': predicted_type,
        'confidence': confidence,
        'images_count': len(images),
        'processing_time': elapsed
    }


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process fruit folder for grading')
    parser.add_argument('folder_path', help='Path to folder containing angle directories (angle_0, angle_1, etc.)')
    parser.add_argument('--test', action='store_true', help='Run quick test suite before processing')
    parser.add_argument('--db', default=None, help='Database name (default: from .env)')
    
    args = parser.parse_args()
    
    result = process_new_fruit_folder(
        args.folder_path,
        db_name=args.db,
        run_tests=args.test
    )
    
    if result is None:
        print("Pipeline failed")
        sys.exit(1)
    else:
        print("Pipeline completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()