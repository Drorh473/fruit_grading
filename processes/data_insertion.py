import os
import sys
import time
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

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME', 'fruit_grading')


def run_classification_model(feature_vector):
    """Run trained model on feature vector and return prediction."""
    # TODO: Load and run your trained classifier model here
    # For now, return mock prediction
    
    # Placeholder for actual model inference
    # model = load_trained_model()
    # prediction = model.predict(feature_vector)
    # predicted_type = decode_prediction(prediction)
    # confidence = get_confidence(prediction)
    
    # Mock prediction
    predicted_type = "market"
    confidence = 0.95
    
    return predicted_type, confidence


def process_new_fruit_folder(folder_path, db_name=None, collection_name="images"):
    """Complete pipeline: validate, add to DB, preprocess, extract features, and classify."""
    start_time = time.time()
    
    if db_name is None:
        db_name = DB_NAME
    
    print(f"\n{'='*60}")
    print(f"FRUIT GRADING PIPELINE")
    print(f"{'='*60}\n")
    
    # Step 1: Validate folder structure
    print("Step 1: Validating folder structure...")
    is_valid, error_msg = validate_folder_structure(folder_path)
    if not is_valid:
        print(f"✗ Error: {error_msg}")
        return None
    print("✓ Folder structure valid")
    
    # Step 2: Collect metadata
    print("\nStep 2: Collecting image metadata...")
    image_data = collect_images_metadata(folder_path, db_name, collection_name)
    if not image_data:
        print("✗ No images found")
        return None
    object_id = image_data[0]['object_id']
    print(f"✓ Collected {len(image_data)} images for {object_id}")
    
    # Step 3: Insert into database
    print("\nStep 3: Inserting into database...")
    inserted_ids = insert_images_to_db(image_data, db_name, collection_name)
    print(f"✓ Inserted {len(inserted_ids)} records")
    
    # Step 4: Copy to stored dataset
    print("\nStep 4: Copying images to stored dataset...")
    copy_results = copy_images_to_stored(image_data, inserted_ids)
    success_count = update_stored_paths(image_data, inserted_ids, copy_results, db_name, collection_name)
    print(f"✓ Copied {success_count} images")
    
    # Step 5: Preprocess images
    print("\nStep 5: Preprocessing images...")
    preprocess_results = preprocess_images_batch(image_data, inserted_ids)
    preprocessed_count = update_preprocessed_paths(preprocess_results, db_name, collection_name)
    print(f"✓ Preprocessed {preprocessed_count} images")
    
    # Step 6: Split into training/testing
    print("\nStep 6: Splitting into training and testing sets...")
    split_training_testing(db_name, collection_name)
    print("✓ Data split complete")
    
    # Step 7: Get images and create generator
    print("\nStep 7: Loading images for feature extraction...")
    images = get_images_by_object(object_id, db_name, collection_name)
    generator, count = create_generator_for_object(images)
    print(f"✓ Created generator with {count} images")
    
    # Step 8: Extract and fuse features
    print("\nStep 8: Extracting and fusing features...")
    fused_features = extract_and_fuse_features(generator)
    feature_vector = get_feature_vector(fused_features, object_id)
    
    if feature_vector is None:
        print("✗ Could not extract feature vector")
        return None
    print(f"✓ Extracted feature vector (dim: {feature_vector.shape[0]:,})")
    
    # Step 9: Run classification model
    print("\nStep 9: Running classification model...")
    predicted_type, confidence = run_classification_model(feature_vector)
    print(f"✓ Classification complete")
    
    # Step 10: Update database with results
    print("\nStep 10: Updating database with results...")
    updated_count = update_fruit_type(object_id, predicted_type, confidence, db_name, collection_name)
    print(f"✓ Updated {updated_count} records")
    
    # Print final assessment
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"MODEL ASSESSMENT")
    print(f"{'='*60}")
    print(f"Object ID:      {object_id}")
    print(f"Predicted Type: {predicted_type}")
    print(f"Confidence:     {confidence:.2%}")
    print(f"Images:         {len(images)}")
    print(f"Processing Time: {elapsed:.2f} seconds")
    print(f"{'='*60}\n")
    
    return {
        'object_id': object_id,
        'predicted_type': predicted_type,
        'confidence': confidence,
        'images_count': len(images),
        'processing_time': elapsed
    }


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python process_fruit.py <folder_path>")
        print("  folder_path: Path to folder containing angle directories (angle_0, angle_1, etc.)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    result = process_new_fruit_folder(folder_path)
    
    if result is None:
        print("\n✗ Pipeline failed")
        sys.exit(1)
    else:
        print("\n✓ Pipeline completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()