import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Import pipeline components
from Streamers.database_creation import process_dataset
from preprocessing.preprocessing_from_db import load_dataset_with_preprocessing
from cnn.pre_trained_feature_map import process_features

# Import test modules
from Tests.test_database_creation import TestDatabaseCreation
from Tests.test_preprocessing_from_db import (
    TestCustomPreprocessing,
    TestProcessImage,
    TestDatabaseFunctions,
    TestPreprocessingIntegration
)

# Get configuration
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')


def run_tests():
    """Run all test suites"""
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessImage))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0,buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60 + "\n")
    return result.wasSuccessful()


def setup_database():
    """Step 1: Create and populate database"""
    print("\n" + "="*60)
    print("STEP 1: DATABASE SETUP")
    print("="*60 + "\n")
    
    if not ORIGINAL_DATASET_PATH or not os.path.exists(ORIGINAL_DATASET_PATH):
        print(f"Error: Dataset path not found: {ORIGINAL_DATASET_PATH}")
        print("Please set ORIGINAL_DATASET_PATH in .env file")
        return False
    
    try:
        db_name, collection_name = process_dataset(ORIGINAL_DATASET_PATH, DB_NAME)
        print(f"\n✓ Database setup complete: {db_name}/{collection_name}")
        return True
    except Exception as e:
        print(f"\n✗ Database setup failed: {e}")
        return False


def preprocess_data():
    """Step 2: Preprocess images"""
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60 + "\n")
    
    try:
        train_gen, test_gen = load_dataset_with_preprocessing()
        print(f"\n Preprocessing complete")
        print(f"  Training batches: {train_gen.num_batches if train_gen else 0}")
        print(f"  Testing batches: {test_gen.num_batches if test_gen else 0}")
        return train_gen, test_gen
    except Exception as e:
        print(f"\n Preprocessing failed: {e}")
        return None, None


def extract_features(train_gen, test_gen):
    """Step 3: Extract features using pre-trained CNN"""
    print("\n" + "="*60)
    print("STEP 3: FEATURE EXTRACTION")
    print("="*60 + "\n")
    
    try:
        fused_features_test = process_features(test_gen, 'testing')
        fused_features_train = process_features(train_gen, 'training')
        print(f"\n Feature extraction complete")
        print(f"  Total fused feature vectors: {len(fused_features_test)+len(fused_features_train)}")
        return True
    except Exception as e:
        print(f"\n Feature extraction failed: {e}")
        return False


def run_full_pipeline(skip_tests=True):
    # Step 0: Run tests (optional)
    if not skip_tests:
        test_success = run_tests()
        if not test_success:
            print("\ns Warning: Some tests failed. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Pipeline aborted.")
                return False
    
    # Step 1: Database setup
    if not setup_database():
        print("\n✗ Pipeline failed at database setup")
        return False
    
    # Step 2: Preprocessing
    train_gen , test_gen = preprocess_data()
    if not train_gen and not test_gen:
        print("\n Pipeline failed at preprocessing")
        return False
    
    # Step 3: Feature extraction
    if not extract_features(train_gen , test_gen):
        print("\n Pipeline failed at feature extraction")
        return False
    
    # Success
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    return True


def main():
    run_full_pipeline()
if __name__ == "__main__":
    main()