import os
import sys
import unittest
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env' 
load_dotenv(dotenv_path=env_path)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = os.getenv('MODEL_DIR')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')

# Import pipeline components
from Streamers.database_creation import process_dataset
from preprocessing.preprocessing_from_db import load_dataset_with_preprocessing
from cnn.pre_trained_feature_map import process_features
from cnn.fully_connected_layer import train, evaluate, save_model
from visuals.confusion_matrix import generate_full_confusion_matrix_report
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
        return fused_features_train, fused_features_test 
    except Exception as e:
        print(f"\n Feature extraction failed: {e}")
        return False

def train_classifier(train_features, test_features, 
                    hidden_dim=256, epochs=100, learning_rate=0.001):
    """
    Step 4: Train fully connected classifier
    
    Args:
        train_features: Dictionary with fused feature vectors from training set
                       Format: {key: {'features': np.array, 'label': int, 'fruit_type': str}}
        test_features: Dictionary with fused feature vectors from test set
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        params: Trained parameters
        results: Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("STEP 4: CLASSIFIER TRAINING")
    print("="*60 + "\n")
    
    try:
        # Label mapping
        label_mapping = {
            'market': 0,
            'standard': 1,
            'premium': 2
        }
        
        # Prepare training data from fused features
        print("Preparing training data from fused features...")
        X_train_list = []
        y_train_list = []
        
        for key, data in train_features.items():
            X_train_list.append(data['features'])
            y_train_list.append(data['label'])
        
        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int64)
        
        # Prepare testing data from fused features
        print("Preparing testing data from fused features...")
        X_test_list = []
        y_test_list = []
        
        for key, data in test_features.items():
            X_test_list.append(data['features'])
            y_test_list.append(data['label'])
        
        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int64)
        
        # Get dimensions
        input_dim = X_train.shape[1]
        num_classes = len(label_mapping)
        
        print(f"\nDataset info:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Feature dimension: {input_dim:,}")
        print(f"  Number of classes: {num_classes}")
        
        # Check label distribution
        print(f"\nTraining label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_train == label)
            print(f"  {fruit_type}: {count} samples")
        
        print(f"\nTesting label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_test == label)
            print(f"  {fruit_type}: {count} samples")
        
        print(f"\nTraining for {epochs} epochs...")
        params, history = train(
            X_train, y_train,
            X_test, y_test,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=32,
            learning_rate=learning_rate,
            verbose=True
        )
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        train_loss, train_acc = evaluate(X_train, y_train, params, num_classes)
        test_loss, test_acc = evaluate(X_test, y_test, params, num_classes)
        
        print(f"\nTraining set:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        print(f"\nTest set:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
        save_model(params, history, input_dim, hidden_dim, num_classes, model_path)
        
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'history': history,
            'X_test': X_test,
            'y_test': y_test,
            'params': params,
            'label_mapping': label_mapping
        }
        
        print(f"\n✓ Classifier training complete")
        return params, results
        
    except Exception as e:
        print(f"\n✗ Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def train_classifier(train_features, test_features, 
                    hidden_dim=32, epochs=100, learning_rate=0.001):
    """
    Step 4: Train fully connected classifier
    
    Args:
        train_features: Dictionary with fused feature vectors from training set
        test_features: Dictionary with fused feature vectors from test set
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        params: Trained parameters
        results: Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("STEP 4: CLASSIFIER TRAINING")
    print("="*60 + "\n")
    
    try:
        # Label mapping
        label_mapping = {
            'market': 0,
            'standard': 1,
            'premium': 2
        }
        
        if train_features:
            first_key = list(train_features.keys())[0]
            first_value = train_features[first_key]        
        # Prepare training data from fused features
        print("\nPreparing training data from fused features...")
        X_train_list = []
        y_train_list = []
        
        for key, data in train_features.items():
            # Handle both old structure (numpy array) and new structure (dict)
            if isinstance(data, dict):
                X_train_list.append(data['features'])
                y_train_list.append(data['label'])
            elif isinstance(data, np.ndarray):
                # Fallback: extract label from key
                print(f"WARNING: Found numpy array instead of dict for key: {key}")
                fruit_type = key.split('_')[0]
                X_train_list.append(data)
                y_train_list.append(label_mapping.get(fruit_type, 2))
            else:
                print(f"ERROR: Unexpected data type for key {key}: {type(data)}")
                continue
        
        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int64)
        
        # Prepare testing data from fused features
        print("Preparing testing data from fused features...")
        X_test_list = []
        y_test_list = []
        
        for key, data in test_features.items():
            # Handle both old structure (numpy array) and new structure (dict)
            if isinstance(data, dict):
                X_test_list.append(data['features'])
                y_test_list.append(data['label'])
            elif isinstance(data, np.ndarray):
                # Fallback: extract label from key
                fruit_type = key.split('_')[0]
                X_test_list.append(data)
                y_test_list.append(label_mapping.get(fruit_type, 2))
            else:
                print(f"ERROR: Unexpected data type for key {key}: {type(data)}")
                continue
        
        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int64)
        
        # Get dimensions
        input_dim = X_train.shape[1]
        num_classes = len(label_mapping)
        
        print(f"\nDataset info:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Feature dimension: {input_dim:,}")
        print(f"  Number of classes: {num_classes}")
        
        # Check label distribution
        print(f"\nTraining label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_train == label)
            print(f"  {fruit_type}: {count} samples")
        
        print(f"\nTesting label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_test == label)
            print(f"  {fruit_type}: {count} samples")
        
        print(f"\nTraining for {epochs} epochs...")
        params, history = train(
            X_train, y_train,
            X_test, y_test,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=32,
            learning_rate=learning_rate,
            verbose=True
        )
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        train_loss, train_acc = evaluate(X_train, y_train, params, num_classes)
        test_loss, test_acc = evaluate(X_test, y_test, params, num_classes)
        
        print(f"\nTraining set:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        print(f"\nTest set:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
        save_model(params, history, input_dim, hidden_dim, num_classes, model_path)
        
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'history': history,
            'X_test': X_test,
            'y_test': y_test,
            'params': params,
            'label_mapping': label_mapping
        }
        
        print(f"\n✓ Classifier training complete")
        return params, results
        
    except Exception as e:
        print(f"\n✗ Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def generate_confusion_matrix(results):
    """
    Step 5 (Optional): Generate confusion matrix from training results
    
    Args:
        results: Results dictionary from train_classifier
    
    Returns:
        report: Confusion matrix report dictionary
    """
    print("\n" + "="*60)
    print("STEP 5: CONFUSION MATRIX GENERATION")
    print("="*60 + "\n")
    
    try:     
        # Generate complete report (does everything!)
        report = generate_full_confusion_matrix_report(results, save_dir=MODEL_DIR)
        
        if report:
            print(f"\n✓ Confusion matrix generation complete")
            return report['cm']
        else:
            return None
        
    except Exception as e:
        print(f"\n✗ Confusion matrix generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_pipeline(skip_tests=True, 
                     hidden_dim=256, epochs=100, learning_rate=0.001):
    # Step 0: Run tests (optional)
    if not skip_tests:
        test_success = run_tests()
        if not test_success:
            print("\n⚠ Warning: Some tests failed. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Pipeline aborted.")
                return False
    
    # Check if dataset folder exists (from .env)
    stored_dataset_exists = os.path.exists(STORED_DATASET_PATH) if STORED_DATASET_PATH else False
    processed_dataset_exists = os.path.exists(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH else False
    
    # Step 1: Database setup (skip if stored dataset exists)
    if stored_dataset_exists:
        print("\n" + "="*60)
        print("STEP 1: DATABASE SETUP - SKIPPED (Dataset folder exists)")
        print("="*60 + "\n")
        print(f"✓ Using existing dataset at: {STORED_DATASET_PATH}")
    else:
        if not setup_database():
            print("\n✗ Pipeline failed at database setup")
            return False
    
    # Step 2: Preprocessing (skip if processed folder exists)
    if processed_dataset_exists:
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING - SKIPPED (Processed folder exists)")
        print("="*60 + "\n")
        print(f"✓ Using existing preprocessed data at: {PROCESSED_DATASET_PATH}")
        
        # Still need to load the data
        print("Loading preprocessed data...")
        train_gen, test_gen = preprocess_data()
        if not train_gen or not test_gen:
            print("\n✗ Pipeline failed at loading preprocessed data")
            return False
    else:
        train_gen, test_gen = preprocess_data()
        if not train_gen or not test_gen:
            print("\n✗ Pipeline failed at preprocessing")
            return False
    
    # Step 3: Feature extraction
    train_features, test_features = extract_features(train_gen, test_gen)
    if not train_features or not test_features:
        print("\n✗ Pipeline failed at feature extraction")
        return False
    
    # Step 4: Train classifier
    params, results = train_classifier(
        train_features, test_features,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate
    )
    if params is None:
        print("\n✗ Pipeline failed at classifier training")
        return False
    label_mapping = results['label_mapping']    # Step 5: Generate confusion matrix
    cm = generate_confusion_matrix(results)
    if cm is not None:
        results['confusion_matrix'] = cm
    
    # Success
    print("\n" + "="*60)
    print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Classes: {len(label_mapping)}")
    print(f"  Training Accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  Model saved in: {MODEL_DIR}")
    
    # Display confusion matrix breakdown if available
    if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
        cm = results['confusion_matrix']
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX BREAKDOWN (Per Class)")
        print("="*60)
        
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name.upper()}:")
            print(f"  Correctly predicted: {cm[i, i]}/{cm[i, :].sum()} samples")
            
            # Show misclassifications if any
            for j, other_class in enumerate(class_names):
                if i != j and cm[i, j] > 0:
                    print(f"  Misclassified as '{other_class}': {cm[i, j]} samples")
        
        # Overall summary
        print(f"\n" + "-"*60)
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        print(f"Total: {total_correct}/{total_samples} correctly classified ({total_correct/total_samples*100:.2f}%)")
    
    print("\n" + "="*60 + "\n")
    
    return True
def main():
        run_full_pipeline(
        skip_tests=True,       
        hidden_dim=256,       
        epochs=100,           
        learning_rate=0.001    
    )
if __name__ == "__main__":
    main()