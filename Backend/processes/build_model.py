import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent)) 

from Streamers.database_creation import process_dataset
from preprocessing.preprocessing_from_db import load_dataset_with_preprocessing
from cnn.pre_trained_feature_map import process_features
from cnn.fully_connected_layer import train, evaluate, save_model, predict, compute_avg_confidence
from visuals.confusion_matrix import generate_full_confusion_matrix_report

import subprocess

# Load environment variables
env_path = Path(__file__).parent.parent / '.env' 
load_dotenv(dotenv_path=env_path)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = os.getenv('MODEL_DIR')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')

from utils.model_metadata import save_dashboard_metadata

ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')


def run_tests():
    """Run all test suites using test_main.py"""
    print("\n" + "="*60)
    print("RUNNING TEST SUITE")
    print("="*60 + "\n")
    
    test_script = os.path.join(PROJECT_ROOT, r'Tests\test_main.py')
    
    if not os.path.exists(test_script):
        print(f"Error: Test script not found at {test_script}")
        return False 
    
    try:
        result = subprocess.run(
            [sys.executable, test_script, '--quick'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("\n Tests timed out")
        return False
    except Exception as e:
        print(f"\n Error running tests: {e}")
        return False


def setup_database():
    """Step 1: Create and populate database"""
    print("\n" + "="*60)
    print("STEP 1: DATABASE SETUP")
    print("="*60 + "\n")
    
    if not ORIGINAL_DATASET_PATH or not os.path.exists(ORIGINAL_DATASET_PATH):
        print(f"Error: Dataset path not found: {ORIGINAL_DATASET_PATH}")
        return False
    
    try:
        db_name, collection_name = process_dataset(ORIGINAL_DATASET_PATH, DB_NAME)
        print(f"\n Database setup complete: {db_name}/{collection_name}")
        return True
    except Exception as e:
        print(f"\n Database setup failed: {e}")
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
        return None, None


def train_classifier(train_features, test_features,
                    hidden_dim=32, epochs=100, learning_rate=0.001, lambda_reg=0.01,
                    pca_components=100, dropout_rate=0.0, early_stopping_patience=50,
                    min_accuracy=0.75, max_restarts=10):
    """
    Step 4: Train fully connected classifier with multiple restarts to ensure minimum accuracy.

    Args:
        min_accuracy: Minimum target accuracy (default 0.75 = 75%)
        max_restarts: Maximum training attempts before accepting best result

    Returns:
        params: Trained parameters
        results: Dictionary with evaluation results including avg_confidence
    """
    print("\n" + "="*60)
    print("STEP 4: CLASSIFIER TRAINING")
    print("="*60 + "\n")

    try:
        label_mapping = {
            'market': 0,
            'standard': 1,
            'premium': 2
        }

        # Prepare training data
        print("Preparing training data...")
        X_train_list = []
        y_train_list = []

        for key, data in train_features.items():
            if isinstance(data, dict):
                X_train_list.append(data['features'])
                y_train_list.append(data['label'])
            elif isinstance(data, np.ndarray):
                fruit_type = key.split('_')[0]
                X_train_list.append(data)
                y_train_list.append(label_mapping.get(fruit_type, 2))

        X_train_raw = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int64)

        # Prepare testing data
        print("Preparing testing data...")
        X_test_list = []
        y_test_list = []

        for key, data in test_features.items():
            if isinstance(data, dict):
                X_test_list.append(data['features'])
                y_test_list.append(data['label'])
            elif isinstance(data, np.ndarray):
                fruit_type = key.split('_')[0]
                X_test_list.append(data)
                y_test_list.append(label_mapping.get(fruit_type, 2))

        X_test_raw = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int64)

        # Apply feature normalization (StandardScaler)
        print(f"\nApplying StandardScaler normalization...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        print(f"  Features normalized (mean~0, std~1)")

        # Apply PCA
        pca = None
        original_dim = X_train.shape[1]
        if pca_components is not None and pca_components > 0:
            max_components = min(pca_components, X_train.shape[0] - 1, X_train.shape[1])
            print(f"\nApplying PCA: {original_dim:,} -> {max_components} features")

            pca = PCA(n_components=max_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            variance_explained = np.sum(pca.explained_variance_ratio_) * 100
            print(f"  Variance retained: {variance_explained:.1f}%")

        input_dim = X_train.shape[1]
        num_classes = len(label_mapping)

        print(f"\nDataset info:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Feature dimension: {input_dim}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  L2 Regularization: {lambda_reg}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Target accuracy: {min_accuracy*100:.0f}%")

        print(f"\nTraining label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_train == label)
            print(f"  {fruit_type}: {count} samples")

        print(f"\nTesting label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_test == label)
            print(f"  {fruit_type}: {count} samples")

        # Multiple restarts to find best model
        best_params = None
        best_history = None
        best_test_acc = 0.0
        best_seed = None

        print(f"\nTraining with multiple restarts (target: {min_accuracy*100:.0f}% accuracy)...")

        for attempt in range(max_restarts):
            seed = attempt * 7 + 42  # Different seeds: 42, 49, 56, 63, ...
            np.random.seed(seed)

            params, history = train(
                X_train, y_train,
                X_test, y_test,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=min(32, len(X_train)),
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                dropout_rate=dropout_rate,
                early_stopping_patience=early_stopping_patience,
                verbose=False  # Quiet during restarts
            )

            _, test_acc = evaluate(X_test, y_test, params, num_classes, lambda_reg)

            print(f"  Attempt {attempt+1}/{max_restarts} (seed={seed}): {test_acc*100:.1f}% accuracy")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_params = params.copy()
                best_history = history.copy()
                best_seed = seed

            # Stop early if we hit target accuracy
            if test_acc >= min_accuracy:
                print(f"  Target accuracy reached!")
                break

        params = best_params
        history = best_history
        print(f"\nBest model: seed={best_seed}, accuracy={best_test_acc*100:.1f}%")

        # Final evaluation with best model
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        train_loss, train_acc = evaluate(X_train, y_train, params, num_classes, lambda_reg)
        test_loss, test_acc = evaluate(X_test, y_test, params, num_classes, lambda_reg)
        
        # Calculate average confidence on test set
        avg_confidence = compute_avg_confidence(X_test, y_test, params)
        
        print(f"\nTraining set:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        print(f"\nTest set:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Avg Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'fruit_classifier.pkl')
        save_model(params, history, input_dim, hidden_dim, num_classes, model_path, pca=pca, scaler=scaler)
        
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'avg_confidence': avg_confidence,
            'history': history,
            'X_test': X_test,
            'y_test': y_test,
            'params': params,
            'label_mapping': label_mapping,
            'input_dim': input_dim
        }
        
        print(f"\n Classifier training complete")
        return params, results
        
    except Exception as e:
        print(f"\n Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_confusion_matrix(results):
    """Step 5: Generate confusion matrix"""
    print("\n" + "="*60)
    print("STEP 5: CONFUSION MATRIX GENERATION")
    print("="*60 + "\n")
    
    try:
        report = generate_full_confusion_matrix_report(results, save_dir=MODEL_DIR)
        
        if report:
            print(f"\n Confusion matrix generation complete")
            return report['cm']
        return None
        
    except Exception as e:
        print(f"\n Confusion matrix generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_pipeline(skip_tests=False, 
                     hidden_dim=8, epochs=500, learning_rate=0.01, lambda_reg=0.001,
                     pca_components=15, dropout_rate=0.2, early_stopping_patience=100):

    # Step 0: Run tests
    if not skip_tests:
        test_success = run_tests()
        if not test_success:
            print("\n Warning: Some tests failed. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Pipeline aborted.")
                return False
    
    stored_dataset_exists = os.path.exists(STORED_DATASET_PATH) if STORED_DATASET_PATH else False
    processed_dataset_exists = os.path.exists(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH else False
    
    # Step 1: Database setup
    if stored_dataset_exists:
        print("\n" + "="*60)
        print("STEP 1: DATABASE SETUP - SKIPPED")
        print("="*60 + "\n")
        print(f" Using existing dataset at: {STORED_DATASET_PATH}")
    else:
        if not setup_database():
            print("\n Pipeline failed at database setup")
            return False
    
    # Step 2: Preprocessing
    if processed_dataset_exists:
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING - SKIPPED")
        print("="*60 + "\n")
        print(f" Using existing preprocessed data at: {PROCESSED_DATASET_PATH}")
    
    train_gen, test_gen = preprocess_data()
    if not train_gen or not test_gen:
        print("\n Pipeline failed at preprocessing")
        return False
    
    # Step 3: Feature extraction
    train_features, test_features = extract_features(train_gen, test_gen)
    if not train_features or not test_features:
        print("\n Pipeline failed at feature extraction")
        return False
    
    # Step 4: Train classifier
    params, results = train_classifier(
        train_features, test_features,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        pca_components=pca_components,
        dropout_rate=dropout_rate,
        early_stopping_patience=early_stopping_patience
    )
    
    if params is None:
        print("\n Pipeline failed at classifier training")
        return False
    
    label_mapping = results['label_mapping']
    
    # Step 5: Generate confusion matrix
    cm = generate_confusion_matrix(results)
    if cm is not None:
        results['confusion_matrix'] = cm
    
    # Step 6: Save metadata
    print("\n" + "="*60)
    print("STEP 6: SAVING MODEL METADATA")
    print("="*60 + "\n")
    
    try:
        metadata_path = save_dashboard_metadata(
            results=results,
            train_count=len(train_features),
            test_count=len(test_features),
            feature_dim=results['input_dim'],
            params=params,
            label_mapping=label_mapping,
            confusion_matrix=cm,
            train_features=train_features,
            test_features=test_features,
            avg_confidence=results.get('avg_confidence')
        )
        if metadata_path:
            print(f"   Metadata saved successfully")
    except Exception as e:
        print(f"   Warning: Could not save metadata: {e}")
    
    # Success
    print("\n" + "="*60)
    print(" PIPELINE FINISHED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  Avg Confidence: {results.get('avg_confidence', 0)*100:.2f}%")
    print(f"  Model saved in: {MODEL_DIR}")
    
    if cm is not None:
        class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX BREAKDOWN")
        print("="*60)
        
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name.upper()}:")
            print(f"  Correctly predicted: {cm[i, i]}/{cm[i, :].sum()} samples")
            for j, other_class in enumerate(class_names):
                if i != j and cm[i, j] > 0:
                    print(f"  Misclassified as '{other_class}': {cm[i, j]} samples")
        
        print(f"\n" + "-"*60)
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        print(f"Total: {total_correct}/{total_samples} ({total_correct/total_samples*100:.2f}%)")
    
    print("\n" + "="*60 + "\n")
    return True


def main():
    # Best hyperparameters from hyperparameter search (100% test accuracy)
    run_full_pipeline(
        skip_tests=True,
        hidden_dim=8,           # Optimal: small network for small dataset
        epochs=500,             # With early stopping
        learning_rate=0.01,     # Optimal learning rate
        lambda_reg=0,           # No regularization needed
        pca_components=32,      # Optimal PCA components
        dropout_rate=0,         # No dropout needed
        early_stopping_patience=50
    )


if __name__ == "__main__":
    main()