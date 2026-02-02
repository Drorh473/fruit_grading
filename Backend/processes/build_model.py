import os
import sys
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from Streamers.database_creation import process_dataset
from preprocessing.preprocessing_from_db import load_dataset_with_preprocessing
from cnn.pre_trained_feature_map import process_features
from cnn.fully_connected_layer import train, evaluate, save_model, predict, compute_avg_confidence
from cnn.fine_tune_classifier import (
    train_fine_tuned_model,
    save_fine_tuned_model,
    LABEL_MAPPING,
    REVERSE_MAPPING
)
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


def prepare_image_paths_from_db():
    """
    Get image paths and labels from the database for fine-tuning.
    Returns train and test image paths with their labels.
    """
    from pymongo import MongoClient

    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    db = client[DB_NAME]
    collection = db['images']

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    # Get all images grouped by dataset_type
    for doc in collection.find():
        # Get the processed path (preferred) or stored path
        img_path = doc.get('processed_path') or doc.get('stored_path') or doc.get('path')
        if not img_path or not os.path.exists(img_path):
            continue

        fruit_type = doc.get('fruit_type', 'unknown')
        label = LABEL_MAPPING.get(fruit_type, 2)  # Default to 'premium' (2) for unknown

        dataset_type = doc.get('dataset_type', 'train')

        if dataset_type == 'train':
            train_paths.append(img_path)
            train_labels.append(label)
        else:
            test_paths.append(img_path)
            test_labels.append(label)

    client.close()

    return train_paths, train_labels, test_paths, test_labels


def train_fine_tuned_classifier(epochs=50, learning_rate=0.001, batch_size=8,
                                 early_stopping_patience=10, unfreeze_backbone=False):
    """
    Step 4 (Fine-Tuning): Train a fine-tuned ShuffleNet model.

    Instead of extracting features and training a small NN,
    this fine-tunes the last layers of ShuffleNet directly on the fruit images.
    """
    print("\n" + "="*60)
    print("STEP 4: FINE-TUNING SHUFFLENET")
    print("="*60 + "\n")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        # Get image paths from database
        print("Loading image paths from database...")
        train_paths, train_labels, test_paths, test_labels = prepare_image_paths_from_db()

        print(f"  Training images: {len(train_paths)}")
        print(f"  Testing images: {len(test_paths)}")

        if len(train_paths) == 0:
            raise ValueError("No training images found in database!")

        # Show class distribution
        print(f"\nTraining label distribution:")
        for fruit_type, label in sorted(LABEL_MAPPING.items(), key=lambda x: x[1]):
            count = sum(1 for l in train_labels if l == label)
            print(f"  {fruit_type}: {count} images")

        print(f"\nTesting label distribution:")
        for fruit_type, label in sorted(LABEL_MAPPING.items(), key=lambda x: x[1]):
            count = sum(1 for l in test_labels if l == label)
            print(f"  {fruit_type}: {count} images")

        # Train the fine-tuned model
        print(f"\nFine-tuning configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Unfreeze backbone: {unfreeze_backbone}")

        model, history, results = train_fine_tuned_model(
            train_paths=train_paths,
            train_labels=train_labels,
            test_paths=test_paths,
            test_labels=test_labels,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            unfreeze_backbone=unfreeze_backbone
        )

        # Save the model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'fruit_classifier_finetuned.pth')

        metadata = {
            'train_accuracy': results['train_accuracy'],
            'test_accuracy': results['test_accuracy'],
            'avg_confidence': results['avg_confidence'],
            'final_epoch': results['final_epoch'],
            'train_count': len(train_paths),
            'test_count': len(test_paths)
        }

        save_fine_tuned_model(model, model_path, metadata)

        # Prepare results dict for confusion matrix generation
        full_results = {
            'train_loss': results['train_loss'],
            'train_accuracy': results['train_accuracy'],
            'test_loss': results['test_loss'],
            'test_accuracy': results['test_accuracy'],
            'avg_confidence': results['avg_confidence'],
            'history': history,
            'y_test': np.array(test_labels),
            'predictions': results['predictions'],
            'label_mapping': LABEL_MAPPING,
            'input_dim': 'fine-tuned',
            'model_path': model_path
        }

        print(f"\n Fine-tuning complete")
        print(f"  Train Accuracy: {results['train_accuracy']*100:.2f}%")
        print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"  Avg Confidence: {results['avg_confidence']*100:.2f}%")

        return model, full_results, len(train_paths), len(test_paths)

    except Exception as e:
        print(f"\n Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0


def train_classifier(train_features, test_features,
                    hidden_dim=32, epochs=100, learning_rate=0.001, lambda_reg=0.01,
                    pca_components=100, dropout_rate=0.0, early_stopping_patience=50):
    """
    Step 4: Train fully connected classifier

    Returns:
        params: Trained parameters
        results: Dictionary with evaluation results including avg_confidence
    """
    print("\n" + "="*60)
    print("STEP 4: CLASSIFIER TRAINING")
    print("="*60 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

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

        X_train = np.array(X_train_list, dtype=np.float32)
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

        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int64)

        # Apply feature normalization (StandardScaler)
        print(f"\nApplying StandardScaler normalization...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
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

        print(f"\nTraining label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_train == label)
            print(f"  {fruit_type}: {count} samples")

        print(f"\nTesting label distribution:")
        for fruit_type, label in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y_test == label)
            print(f"  {fruit_type}: {count} samples")

        print(f"\nTraining for up to {epochs} epochs...")
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
            verbose=True
        )

        # Final evaluation
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


def run_fine_tuning_pipeline(skip_tests=False, epochs=50, learning_rate=0.001,
                              batch_size=8, early_stopping_patience=10,
                              unfreeze_backbone=False):
    """
    Run the fine-tuning pipeline instead of feature extraction + small NN.

    This approach fine-tunes ShuffleNet's last layers directly on fruit images,
    which typically achieves better and more consistent accuracy with small datasets.
    """
    print("\n" + "="*60)
    print("FRUIT GRADING MODEL - FINE-TUNING PIPELINE")
    print("="*60 + "\n")

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

    # Step 1: Database setup (still needed to have images in DB)
    if stored_dataset_exists:
        print("\n" + "="*60)
        print("STEP 1: DATABASE SETUP - SKIPPED")
        print("="*60 + "\n")
        print(f" Using existing dataset at: {STORED_DATASET_PATH}")
    else:
        if not setup_database():
            print("\n Pipeline failed at database setup")
            return False

    # Step 2: Preprocessing (needed to create processed images)
    processed_dataset_exists = os.path.exists(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH else False

    if processed_dataset_exists:
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING - USING EXISTING")
        print("="*60 + "\n")
        print(f" Using existing preprocessed data at: {PROCESSED_DATASET_PATH}")
    else:
        train_gen, test_gen = preprocess_data()
        if not train_gen or not test_gen:
            print("\n Pipeline failed at preprocessing")
            return False

    # Step 3: SKIPPED - No separate feature extraction needed for fine-tuning
    print("\n" + "="*60)
    print("STEP 3: FEATURE EXTRACTION - SKIPPED (Fine-tuning mode)")
    print("="*60 + "\n")
    print(" Fine-tuning extracts features internally in ShuffleNet")

    # Step 4: Fine-tune the model
    model, results, train_count, test_count = train_fine_tuned_classifier(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        unfreeze_backbone=unfreeze_backbone
    )

    if model is None:
        print("\n Pipeline failed at fine-tuning")
        return False

    # Step 5: Generate confusion matrix
    print("\n" + "="*60)
    print("STEP 5: CONFUSION MATRIX GENERATION")
    print("="*60 + "\n")

    try:
        # Build confusion matrix from predictions
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        y_true = results['y_test']
        y_pred = results['predictions']
        cm = sk_confusion_matrix(y_true, y_pred)

        results['confusion_matrix'] = cm

        print(f" Confusion matrix generated")

    except Exception as e:
        print(f" Warning: Could not generate confusion matrix: {e}")
        cm = None

    # Step 6: Save metadata
    print("\n" + "="*60)
    print("STEP 6: SAVING MODEL METADATA")
    print("="*60 + "\n")

    try:
        # Prepare data for metadata saving
        from utils.model_metadata import save_dashboard_metadata

        # Create dummy features dicts for metadata (just need counts)
        train_features_dummy = {f"train_{i}": {'label': 0} for i in range(train_count)}
        test_features_dummy = {f"test_{i}": {'label': 0} for i in range(test_count)}

        metadata_path = save_dashboard_metadata(
            results=results,
            train_count=train_count,
            test_count=test_count,
            feature_dim='fine-tuned (1024)',
            params=None,
            label_mapping=LABEL_MAPPING,
            confusion_matrix=cm,
            train_features=train_features_dummy,
            test_features=test_features_dummy,
            avg_confidence=results.get('avg_confidence'),
            model_type='FineTunedShuffleNet'
        )
        if metadata_path:
            print(f"   Metadata saved successfully")
    except Exception as e:
        print(f"   Warning: Could not save metadata: {e}")
        import traceback
        traceback.print_exc()

    # Success
    print("\n" + "="*60)
    print(" FINE-TUNING PIPELINE FINISHED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  Avg Confidence: {results.get('avg_confidence', 0)*100:.2f}%")
    print(f"  Model saved in: {MODEL_DIR}")

    if cm is not None:
        class_names = [name for name, idx in sorted(LABEL_MAPPING.items(), key=lambda x: x[1])]

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
    run_fine_tuning_pipeline(
        skip_tests=True,
        epochs=50,
        learning_rate=0.05,
        batch_size=256,
        early_stopping_patience=10,
        unfreeze_backbone=False
    )


if __name__ == "__main__":
    main()