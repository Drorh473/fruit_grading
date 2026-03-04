"""ML pipeline for fruit grading model training."""
import os
import sys
import numpy as np
import torch
import subprocess
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

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = os.getenv('MODEL_DIR')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
ORIGINAL_DATASET_PATH = os.getenv('ORIGINAL_DATASET_PATH')
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')

from utils.model_metadata import save_dashboard_metadata


def run_tests():
    """Run test suite using test_main.py."""
    print("[1/6] Running tests...")
    test_script = os.path.join(PROJECT_ROOT, r'Tests\test_main.py')

    if not os.path.exists(test_script):
        return False

    try:
        result = subprocess.run(
            [sys.executable, test_script, '--quick'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def setup_database():
    """Create and populate database from dataset."""
    if not ORIGINAL_DATASET_PATH or not os.path.exists(ORIGINAL_DATASET_PATH):
        return False

    try:
        process_dataset(ORIGINAL_DATASET_PATH, DB_NAME)
        return True
    except Exception:
        return False


def preprocess_data():
    """Preprocess images and return train/test generators."""
    try:
        train_gen, test_gen = load_dataset_with_preprocessing()
        return train_gen, test_gen
    except Exception:
        return None, None


def extract_features(train_gen, test_gen):
    """Extract features using pre-trained CNN."""
    try:
        fused_features_test = process_features(test_gen, 'testing')
        fused_features_train = process_features(train_gen, 'training')
        return fused_features_train, fused_features_test
    except Exception:
        return None, None


def prepare_image_paths_from_db():
    """Get image paths and labels from database for fine-tuning."""
    print("  Loading image paths from database...")
    from pymongo import MongoClient

    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    db = client[DB_NAME]
    collection = db['images']

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for doc in collection.find():
        img_path = doc.get('processed_path') or doc.get('stored_path') or doc.get('path')
        if not img_path or not os.path.exists(img_path):
            continue

        fruit_type = doc.get('fruit_type', 'unknown')
        label = LABEL_MAPPING.get(fruit_type, 2)
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
    """Train fine-tuned ShuffleNet model."""
    print("[4/6] Training unified classifier...")
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        train_paths, train_labels, test_paths, test_labels = prepare_image_paths_from_db()

        if len(train_paths) == 0:
            raise ValueError("No training images found in database!")

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

        print("[5/6] Saving model...")
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

        return model, full_results, len(train_paths), len(test_paths)

    except Exception:
        return None, None, 0, 0


def train_classifier(train_features, test_features,
                    hidden_dim=32, epochs=100, learning_rate=0.001, lambda_reg=0.01,
                    pca_components=100, dropout_rate=0.0, early_stopping_patience=50):
    """Train fully connected classifier on extracted features."""
    print("[5/6] Training classifier...")
    np.random.seed(42)

    try:
        label_mapping = {'market': 0, 'standard': 1, 'premium': 2}

        X_train_list, y_train_list = [], []
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

        X_test_list, y_test_list = [], []
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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        pca = None
        if pca_components is not None and pca_components > 0:
            max_components = min(pca_components, X_train.shape[0] - 1, X_train.shape[1])
            pca = PCA(n_components=max_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        input_dim = X_train.shape[1]
        num_classes = len(label_mapping)

        params, history = train(
            X_train, y_train, X_test, y_test,
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

        train_loss, train_acc = evaluate(X_train, y_train, params, num_classes, lambda_reg)
        test_loss, test_acc = evaluate(X_test, y_test, params, num_classes, lambda_reg)
        avg_confidence = compute_avg_confidence(X_test, y_test, params)

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

        return params, results

    except Exception:
        return None, None


def generate_confusion_matrix(results):
    """Generate confusion matrix from results."""
    try:
        report = generate_full_confusion_matrix_report(results, save_dir=MODEL_DIR)
        if report:
            return report['cm']
        return None
    except Exception:
        return None


def run_full_pipeline(skip_tests=False,
                     hidden_dim=8, epochs=500, learning_rate=0.01, lambda_reg=0.001,
                     pca_components=15, dropout_rate=0.2, early_stopping_patience=100):
    """Run full training pipeline: database setup, preprocessing, feature extraction, training."""
    print("=" * 50)
    print("Starting full training pipeline")
    print("=" * 50)
    if not skip_tests:
        test_success = run_tests()
        if not test_success:
            response = input("Warning: Some tests failed. Continue anyway? (y/n)").strip().lower()
            if response != 'y':
                return False

    stored_dataset_exists = os.path.exists(STORED_DATASET_PATH) if STORED_DATASET_PATH else False

    if not stored_dataset_exists:
        print("[2/6] Setting up database...")
        if not setup_database():
            return False
    else:
        print("[2/6] Database exists, skipping...")

    print("[3/6] Preprocessing images...")
    train_gen, test_gen = preprocess_data()
    if not train_gen or not test_gen:
        return False

    print("[4/6] Extracting features...")
    train_features, test_features = extract_features(train_gen, test_gen)
    if not train_features or not test_features:
        return False

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
        return False

    print("[6/6] Generating confusion matrix...")
    label_mapping = results['label_mapping']
    cm = generate_confusion_matrix(results)
    if cm is not None:
        results['confusion_matrix'] = cm

    print("  Saving dashboard metadata...")
    try:
        save_dashboard_metadata(
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
    except Exception:
        pass

    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"  Train accuracy: {results['train_accuracy']:.2%}")
    print(f"  Test accuracy:  {results['test_accuracy']:.2%}")
    print("=" * 50)
    return True


def run_fine_tuning_pipeline(skip_tests=False, epochs=50, learning_rate=0.001,
                              batch_size=8, early_stopping_patience=10,
                              unfreeze_backbone=False):
    """Run fine-tuning pipeline using ShuffleNet."""
    print("=" * 50)
    print("Starting fine-tuning pipeline")
    print("=" * 50)
    if not skip_tests:
        test_success = run_tests()
        if not test_success:
            response = input("Warning: Some tests failed. Continue anyway? (y/n)").strip().lower()
            if response != 'y':
                return False

    stored_dataset_exists = os.path.exists(STORED_DATASET_PATH) if STORED_DATASET_PATH else False
    processed_dataset_exists = os.path.exists(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH else False

    if not stored_dataset_exists:
        print("[2/6] Setting up database...")
        if not setup_database():
            return False
    else:
        print("[2/6] Database exists, skipping...")

    if not processed_dataset_exists:
        print("[3/6] Preprocessing images...")
        train_gen, test_gen = preprocess_data()
        if not train_gen or not test_gen:
            return False
    else:
        print("[3/6] Preprocessed data exists, skipping...")

    model, results, train_count, test_count = train_fine_tuned_classifier(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        unfreeze_backbone=unfreeze_backbone
    )

    if model is None:
        return False

    print("[5/6] Generating confusion matrix...")
    cm = None
    try:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        y_true = results['y_test']
        y_pred = results['predictions']
        cm = sk_confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
    except Exception:
        pass

    print("[6/6] Saving dashboard metadata...")
    try:
        train_features_dummy = {f"train_{i}": {'label': 0} for i in range(train_count)}
        test_features_dummy = {f"test_{i}": {'label': 0} for i in range(test_count)}

        save_dashboard_metadata(
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
    except Exception:
        pass

    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"  Train accuracy: {results['train_accuracy']:.2%}")
    print(f"  Test accuracy:  {results['test_accuracy']:.2%}")
    print("=" * 50)
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