"""
Hyperparameter Search for Fine-Tuned ShuffleNet

Focused search with 20 best parameter combinations based on:
- Fine-tuning best practices (lower learning rates work better)
- Small dataset considerations (smaller batch sizes, more regularization)
- Previous experience with similar tasks
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from itertools import product
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cnn.fine_tune_classifier import (
    train_fine_tuned_model,
    LABEL_MAPPING,
    clear_model_cache
)

# Load environment
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
DB_NAME = os.getenv('DB_NAME', 'fruit_grading')


def prepare_image_paths_from_filesystem():
    """
    Get image paths and labels directly from the original dataset folder.
    Uses object-based train/test split (first N objects for training, rest for testing).
    """
    orig_path = os.getenv('ORIGINAL_DATASET_PATH')
    if not orig_path or not os.path.exists(orig_path):
        raise ValueError(f"ORIGINAL_DATASET_PATH not found: {orig_path}")

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    # Split ratio (use ~60% of objects for training)
    train_ratio = 0.6

    for category in ['market', 'standard', 'premium']:
        category_path = os.path.join(orig_path, category)
        if not os.path.exists(category_path):
            continue

        label = LABEL_MAPPING.get(category)
        if label is None:
            continue

        # Get all object folders (sorted for reproducibility)
        obj_folders = sorted([f for f in os.listdir(category_path)
                              if os.path.isdir(os.path.join(category_path, f)) and f.startswith('obj')])

        # Split objects into train/test
        num_train = max(1, int(len(obj_folders) * train_ratio))
        train_objects = obj_folders[:num_train]
        test_objects = obj_folders[num_train:]

        for obj_folder in train_objects:
            obj_path = os.path.join(category_path, obj_folder)
            for root, dirs, files in os.walk(obj_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        train_paths.append(os.path.join(root, file))
                        train_labels.append(label)

        for obj_folder in test_objects:
            obj_path = os.path.join(category_path, obj_folder)
            for root, dirs, files in os.walk(obj_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_paths.append(os.path.join(root, file))
                        test_labels.append(label)

    return train_paths, train_labels, test_paths, test_labels


def run_hyperparameter_search():
    """
    Run focused hyperparameter search with 20 best configurations.
    """
    print("=" * 60)
    print("FINE-TUNING HYPERPARAMETER SEARCH")
    print("=" * 60)

    # Load data once
    print("\nLoading data from processed dataset folder...")
    train_paths, train_labels, test_paths, test_labels = prepare_image_paths_from_filesystem()
    print(f"  Training images: {len(train_paths)}")
    print(f"  Testing images: {len(test_paths)}")

    # 20 carefully selected configurations based on fine-tuning best practices
    # Lower learning rates (0.0001-0.001) work better for fine-tuning
    # Smaller batches (4-8) work better with small datasets
    # Frozen backbone usually better with very small datasets

    configs = [
        # Config 1-5: Very low learning rates, frozen backbone (safest)
        {'lr': 0.0001, 'batch_size': 4, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.0001, 'batch_size': 8, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.0005, 'batch_size': 4, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.0005, 'batch_size': 8, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.001, 'batch_size': 4, 'epochs': 50, 'patience': 10, 'unfreeze': False},

        # Config 6-10: Standard learning rates, frozen backbone
        {'lr': 0.001, 'batch_size': 8, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.001, 'batch_size': 4, 'epochs': 100, 'patience': 15, 'unfreeze': False},
        {'lr': 0.001, 'batch_size': 8, 'epochs': 100, 'patience': 15, 'unfreeze': False},
        {'lr': 0.002, 'batch_size': 4, 'epochs': 50, 'patience': 10, 'unfreeze': False},
        {'lr': 0.002, 'batch_size': 8, 'epochs': 50, 'patience': 10, 'unfreeze': False},

        # Config 11-15: Unfrozen backbone with very low LR (careful fine-tuning)
        {'lr': 0.00005, 'batch_size': 4, 'epochs': 50, 'patience': 15, 'unfreeze': True},
        {'lr': 0.00005, 'batch_size': 8, 'epochs': 50, 'patience': 15, 'unfreeze': True},
        {'lr': 0.0001, 'batch_size': 4, 'epochs': 50, 'patience': 15, 'unfreeze': True},
        {'lr': 0.0001, 'batch_size': 8, 'epochs': 50, 'patience': 15, 'unfreeze': True},
        {'lr': 0.0002, 'batch_size': 4, 'epochs': 50, 'patience': 15, 'unfreeze': True},

        # Config 16-20: Longer training, various settings
        {'lr': 0.0005, 'batch_size': 4, 'epochs': 100, 'patience': 20, 'unfreeze': False},
        {'lr': 0.0005, 'batch_size': 8, 'epochs': 100, 'patience': 20, 'unfreeze': False},
        {'lr': 0.0001, 'batch_size': 4, 'epochs': 100, 'patience': 20, 'unfreeze': True},
        {'lr': 0.0001, 'batch_size': 8, 'epochs': 100, 'patience': 20, 'unfreeze': True},
        {'lr': 0.001, 'batch_size': 4, 'epochs': 75, 'patience': 15, 'unfreeze': False},
    ]

    print(f"\nRunning {len(configs)} configurations...\n")

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(configs)}")
        print(f"  LR: {config['lr']}, Batch: {config['batch_size']}, "
              f"Epochs: {config['epochs']}, Patience: {config['patience']}, "
              f"Unfreeze: {config['unfreeze']}")
        print("=" * 60)

        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        try:
            # Clear model cache between runs
            clear_model_cache()

            model, history, run_results = train_fine_tuned_model(
                train_paths=train_paths,
                train_labels=train_labels,
                test_paths=test_paths,
                test_labels=test_labels,
                epochs=config['epochs'],
                learning_rate=config['lr'],
                batch_size=config['batch_size'],
                early_stopping_patience=config['patience'],
                unfreeze_backbone=config['unfreeze']
            )

            result = {
                'config': config,
                'train_acc': run_results['train_accuracy'],
                'test_acc': run_results['test_accuracy'],
                'avg_conf': run_results['avg_confidence'],
                'final_epoch': run_results['final_epoch']
            }
            results.append(result)

            print(f"\n  Result: Train={result['train_acc']*100:.1f}%, "
                  f"Test={result['test_acc']*100:.1f}%, "
                  f"Conf={result['avg_conf']*100:.1f}%")

        except Exception as e:
            print(f"\n  FAILED: {e}")
            results.append({
                'config': config,
                'train_acc': 0,
                'test_acc': 0,
                'avg_conf': 0,
                'final_epoch': 0,
                'error': str(e)
            })

    # Sort by test accuracy (primary), then train accuracy (secondary)
    results.sort(key=lambda x: (x['test_acc'], x['train_acc']), reverse=True)

    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS (sorted by test accuracy)")
    print("=" * 80)
    print(f"{'Rank':<5} {'LR':<10} {'Batch':<6} {'Epochs':<7} {'Pat':<5} {'Unfreeze':<9} "
          f"{'Train%':<8} {'Test%':<8} {'Conf%':<8} {'Epoch':<6}")
    print("-" * 80)

    for rank, r in enumerate(results[:20], 1):
        cfg = r['config']
        print(f"{rank:<5} {cfg['lr']:<10.5f} {cfg['batch_size']:<6} {cfg['epochs']:<7} "
              f"{cfg['patience']:<5} {str(cfg['unfreeze']):<9} "
              f"{r['train_acc']*100:<8.1f} {r['test_acc']*100:<8.1f} "
              f"{r['avg_conf']*100:<8.1f} {r['final_epoch']:<6}")

    # Best configuration
    best = results[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"  Learning Rate: {best['config']['lr']}")
    print(f"  Batch Size: {best['config']['batch_size']}")
    print(f"  Epochs: {best['config']['epochs']}")
    print(f"  Early Stopping Patience: {best['config']['patience']}")
    print(f"  Unfreeze Backbone: {best['config']['unfreeze']}")
    print(f"\n  Train Accuracy: {best['train_acc']*100:.2f}%")
    print(f"  Test Accuracy: {best['test_acc']*100:.2f}%")
    print(f"  Avg Confidence: {best['avg_conf']*100:.2f}%")
    print(f"  Final Epoch: {best['final_epoch']}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_hyperparameter_search()
