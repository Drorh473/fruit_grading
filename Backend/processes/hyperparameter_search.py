"""
Hyperparameter Search Script
Systematically tests different configurations to find the best model.
"""

import os
import sys
import json
import itertools
from datetime import datetime
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from build_model import preprocess_data, extract_features, train_classifier

# Define search space - focused on most impactful parameters
SEARCH_SPACE = {
    'hidden_dim': [8, 16, 32, 64],
    'epochs': [500],
    'learning_rate': [0.005, 0.01],
    'lambda_reg': [0, 0.001],
    'pca_components': [0, 16, 32],
    'dropout_rate': [0, 0.2, 0.3],
}

def run_hyperparameter_search():
    """Run hyperparameter search and find best configuration."""

    print("="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)

    # Calculate total combinations
    total_combinations = 1
    for values in SEARCH_SPACE.values():
        total_combinations *= len(values)

    print(f"\nSearch space:")
    for param, values in SEARCH_SPACE.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations to test: {total_combinations}")

    # Step 1: Preprocess data (only once)
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING DATA (once for all experiments)")
    print("="*70)

    train_gen, test_gen = preprocess_data()
    if not train_gen or not test_gen:
        print("Failed to preprocess data!")
        return None

    # Step 2: Extract features (only once)
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING FEATURES (once for all experiments)")
    print("="*70)

    train_features, test_features = extract_features(train_gen, test_gen)
    if not train_features or not test_features:
        print("Failed to extract features!")
        return None

    # Step 3: Run experiments
    print("\n" + "="*70)
    print("STEP 3: RUNNING EXPERIMENTS")
    print("="*70)

    results = []
    best_result = None
    best_accuracy = 0

    # Generate all combinations
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())

    for i, combination in enumerate(itertools.product(*param_values)):
        config = dict(zip(param_names, combination))

        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i+1}/{total_combinations}")
        print(f"{'='*70}")
        print(f"Config: {config}")

        try:
            # Train with current config
            params, eval_results = train_classifier(
                train_features,
                test_features,
                hidden_dim=config['hidden_dim'],
                epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                lambda_reg=config['lambda_reg'],
                pca_components=config['pca_components'],
                dropout_rate=config['dropout_rate'],
                early_stopping_patience=50  # Fixed for faster search
            )

            if eval_results:
                test_accuracy = eval_results['test_accuracy']
                train_accuracy = eval_results['train_accuracy']
                avg_confidence = eval_results.get('avg_confidence', 0)

                result = {
                    'config': config,
                    'test_accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                    'avg_confidence': avg_confidence,
                    'test_loss': eval_results['test_loss'],
                    'train_loss': eval_results['train_loss'],
                }
                results.append(result)

                print(f"\n>>> Test Accuracy: {test_accuracy*100:.2f}%")
                print(f">>> Train Accuracy: {train_accuracy*100:.2f}%")
                print(f">>> Avg Confidence: {avg_confidence*100:.2f}%")

                # Track best
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_result = result
                    print(f"\n*** NEW BEST MODEL! ***")
            else:
                print(f"Training failed for this config")

        except Exception as e:
            print(f"Error with config {config}: {e}")
            continue

    # Step 4: Report results
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*70)

    if results:
        # Sort by test accuracy
        results.sort(key=lambda x: x['test_accuracy'], reverse=True)

        print(f"\nTOP 10 CONFIGURATIONS:")
        print("-"*70)

        for i, r in enumerate(results[:10]):
            print(f"\n{i+1}. Test Accuracy: {r['test_accuracy']*100:.2f}% | "
                  f"Train Accuracy: {r['train_accuracy']*100:.2f}% | "
                  f"Confidence: {r['avg_confidence']*100:.2f}%")
            print(f"   Config: {r['config']}")

        print("\n" + "="*70)
        print("BEST CONFIGURATION")
        print("="*70)

        if best_result:
            print(f"\nTest Accuracy: {best_result['test_accuracy']*100:.2f}%")
            print(f"Train Accuracy: {best_result['train_accuracy']*100:.2f}%")
            print(f"Avg Confidence: {best_result['avg_confidence']*100:.2f}%")
            print(f"\nConfiguration:")
            for param, value in best_result['config'].items():
                print(f"  {param}: {value}")

        # Save results to file
        results_file = Path(__file__).parent.parent / 'hyperparameter_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(results),
                'best_result': best_result,
                'all_results': results
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return best_result


if __name__ == "__main__":
    run_hyperparameter_search()
