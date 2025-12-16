import unittest
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Add Tests directory to path
TESTS_DIR = os.path.join(PROJECT_DIR, 'Tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Import test modules from Tests directory
try:
    from Tests.test_database_creation import TestDatabaseCreation
    from Tests.test_preprocessing_from_db import (
        TestCustomPreprocessing,
        TestProcessImage,
        TestDatabaseFunctions,
        TestPreprocessingIntegration
    )
except ImportError:
    # Fallback if Tests directory structure is different
    from test_database_creation import TestDatabaseCreation
    from test_preprocessing_from_db import (
        TestCustomPreprocessing,
        TestProcessImage,
        TestDatabaseFunctions,
        TestPreprocessingIntegration
    )

# Try to import optional test modules
try:
    from Tests.test_feature_extraction import (
        TestLoadModel,
        TestFlattenFeatures,
        TestTemporalPooling,
        TestMultiViewFusion,
        TestFeatureExtractionIntegration,
        TestFeatureExtractionEdgeCases
    )
    FEATURE_TESTS_AVAILABLE = True
except ImportError:
    FEATURE_TESTS_AVAILABLE = False
    print("Warning: Feature extraction tests not found, skipping...")

try:
    from Tests.test_network import (
        TestActivationFunctions,
        TestParameterInitialization,
        TestForwardPass,
        TestLossComputation,
        TestBackwardPass,
        TestUpdateParameters,
        TestTrainStep,
        TestPrediction,
        TestEvaluate,
        TestTraining,
        TestModelSaveLoad,
        TestTrainingEdgeCases
    )
    FC_TESTS_AVAILABLE = True
except ImportError:
    FC_TESTS_AVAILABLE = False
    print("Warning: Fully connected tests not found, skipping...")

try:
    from Tests.test_integration_build_model import (
        TestDatabaseSetup,
        TestPreprocessData,
        TestExtractFeatures,
        TestTrainClassifier,
        TestGenerateConfusionMatrix,
        TestPipelineIntegration,
        TestPipelineHyperparameters,
        TestPipelineDataValidation,
        TestPipelineOutputs
    )
    PIPELINE_TESTS_AVAILABLE = True
except ImportError:
    PIPELINE_TESTS_AVAILABLE = False
    print("Warning: Pipeline integration tests not found, skipping...")


class TestSuiteOrganizer:
    """Organize and run test suites by component"""
    
    def __init__(self):
        self.suites = {}
        
        # Always available: Database and Preprocessing
        self.suites['Database'] = [TestDatabaseCreation, TestDatabaseFunctions]
        self.suites['Preprocessing'] = [
            TestCustomPreprocessing,
            TestProcessImage,
            TestPreprocessingIntegration
        ]
        
        # Conditionally add Feature Extraction tests
        if FEATURE_TESTS_AVAILABLE:
            self.suites['Feature Extraction'] = [
                TestLoadModel,
                TestFlattenFeatures,
                TestTemporalPooling,
                TestMultiViewFusion,
                TestFeatureExtractionIntegration,
                TestFeatureExtractionEdgeCases
            ]
        
        # Conditionally add Neural Network tests
        if FC_TESTS_AVAILABLE:
            self.suites['Neural Network'] = [
                TestActivationFunctions,
                TestParameterInitialization,
                TestForwardPass,
                TestLossComputation,
                TestBackwardPass,
                TestUpdateParameters,
                TestTrainStep,
                TestPrediction,
                TestEvaluate,
                TestTraining,
                TestModelSaveLoad,
                TestTrainingEdgeCases
            ]
        
        # Conditionally add Pipeline Integration tests
        if PIPELINE_TESTS_AVAILABLE:
            self.suites['Pipeline Integration'] = [
                TestDatabaseSetup,
                TestPreprocessData,
                TestExtractFeatures,
                TestTrainClassifier,
                TestGenerateConfusionMatrix,
                TestPipelineIntegration,
                TestPipelineHyperparameters,
                TestPipelineDataValidation,
                TestPipelineOutputs
            ]
        
        self.results = {}
    
    def run_suite(self, suite_name, test_classes, verbosity=2):
        """Run a single test suite"""
        print(f"\n{'='*70}")
        print(f"RUNNING: {suite_name} Tests")
        print(f"{'='*70}\n")
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
        result = runner.run(suite)
        elapsed_time = time.time() - start_time
        
        # Store results
        self.results[suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'time': elapsed_time
        }
        
        return result.wasSuccessful()
    
    def run_all_suites(self, verbosity=2):
        """Run all test suites"""
        print("\n" + "="*70)
        print("FRUIT GRADING SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        all_passed = True
        
        for suite_name, test_classes in self.suites.items():
            suite_passed = self.run_suite(suite_name, test_classes, verbosity)
            if not suite_passed:
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0
        
        for suite_name, result in self.results.items():
            status = " PASS" if result['success'] else "✗ FAIL"
            
            print(f"\n{suite_name}:")
            print(f"  Status: {status}")
            print(f"  Tests Run: {result['tests_run']}")
            print(f"  Failures: {result['failures']}")
            print(f"  Errors: {result['errors']}")
            print(f"  Skipped: {result['skipped']}")
            print(f"  Time: {result['time']:.2f}s")
            
            total_tests += result['tests_run']
            total_failures += result['failures']
            total_errors += result['errors']
            total_skipped += result['skipped']
            total_time += result['time']
        
        print("\n" + "-"*70)
        print("OVERALL STATISTICS")
        print("-"*70)
        print(f"Total Tests: {total_tests}")
        print(f"Successes: {total_tests - total_failures - total_errors}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Total Time: {total_time:.2f}s")
        if total_tests > 0:
            print(f"Pass Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
        print("="*70 + "\n")
    
    def generate_coverage_report(self):
        """Generate component coverage report"""
        print("\n" + "="*70)
        print("COMPONENT COVERAGE ANALYSIS")
        print("="*70)
        
        components = {
            'Database Operations': [
                'create_database',
                'collect_images',
                'store_in_database',
                'split_data',
                'create_directory_structure',
                'update_data_directory'
            ],
            'Preprocessing': [
                'custom_preprocessing',
                'process_image',
                'load_dataset_split_by_camera',
                'set_generator',
                'preprocess_and_save_dataset',
                'load_dataset_with_preprocessing'
            ]
        }
        
        if FEATURE_TESTS_AVAILABLE:
            components['Feature Extraction'] = [
                'load_model',
                'extract_features_from_generator',
                'flatten_features',
                'temporal_pooling',
                'multi_view_fusion',
                'process_features'
            ]
        
        if FC_TESTS_AVAILABLE:
            components['Neural Network'] = [
                'initialize_parameters',
                'forward_pass',
                'backward_pass',
                'compute_loss',
                'update_parameters',
                'train_step',
                'predict',
                'evaluate',
                'train',
                'train_from_generator',
                'save_model',
                'load_model'
            ]
        
        if PIPELINE_TESTS_AVAILABLE:
            components['Pipeline'] = [
                'setup_database',
                'preprocess_data',
                'extract_features',
                'train_classifier',
                'generate_confusion_matrix',
                'run_full_pipeline'
            ]
        
        for component, functions in components.items():
            print(f"\n{component}:")
            print(f"  Functions covered: {len(functions)}")
            for func in functions:
                print(f"     {func}")
        
        print("\n" + "="*70 + "\n")


def run_specific_suite(suite_name):
    """Run a specific test suite by name"""
    organizer = TestSuiteOrganizer()
    
    if suite_name not in organizer.suites:
        print(f"Error: Suite '{suite_name}' not found")
        print(f"Available suites: {', '.join(organizer.suites.keys())}")
        return False
    
    success = organizer.run_suite(suite_name, organizer.suites[suite_name])
    organizer.print_summary()
    
    return success


def run_quick_test():
    """Run a quick subset of critical tests"""
    print("\n" + "="*70)
    print("QUICK TEST - Critical Components Only")
    print("="*70 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add critical tests - only the ones we're sure exist
    critical_tests = [TestDatabaseCreation, TestCustomPreprocessing]
    
    # Add optional critical tests
    if FC_TESTS_AVAILABLE:
        critical_tests.extend([
            TestActivationFunctions,
            TestParameterInitialization,
            TestForwardPass
        ])
    
    if FEATURE_TESTS_AVAILABLE:
        critical_tests.extend([
            TestFlattenFeatures,
            TestTemporalPooling,
            TestMultiViewFusion
        ])
    
    for test_class in critical_tests:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print(f"\n{'='*70}")
    print(f"Quick Test Result: {'PASS' if result.wasSuccessful() else 'FAIL'}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*70}\n")
    
    return result.wasSuccessful()


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Fruit Grading System Tests')
    parser.add_argument(
        '--suite',
        type=str,
        help='Run specific test suite (Database, Preprocessing, Feature Extraction, Neural Network, Pipeline Integration)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick critical tests only'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Show coverage report after tests'
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        success = run_quick_test()
        return 0 if success else 1
    
    # Specific suite mode
    if args.suite:
        success = run_specific_suite(args.suite)
        return 0 if success else 1
    
    # Full test mode
    organizer = TestSuiteOrganizer()
    all_passed = organizer.run_all_suites(verbosity=args.verbose)
    organizer.print_summary()
    
    if args.coverage:
        organizer.generate_coverage_report()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())