import unittest
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import Mock, patch


# Add project to path
PROJECT_DIR = '/mnt/project'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from processes.build_model import (
    setup_database,
    preprocess_data,
    extract_features,
    train_classifier,
    generate_confusion_matrix,
    run_full_pipeline
)

# Load environment
env_path = Path('.') / '.env.test'
load_dotenv(dotenv_path=env_path)


class TestDatabaseSetup(unittest.TestCase):
    """Test cases for database setup step"""
    
    @patch('processes.build_model.process_dataset')
    @patch('processes.build_model.ORIGINAL_DATASET_PATH', '/test/dataset/path')
    @patch('os.path.exists')
    def test_setup_database_success(self, mock_exists, mock_process):
        """Test successful database setup"""
        mock_exists.return_value = True
        mock_process.return_value = ('test_db', 'test_collection')
        
        result = setup_database()
        
        self.assertTrue(result)
        mock_process.assert_called_once()
    
    @patch('processes.build_model.ORIGINAL_DATASET_PATH', None)
    def test_setup_database_no_path(self):
        """Test database setup with missing dataset path"""
        result = setup_database()
        
        self.assertFalse(result)
    
    @patch('processes.build_model.process_dataset')
    @patch('processes.build_model.ORIGINAL_DATASET_PATH', '/test/dataset/path')
    @patch('os.path.exists')
    def test_setup_database_exception(self, mock_exists, mock_process):
        """Test database setup handles exceptions"""
        mock_exists.return_value = True
        mock_process.side_effect = Exception("Database error")
        
        result = setup_database()
        
        self.assertFalse(result)


class TestPreprocessData(unittest.TestCase):
    """Test cases for data preprocessing step"""
    
    @patch('processes.build_model.load_dataset_with_preprocessing')
    def test_preprocess_data_success(self, mock_load):
        """Test successful data preprocessing"""
        # Create mock generators
        mock_train_gen = Mock()
        mock_train_gen.num_batches = 10
        mock_test_gen = Mock()
        mock_test_gen.num_batches = 3
        
        mock_load.return_value = (mock_train_gen, mock_test_gen)
        
        train_gen, test_gen = preprocess_data()
        
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(test_gen)
        self.assertEqual(train_gen.num_batches, 10)
        self.assertEqual(test_gen.num_batches, 3)
    
    @patch('processes.build_model.load_dataset_with_preprocessing')
    def test_preprocess_data_failure(self, mock_load):
        """Test preprocessing handles failures"""
        mock_load.side_effect = Exception("Preprocessing error")
        
        train_gen, test_gen = preprocess_data()
        
        self.assertIsNone(train_gen)
        self.assertIsNone(test_gen)


class TestExtractFeatures(unittest.TestCase):
    """Test cases for feature extraction step"""
    
    def create_mock_generator(self):
        """Create a mock generator for testing"""
        mock_gen = Mock()
        mock_gen.num_batches = 1
        return mock_gen
    
    @patch('processes.build_model.process_features')
    def test_extract_features_success(self, mock_process):
        """Test successful feature extraction"""
        mock_train_gen = self.create_mock_generator()
        mock_test_gen = self.create_mock_generator()
        
        # Create mock feature dictionaries
        mock_train_features = {
            'apple_obj001': {
                'features': np.random.rand(4096),
                'label': 0
            }
        }
        mock_test_features = {
            'apple_obj002': {
                'features': np.random.rand(4096),
                'label': 0
            }
        }
        
        mock_process.side_effect = [mock_test_features, mock_train_features]
        
        train_features, test_features = extract_features(mock_train_gen, mock_test_gen)
        
        self.assertIsNotNone(train_features)
        self.assertIsNotNone(test_features)
        self.assertGreater(len(train_features), 0)
        self.assertGreater(len(test_features), 0)
    
    @patch('processes.build_model.process_features')
    def test_extract_features_failure(self, mock_process):
        """Test feature extraction handles failures"""
        mock_train_gen = self.create_mock_generator()
        mock_test_gen = self.create_mock_generator()

        mock_process.side_effect = Exception("Feature extraction error")

        train_features, test_features = extract_features(mock_train_gen, mock_test_gen)

        # On failure, both should be None
        self.assertIsNone(train_features)
        self.assertIsNone(test_features)


class TestTrainClassifier(unittest.TestCase):
    """Test cases for classifier training step"""
    
    def setUp(self):
        """Set up mock feature data"""
        # Create mock fused features with correct structure
        self.train_features = {
            'market_obj001': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 0
            },
            'market_obj002': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 0
            },
            'standard_obj003': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 1
            },
            'standard_obj004': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 1
            },
            'premium_obj005': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 2
            },
            'premium_obj006': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 2
            }
        }
        
        self.test_features = {
            'market_obj007': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 0
            },
            'standard_obj008': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 1
            },
            'premium_obj009': {
                'features': np.random.rand(4096).astype(np.float32),
                'label': 2
            }
        }
    
    def test_train_classifier_structure(self):
        """Test classifier training returns correct structure"""
        params, results = train_classifier(
            self.train_features,
            self.test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01
        )
        
        # Check params returned
        self.assertIsNotNone(params)
        self.assertIn('W1', params)
        self.assertIn('W2', params)
        
        # Check results structure
        self.assertIsNotNone(results)
        self.assertIn('train_loss', results)
        self.assertIn('train_accuracy', results)
        self.assertIn('test_loss', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('history', results)
        self.assertIn('params', results)
        self.assertIn('label_mapping', results)
    
    def test_train_classifier_label_mapping(self):
        """Test classifier uses correct label mapping"""
        _, results = train_classifier(
            self.train_features,
            self.test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01
        )
        
        label_mapping = results['label_mapping']
        
        self.assertEqual(label_mapping['market'], 0)
        self.assertEqual(label_mapping['standard'], 1)
        self.assertEqual(label_mapping['premium'], 2)
    
    def test_train_classifier_feature_dimensions(self):
        """Test classifier handles correct input dimensions"""
        # Disable PCA to test raw feature dimensions
        params, results = train_classifier(
            self.train_features,
            self.test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01,
            pca_components=None  # Disable PCA
        )

        # W1 should be (4096, hidden_dim) when PCA is disabled
        self.assertEqual(params['W1'].shape[0], 4096)
        self.assertEqual(params['W1'].shape[1], 16)

        # W2 should be (hidden_dim, 3)
        self.assertEqual(params['W2'].shape[0], 16)
        self.assertEqual(params['W2'].shape[1], 3)
    
    def test_train_classifier_empty_features(self):
        """Test classifier handles empty features"""
        params, results = train_classifier({}, {}, hidden_dim=16, epochs=5)
        
        # Should fail gracefully
        self.assertIsNone(params)
        self.assertIsNone(results)


class TestGenerateConfusionMatrix(unittest.TestCase):
    """Test cases for confusion matrix generation"""
    
    def setUp(self):
        """Set up mock results"""
        np.random.seed(42)
        
        self.results = {
            'X_test': np.random.rand(10, 100).astype(np.float32),
            'y_test': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
            'params': {
                'W1': np.random.rand(100, 50),
                'b1': np.zeros((1, 50)),
                'W2': np.random.rand(50, 3),
                'b2': np.zeros((1, 3))
            },
            'label_mapping': {
                'market': 0,
                'standard': 1,
                'premium': 2
            }
        }
    
    @patch('processes.build_model.generate_full_confusion_matrix_report')
    def test_generate_confusion_matrix_success(self, mock_generate):
        """Test successful confusion matrix generation"""
        mock_cm = np.array([[3, 0, 0],
                           [0, 3, 0],
                           [0, 0, 4]])
        mock_report = {'cm': mock_cm}
        mock_generate.return_value = mock_report
        
        cm = generate_confusion_matrix(self.results)
        
        self.assertIsNotNone(cm)
        mock_generate.assert_called_once()
    
    @patch('processes.build_model.generate_full_confusion_matrix_report')
    def test_generate_confusion_matrix_failure(self, mock_generate):
        """Test confusion matrix handles failures"""
        mock_generate.side_effect = Exception("Confusion matrix error")
        
        cm = generate_confusion_matrix(self.results)
        
        self.assertIsNone(cm)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    @patch('processes.build_model.setup_database')
    @patch('processes.build_model.preprocess_data')
    @patch('processes.build_model.extract_features')
    @patch('processes.build_model.train_classifier')
    @patch('processes.build_model.generate_confusion_matrix')
    @patch('processes.build_model.save_dashboard_metadata')
    @patch('os.path.exists')
    def test_run_full_pipeline_success(self, mock_exists, mock_save_metadata, mock_cm, 
                                       mock_train, mock_extract, mock_preprocess, mock_setup):
        """Test successful full pipeline run"""
        # Mock all steps to succeed
        mock_exists.return_value = True
        mock_setup.return_value = True
        
        mock_train_gen = Mock()
        mock_test_gen = Mock()
        mock_preprocess.return_value = (mock_train_gen, mock_test_gen)
        
        mock_features = {
            'obj001': {'features': np.random.rand(100), 'label': 0}
        }
        mock_extract.return_value = (mock_features, mock_features)
        
        mock_params = {
            'W1': np.random.rand(100, 50),
            'W2': np.random.rand(50, 3)
        }
        mock_results = {
            'train_accuracy': 0.95,
            'test_accuracy': 0.90,
            'label_mapping': {'market': 0, 'standard': 1, 'premium': 2}
        }
        mock_train.return_value = (mock_params, mock_results)
        
        mock_cm.return_value = np.eye(3)
        mock_save_metadata.return_value = 'metadata.json'
        
        # Run pipeline
        result = run_full_pipeline(skip_tests=True, epochs=5)
        
        self.assertTrue(result)
    
    @patch('processes.build_model.setup_database')
    @patch('os.path.exists')
    def test_run_full_pipeline_database_failure(self, mock_exists, mock_setup):
        """Test pipeline stops on database setup failure"""
        mock_exists.return_value = False
        mock_setup.return_value = False
        
        result = run_full_pipeline(skip_tests=True)
        
        self.assertFalse(result)
    
    @patch('processes.build_model.setup_database')
    @patch('processes.build_model.preprocess_data')
    @patch('os.path.exists')
    def test_run_full_pipeline_preprocessing_failure(self, mock_exists, 
                                                     mock_preprocess, mock_setup):
        """Test pipeline stops on preprocessing failure"""
        mock_exists.return_value = False
        mock_setup.return_value = True
        mock_preprocess.return_value = (None, None)
        
        result = run_full_pipeline(skip_tests=True)
        
        self.assertFalse(result)
    
    @patch('os.path.exists')
    @patch('processes.build_model.preprocess_data')
    def test_run_full_pipeline_skip_existing_datasets(self, mock_preprocess, mock_exists):
        """Test pipeline skips setup when datasets exist"""
        # Mock both stored and processed datasets existing
        mock_exists.return_value = True
        
        mock_train_gen = Mock()
        mock_test_gen = Mock()
        mock_preprocess.return_value = (mock_train_gen, mock_test_gen)
        
        # The pipeline should skip database setup and preprocessing
        with patch('processes.build_model.setup_database') as mock_setup:
            with patch('processes.build_model.extract_features') as mock_extract:
                # Return proper tuple, not False
                mock_features = {
                    'obj001': {'features': np.random.rand(100), 'label': 0}
                }
                mock_extract.return_value = (mock_features, mock_features)
                
                with patch('processes.build_model.train_classifier') as mock_train:
                    # Mock train to fail early
                    mock_train.return_value = (None, None)
                    
                    run_full_pipeline(skip_tests=True)
                    
                    # setup_database should not be called
                    mock_setup.assert_not_called()

class TestPipelineDataValidation(unittest.TestCase):
    """Test cases for data validation throughout pipeline"""
    
    def test_train_classifier_validates_labels(self):
        """Test classifier validates label consistency"""
        # Create features with proper structure
        train_features = {
            'market_obj001': {'features': np.random.rand(100).astype(np.float32), 'label': 0},
            'market_obj002': {'features': np.random.rand(100).astype(np.float32), 'label': 0},
            'standard_obj003': {'features': np.random.rand(100).astype(np.float32), 'label': 1},
            'premium_obj004': {'features': np.random.rand(100).astype(np.float32), 'label': 2}
        }
        
        test_features = {
            'market_obj005': {'features': np.random.rand(100).astype(np.float32), 'label': 0}
        }
        
        params, results = train_classifier(
            train_features,
            test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01
        )
        
        # Should handle the labels correctly
        self.assertIsNotNone(params)
        self.assertEqual(len(results['label_mapping']), 3)
    
    def test_train_classifier_validates_feature_dimensions(self):
        """Test classifier validates consistent feature dimensions"""
        # All features should have same dimension
        train_features = {
            'market_obj001': {'features': np.random.rand(100).astype(np.float32), 'label': 0},
            'standard_obj002': {'features': np.random.rand(100).astype(np.float32), 'label': 1},
            'premium_obj003': {'features': np.random.rand(100).astype(np.float32), 'label': 2}
        }

        test_features = {
            'market_obj004': {'features': np.random.rand(100).astype(np.float32), 'label': 0}
        }

        params, results = train_classifier(
            train_features,
            test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01,
            pca_components=None  # Disable PCA to test raw dimensions
        )

        # All feature vectors should have same dimension
        self.assertEqual(params['W1'].shape[0], 100)


class TestPipelineOutputs(unittest.TestCase):
    """Test cases for pipeline outputs and artifacts"""
    
    @patch('os.makedirs')
    @patch('processes.build_model.save_model')
    def test_train_classifier_saves_model(self, mock_save, mock_makedirs):
        """Test classifier saves model to correct location"""
        train_features = {
            'market_obj001': {'features': np.random.rand(100).astype(np.float32), 'label': 0},
            'standard_obj002': {'features': np.random.rand(100).astype(np.float32), 'label': 1},
            'premium_obj003': {'features': np.random.rand(100).astype(np.float32), 'label': 2}
        }
        
        test_features = {
            'market_obj004': {'features': np.random.rand(100).astype(np.float32), 'label': 0}
        }
        
        train_classifier(
            train_features,
            test_features,
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01
        )
        
        # Should attempt to save model
        mock_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()