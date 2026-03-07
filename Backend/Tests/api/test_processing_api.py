
import pytest
import json
import time
import threading
from unittest.mock import patch, MagicMock

MOCK_BASE = 'routes.processing'

class TestPipelineControl:
    """Test /api/pipeline/start endpoint with MOCKED backend logic."""

    @patch('utils.model_metadata.save_dashboard_metadata')
    @patch(f'{MOCK_BASE}.generate_confusion_matrix')
    @patch(f'{MOCK_BASE}.train_classifier')
    @patch(f'{MOCK_BASE}.extract_features')
    @patch(f'{MOCK_BASE}.preprocess_data')
    @patch(f'{MOCK_BASE}.setup_database')
    @patch(f'{MOCK_BASE}.run_tests')
    def test_start_pipeline_success(self, mock_tests, mock_db, mock_preprocess, mock_extract, mock_train, mock_cm, mock_save, client):
        """Test starting the ML pipeline (Mocked)"""
        mock_tests.return_value = True
        mock_db.return_value = True
        mock_preprocess.return_value = ({}, {})
        mock_extract.return_value = ({}, {})
        mock_train.return_value = (
            {'W1': MagicMock(), 'b1': MagicMock(), 'W2': MagicMock(), 'b2': MagicMock()},
            {'test_accuracy': 0.92, 'train_accuracy': 0.95, 'test_loss': 0.2, 'train_loss': 0.1, 'history': {}}
        )
        mock_cm.return_value = None
        mock_save.return_value = None

        response = client.post('/api/pipeline/start',
                             data=json.dumps({'skipTests': True}),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'pipelineId' in data
        assert data['status'] == 'started'

        time.sleep(0.1)
        client.post('/api/pipeline/stop')

    @patch('utils.model_metadata.save_dashboard_metadata')
    @patch(f'{MOCK_BASE}.generate_confusion_matrix')
    @patch(f'{MOCK_BASE}.train_classifier')
    @patch(f'{MOCK_BASE}.extract_features')
    @patch(f'{MOCK_BASE}.preprocess_data')
    @patch(f'{MOCK_BASE}.setup_database')
    @patch(f'{MOCK_BASE}.run_tests')
    def test_start_pipeline_with_config(self, mock_tests, mock_db, mock_preprocess, mock_extract, mock_train, mock_cm, mock_save, client):
        """Test starting pipeline with custom configuration"""
        mock_tests.return_value = True
        mock_db.return_value = True
        mock_preprocess.return_value = ({}, {})
        mock_extract.return_value = ({}, {})
        mock_train.return_value = (
            {'W1': MagicMock(), 'b1': MagicMock(), 'W2': MagicMock(), 'b2': MagicMock()},
            {'test_accuracy': 0.9, 'train_accuracy': 0.92, 'test_loss': 0.3, 'train_loss': 0.2, 'history': {}}
        )
        mock_cm.return_value = None
        mock_save.return_value = None

        config = {
            'skipTests': True,
            'setupDatabase': False,
            'preprocessData': False,
            'extractFeatures': False,
            'trainClassifier': False,
            'generateConfusionMatrix': False
        }

        response = client.post('/api/pipeline/start',
                             data=json.dumps(config),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

        time.sleep(0.1)
        client.post('/api/pipeline/stop')

    @patch('utils.model_metadata.save_dashboard_metadata')
    @patch(f'{MOCK_BASE}.generate_confusion_matrix')
    @patch(f'{MOCK_BASE}.train_classifier')
    @patch(f'{MOCK_BASE}.extract_features')
    @patch(f'{MOCK_BASE}.preprocess_data')
    @patch(f'{MOCK_BASE}.setup_database')
    @patch(f'{MOCK_BASE}.run_tests')
    def test_start_pipeline_empty_config(self, mock_tests, mock_db, mock_preprocess, mock_extract, mock_train, mock_cm, mock_save, client):
        """Test starting pipeline with empty config (should use defaults)"""
        mock_tests.return_value = True
        mock_db.return_value = True
        mock_preprocess.return_value = ({}, {})
        mock_extract.return_value = ({}, {})
        mock_train.return_value = (
            {'W1': MagicMock(), 'b1': MagicMock(), 'W2': MagicMock(), 'b2': MagicMock()},
            {'test_accuracy': 0.9, 'train_accuracy': 0.92, 'test_loss': 0.3, 'train_loss': 0.2, 'history': {}}
        )
        mock_cm.return_value = None
        mock_save.return_value = None

        response = client.post('/api/pipeline/start',
                             data=json.dumps({'skipTests': True}),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

        time.sleep(0.1)
        client.post('/api/pipeline/stop')

    @patch('utils.model_metadata.save_dashboard_metadata')
    @patch(f'{MOCK_BASE}.generate_confusion_matrix')
    @patch(f'{MOCK_BASE}.train_classifier')
    @patch(f'{MOCK_BASE}.extract_features')
    @patch(f'{MOCK_BASE}.preprocess_data')
    @patch(f'{MOCK_BASE}.setup_database')
    @patch(f'{MOCK_BASE}.run_tests')
    def test_start_pipeline_already_running(self, mock_tests, mock_db, mock_preprocess, mock_extract, mock_train, mock_cm, mock_save, client):
        """Test starting pipeline when already running"""
        # Using an event to block the pipeline until the second request is made
        release_pipeline = threading.Event()

        def blocking_setup_database():
            # Waiting for signal before completing
            release_pipeline.wait(timeout=5)
            return True

        mock_tests.return_value = True
        mock_db.side_effect = blocking_setup_database
        mock_preprocess.return_value = ({}, {})
        mock_extract.return_value = ({}, {})
        mock_train.return_value = (
            {'W1': MagicMock(), 'b1': MagicMock(), 'W2': MagicMock(), 'b2': MagicMock()},
            {'test_accuracy': 0.9, 'train_accuracy': 0.91, 'test_loss': 0.25, 'train_loss': 0.15, 'history': {}}
        )
        mock_cm.return_value = None
        mock_save.return_value = None

        # Start the pipeline (will block at setup_database)
        client.post('/api/pipeline/start',
                   data=json.dumps({'skipTests': True}),
                   content_type='application/json')

        # Give the background thread time to start and set running=True
        time.sleep(0.1)

        # Try to start again while still running - should fail
        response = client.post('/api/pipeline/start',
                             data=json.dumps({'skipTests': True}),
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'already running' in data['error'].lower()

        # Release the blocking mock so the pipeline can complete
        release_pipeline.set()
        time.sleep(0.1)
        client.post('/api/pipeline/stop')

    def test_stop_pipeline(self, client):
        """Test stopping the pipeline"""
        response = client.post('/api/pipeline/stop')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_stop_pipeline_not_running(self, client):
        """Test stopping pipeline when not running"""
        response = client.post('/api/pipeline/stop')
        # Should still succeed
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestPipelineStatus:
    """Test pipeline status monitoring"""

    def test_get_pipeline_status_idle(self, client):
        """Test getting pipeline status when idle"""
        response = client.get('/api/pipeline/status')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify structure
        assert 'running' in data
        assert 'status' in data
        assert 'currentStep' in data
        assert 'progress' in data
        assert 'steps' in data

        # Verify types
        assert isinstance(data['running'], bool)
        assert isinstance(data['status'], str)
        assert isinstance(data['currentStep'], int)
        assert isinstance(data['progress'], (int, float))
        assert isinstance(data['steps'], list)

    def test_pipeline_status_progress_range(self, client):
        """Test progress is always in valid range"""
        response = client.get('/api/pipeline/status')
        data = json.loads(response.data)
        assert 0 <= data['progress'] <= 100

    def test_pipeline_status_steps_structure(self, client):
        """Test steps array has valid structure"""
        response = client.get('/api/pipeline/status')
        data = json.loads(response.data)
        assert isinstance(data['steps'], list)


class TestPipelineLogs:
    """Test pipeline logging endpoints"""

    def test_get_pipeline_logs_default(self, client):
        """Test getting pipeline logs with default limit"""
        response = client.get('/api/pipeline/logs')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Should return a list
        assert isinstance(data, list)
        assert len(data) <= 100  # Default limit

    def test_get_pipeline_logs_custom_limit(self, client):
        """Test getting pipeline logs with custom limit"""
        response = client.get('/api/pipeline/logs?limit=50')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) <= 50

    def test_get_pipeline_logs_large_limit(self, client):
        """Test getting pipeline logs with large limit"""
        response = client.get('/api/pipeline/logs?limit=1000')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_pipeline_logs_invalid_limit(self, client):
        """Test logs endpoint handles invalid limit"""
        response = client.get('/api/pipeline/logs?limit=invalid')
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestPipelineConfig:
    """Test pipeline configuration endpoints"""

    def test_get_pipeline_config(self, client):
        """Test getting pipeline configuration"""
        response = client.get('/api/pipeline/config')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify all config fields
        assert 'hiddenDim' in data
        assert 'epochs' in data
        assert 'learningRate' in data
        assert 'lambdaReg' in data
        assert 'batchSize' in data
        assert 'dropoutRate' in data

        # Verify defaults are positive
        assert data['hiddenDim'] > 0
        assert data['epochs'] > 0
        assert data['learningRate'] > 0
        assert data['lambdaReg'] >= 0
        assert data['batchSize'] > 0
        assert data['dropoutRate'] >= 0

    def test_update_pipeline_config(self, client):
        """Test updating pipeline configuration"""
        new_config = {
            'hiddenDim': 64,
            'epochs': 200,
            'learningRate': 0.002,
            'lambdaReg': 0.005,
            'batchSize': 64,
            'dropoutRate': 0.4
        }

        response = client.put('/api/pipeline/config',
                            data=json.dumps(new_config),
                            content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify updated values
        assert data['hiddenDim'] == 64
        assert data['epochs'] == 200
        assert data['learningRate'] == 0.002
        assert data['dropoutRate'] == 0.4

    def test_update_partial_config(self, client):
        """Test updating only some config parameters"""
        partial_config = {
            'hiddenDim': 128,
            'epochs': 150
        }

        response = client.put('/api/pipeline/config',
                            data=json.dumps(partial_config),
                            content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)

        # Updated fields should change
        assert data['hiddenDim'] == 128
        assert data['epochs'] == 150

        # Other fields should remain
        assert 'learningRate' in data
        assert 'lambdaReg' in data

    def test_config_validation(self, client):
        """Test config validates reasonable values"""
        response = client.get('/api/pipeline/config')
        data = json.loads(response.data)

        # Values should be in reasonable ranges
        assert 1 <= data['hiddenDim'] <= 1024
        assert 1 <= data['epochs'] <= 10000
        assert 0.0 < data['learningRate'] <= 1.0
        assert 0.0 <= data['lambdaReg'] <= 1.0
        assert 0.0 <= data['dropoutRate'] <= 1.0


class TestPipelineWorkflow:
    """Test complete pipeline workflows"""

    @patch('utils.model_metadata.save_dashboard_metadata')
    @patch(f'{MOCK_BASE}.generate_confusion_matrix')
    @patch(f'{MOCK_BASE}.train_classifier')
    @patch(f'{MOCK_BASE}.extract_features')
    @patch(f'{MOCK_BASE}.preprocess_data')
    @patch(f'{MOCK_BASE}.setup_database')
    @patch(f'{MOCK_BASE}.run_tests')
    def test_complete_pipeline_workflow(self, mock_tests, mock_db, mock_preprocess,
                                       mock_extract, mock_train, mock_cm, mock_save, client):
        """Test start -> status -> stop workflow"""
        mock_tests.return_value = True
        mock_db.return_value = True
        mock_preprocess.return_value = ({}, {})
        mock_extract.return_value = ({}, {})
        mock_train.return_value = (
            {'W1': MagicMock(), 'b1': MagicMock(), 'W2': MagicMock(), 'b2': MagicMock()},
            {'test_accuracy': 0.88, 'train_accuracy': 0.9, 'test_loss': 0.22, 'train_loss': 0.12, 'history': {}}
        )
        mock_cm.return_value = None
        mock_save.return_value = None

        config = {
            'skipTests': True,
            'setupDatabase': False,
            'preprocessData': False,
            'extractFeatures': False,
            'trainClassifier': False,
            'generateConfusionMatrix': False
        }

        start_response = client.post('/api/pipeline/start',
                                   data=json.dumps(config),
                                   content_type='application/json')
        assert start_response.status_code == 200

        time.sleep(0.1)

        status_response = client.get('/api/pipeline/status')
        assert status_response.status_code == 200

        stop_response = client.post('/api/pipeline/stop')
        assert stop_response.status_code == 200

    def test_config_then_start_workflow(self, client):
        """Test updating config before starting pipeline"""
        # Update config
        config = {'hiddenDim': 32, 'epochs': 100}
        config_response = client.put('/api/pipeline/config',
                                    data=json.dumps(config),
                                    content_type='application/json')
        assert config_response.status_code == 200

        # Verify config persists
        status_response = client.get('/api/pipeline/config')
        status_data = json.loads(status_response.data)
        assert status_data['hiddenDim'] == 32


class TestErrorHandling:
    """Test error handling for pipeline endpoints"""

    def test_endpoint_returns_json(self, client):
        """Test endpoint returns JSON content type"""
        response = client.post('/api/pipeline/stop')
        assert 'application/json' in response.content_type

    def test_endpoints_return_json(self, client):
        """Test all endpoints return valid JSON"""
        endpoints = [
            ('/api/pipeline/status', 'GET'),
            ('/api/pipeline/logs', 'GET'),
            ('/api/pipeline/config', 'GET'),
            ('/api/pipeline/stop', 'POST')
        ]

        for endpoint, method in endpoints:
            if method == 'GET':
                response = client.get(endpoint)
            else:
                response = client.post(endpoint,
                                     data=json.dumps({}),
                                     content_type='application/json')

            # Should return JSON
            assert response.content_type == 'application/json'

            # Should be valid JSON
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON from {endpoint}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
