"""
Settings API Tests
Essential tests for system settings and configuration endpoints
"""
import pytest
import json
from pathlib import Path
import sys

class TestGetSettings:
    """Test get settings endpoint"""
    
    def test_get_all_settings(self, client):
        """Test getting all system settings"""
        response = client.get('/api/settings')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields
        assert 'dbName' in data
        assert 'mongoConnection' in data
        assert 'storedDataset' in data
        assert 'originalDataset' in data
        assert 'processedDataset' in data
        assert 'cameraFps' in data
        assert 'numCameras' in data
        assert 'imageSize' in data
        assert 'batchSize' in data
        
        # Verify types
        assert isinstance(data, dict)
    
    def test_settings_values(self, client):
        """Test settings have valid values"""
        response = client.get('/api/settings')
        data = json.loads(response.data)
        
        # Camera FPS should be positive
        if data['cameraFps'] is not None:
            assert data['cameraFps'] > 0
        
        # Number of cameras should be positive
        if data['numCameras'] is not None:
            assert data['numCameras'] > 0
        
        # Image size should be correct format
        assert data['imageSize'] == '224x224'
        
        # Batch size should be positive
        if data['batchSize'] is not None:
            assert data['batchSize'] > 0
    
    def test_settings_returns_strings(self, client):
        """Test settings fields are proper types"""
        response = client.get('/api/settings')
        data = json.loads(response.data)
        
        # Database name should be string (if set)
        if data['dbName']:
            assert isinstance(data['dbName'], str)
        
        # Paths should be strings (if set)
        if data['storedDataset']:
            assert isinstance(data['storedDataset'], str)


class TestUpdateSettings:
    """Test update settings endpoint"""
    
    def test_update_settings_success(self, client):
        """Test updating system settings"""
        new_settings = {
            'cameraFps': 25,
            'batchSize': 64,
            'numCameras': 4
        }
        
        response = client.put('/api/settings',
                             data=json.dumps(new_settings),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return the settings
        assert isinstance(data, dict)
    
    def test_update_partial_settings(self, client):
        """Test updating only some settings"""
        partial_settings = {
            'cameraFps': 30
        }
        
        response = client.put('/api/settings',
                             data=json.dumps(partial_settings),
                             content_type='application/json')
        
        assert response.status_code == 200
    
    def test_update_settings_with_valid_values(self, client):
        """Test updating settings with valid values"""
        valid_settings = {
            'cameraFps': 60,
            'batchSize': 128,
            'numCameras': 4,
            'imageSize': '224x224'
        }
        
        response = client.put('/api/settings',
                             data=json.dumps(valid_settings),
                             content_type='application/json')
        
        assert response.status_code == 200


class TestDatabaseConnection:
    """Test database connection testing endpoint"""
    
    def test_test_database_without_connection_string(self, client):
        """Test database test without connection string"""
        response = client.post('/api/settings/test-database',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['success'] is False
        assert 'message' in data
    
    def test_test_database_with_invalid_connection(self, client):
        """Test database with invalid connection string"""
        invalid_connection = {
            'connectionString': 'mongodb://invalid:27017'
        }
        
        response = client.post('/api/settings/test-database',
                              data=json.dumps(invalid_connection),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should indicate failure
        assert 'success' in data
        assert 'message' in data
        assert data['success'] is False
    
    def test_test_database_response_structure(self, client):
        """Test database test response has correct structure"""
        test_data = {
            'connectionString': 'mongodb://localhost:27017'
        }
        
        response = client.post('/api/settings/test-database',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        
        # Should have success and message
        assert 'success' in data
        assert 'message' in data
        assert isinstance(data['success'], bool)
        assert isinstance(data['message'], str)
    
    def test_test_database_with_empty_string(self, client):
        """Test database with empty connection string"""
        empty_connection = {
            'connectionString': ''
        }
        
        response = client.post('/api/settings/test-database',
                              data=json.dumps(empty_connection),
                              content_type='application/json')
        
        assert response.status_code == 400


class TestSettingsStatus:
    """Test settings status endpoint"""
    
    def test_get_settings_status(self, client):
        """Test getting system status for settings page"""
        response = client.get('/api/settings/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields
        assert 'database' in data
        assert 'model' in data
        assert 'pipeline' in data
        assert 'cameras' in data
        
        # Verify types
        assert isinstance(data['database'], str)
        assert isinstance(data['model'], str)
        assert isinstance(data['pipeline'], str)
        assert isinstance(data['cameras'], list)
    
    def test_settings_status_values(self, client):
        """Test settings status has valid values"""
        response = client.get('/api/settings/status')
        data = json.loads(response.data)
        
        # Database should be valid state
        assert data['database'] in ['connected', 'disconnected']
        
        # Model should be valid state
        assert data['model'] in ['loaded', 'trained', 'not_trained', 'unknown']
        
        # Pipeline should be valid state
        assert data['pipeline'] in ['idle', 'running', 'completed', 'failed', 'stopped']
        
        # Cameras should be 4 booleans
        assert len(data['cameras']) == 4
        assert all(isinstance(c, bool) for c in data['cameras'])
    
    def test_settings_status_camera_count(self, client):
        """Test camera count matches configuration"""
        response = client.get('/api/settings/status')
        data = json.loads(response.data)
        
        # Should have 4 cameras
        assert len(data['cameras']) == 4


class TestDatasetPaths:
    """Test dataset paths endpoints"""
    
    def test_get_dataset_paths(self, client):
        """Test getting dataset paths"""
        response = client.get('/api/settings/paths')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all path fields
        assert 'storedDataset' in data
        assert 'originalDataset' in data
        assert 'processedDataset' in data
    
    def test_update_dataset_paths(self, client):
        """Test updating dataset paths"""
        new_paths = {
            'storedDataset': '/path/to/stored',
            'originalDataset': '/path/to/original',
            'processedDataset': '/path/to/processed'
        }
        
        response = client.put('/api/settings/paths',
                             data=json.dumps(new_paths),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return the paths
        assert isinstance(data, dict)
    
    def test_update_partial_paths(self, client):
        """Test updating only some paths"""
        partial_paths = {
            'storedDataset': '/new/path/to/stored'
        }
        
        response = client.put('/api/settings/paths',
                             data=json.dumps(partial_paths),
                             content_type='application/json')
        
        assert response.status_code == 200
    
    def test_paths_are_strings(self, client):
        """Test that paths are returned as strings"""
        response = client.get('/api/settings/paths')
        data = json.loads(response.data)
        
        # Check types (allow None)
        if data['storedDataset'] is not None:
            assert isinstance(data['storedDataset'], str)
        if data['originalDataset'] is not None:
            assert isinstance(data['originalDataset'], str)
        if data['processedDataset'] is not None:
            assert isinstance(data['processedDataset'], str)


class TestSettingsIntegration:
    """Test integration between settings endpoints"""
    
    def test_settings_and_status_consistency(self, client):
        """Test consistency between settings and status"""
        # Get settings
        settings_response = client.get('/api/settings')
        settings_data = json.loads(settings_response.data)
        
        # Get status
        status_response = client.get('/api/settings/status')
        status_data = json.loads(status_response.data)
        
        # Number of cameras should match
        assert settings_data['numCameras'] == len(status_data['cameras'])
    
    def test_paths_in_settings(self, client):
        """Test that paths in settings match paths endpoint"""
        # Get settings
        settings_response = client.get('/api/settings')
        settings_data = json.loads(settings_response.data)
        
        # Get paths
        paths_response = client.get('/api/settings/paths')
        paths_data = json.loads(paths_response.data)
        
        # Paths should match
        assert settings_data['storedDataset'] == paths_data['storedDataset']
        assert settings_data['originalDataset'] == paths_data['originalDataset']
        assert settings_data['processedDataset'] == paths_data['processedDataset']
    
    def test_update_and_retrieve_workflow(self, client):
        """Test updating and retrieving settings"""
        # Update settings
        new_settings = {'cameraFps': 25}
        update_response = client.put('/api/settings',
                                     data=json.dumps(new_settings),
                                     content_type='application/json')
        
        assert update_response.status_code == 200
        
        # Retrieve settings (they may not persist in test mode)
        get_response = client.get('/api/settings')
        assert get_response.status_code == 200


class TestErrorHandling:
    """Test error handling for settings endpoints"""
    
    def test_invalid_json_format(self, client):
        """Test invalid JSON format handling"""
        invalid_json = "not a json"
        
        response = client.put('/api/settings',
                             data=invalid_json,
                             content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    def test_settings_endpoints_resilience(self, client):
        """Test all settings endpoints are resilient"""
        endpoints = [
            ('/api/settings', 'GET'),
            ('/api/settings/status', 'GET'),
            ('/api/settings/paths', 'GET')
        ]
        
        for endpoint, method in endpoints:
            if method == 'GET':
                response = client.get(endpoint)
            
            # Should return 200
            assert response.status_code == 200
    
    def test_invalid_http_methods(self, client):
        """Test endpoints reject invalid HTTP methods"""
        # Settings endpoint should not accept DELETE
        response = client.delete('/api/settings')
        assert response.status_code in [405, 404]
        
        # Status should not accept POST
        response = client.post('/api/settings/status')
        assert response.status_code in [405, 404]
    
    def test_malformed_database_test_request(self, client):
        """Test database test with malformed request"""
        malformed_data = {'invalid_field': 'value'}
        
        response = client.post('/api/settings/test-database',
                              data=json.dumps(malformed_data),
                              content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestResponseFormats:
    """Test response formats are correct"""
    
    def test_all_endpoints_return_json(self, client):
        """Test all GET endpoints return valid JSON"""
        endpoints = [
            '/api/settings',
            '/api/settings/status',
            '/api/settings/paths'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert 'application/json' in response.content_type
            
            # Should be valid JSON
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON from {endpoint}")
    
    def test_post_endpoints_return_json(self, client):
        """Test POST endpoints return JSON"""
        test_data = {'connectionString': 'mongodb://localhost:27017'}
        
        response = client.post('/api/settings/test-database',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert 'application/json' in response.content_type


class TestSettingsValidation:
    """Test settings value validation"""
    
    def test_camera_fps_range(self, client):
        """Test camera FPS is in valid range"""
        response = client.get('/api/settings')
        data = json.loads(response.data)
        
        if data['cameraFps'] is not None:
            assert 1 <= data['cameraFps'] <= 120
    
    def test_num_cameras_range(self, client):
        """Test number of cameras is valid"""
        response = client.get('/api/settings')
        data = json.loads(response.data)
        
        if data['numCameras'] is not None:
            assert 1 <= data['numCameras'] <= 10
    
    def test_batch_size_range(self, client):
        """Test batch size is in valid range"""
        response = client.get('/api/settings')
        data = json.loads(response.data)
        
        if data['batchSize'] is not None:
            assert 1 <= data['batchSize'] <= 512


class TestSettingsEdgeCases:
    """Test edge cases for settings"""
    
    def test_update_with_empty_object(self, client):
        """Test updating with empty object"""
        response = client.put('/api/settings',
                             data=json.dumps({}),
                             content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_concurrent_settings_requests(self, client):
        """Test handling concurrent settings requests"""
        import threading
        
        results = []
        
        def get_settings():
            response = client.get('/api/settings')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = [threading.Thread(target=get_settings) for _ in range(5)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
