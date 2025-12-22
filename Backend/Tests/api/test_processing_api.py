"""
Processing Pipeline API Tests
Essential tests for ML pipeline control and monitoring endpoints
"""
import pytest
import json
import time


class TestPipelineControl:
    """Test pipeline start/stop endpoints"""
    
    def test_start_pipeline_success(self, client):
        """Test starting the ML pipeline"""
        response = client.post('/api/pipeline/start')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify response structure
        assert 'success' in data
        assert data['success'] == True
        assert 'pipelineId' in data
        assert 'status' in data
        assert data['status'] == 'started'
    
    def test_start_pipeline_with_config(self, client):
        """Test starting pipeline with custom configuration"""
        config = {
            'skipTests': True,
            'hiddenDim': 32,
            'epochs': 50,
            'learningRate': 0.001,
            'lambdaReg': 0.01
        }
        
        response = client.post('/api/pipeline/start', 
                              data=json.dumps(config),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_start_pipeline_already_running(self, client):
        """Test starting pipeline when already running"""
        # Start first time
        client.post('/api/pipeline/start')
        
        # Try to start again immediately
        response = client.post('/api/pipeline/start')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
        assert 'error' in data
    
    def test_stop_pipeline(self, client):
        """Test stopping the pipeline"""
        # Start pipeline first
        client.post('/api/pipeline/start')
        
        # Stop it
        response = client.post('/api/pipeline/stop')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True


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
    
    def test_get_pipeline_status_running(self, client):
        """Test getting pipeline status while running"""
        # Start pipeline
        client.post('/api/pipeline/start')
        
        # Get status
        response = client.get('/api/pipeline/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should be running
        assert data['running'] == True
        assert data['status'] in ['running', 'processing', 'started']
        assert data['progress'] >= 0
    
    def test_pipeline_status_with_results(self, client):
        """Test status includes results when completed"""
        # Start and wait for completion (in real scenario)
        client.post('/api/pipeline/start')
        time.sleep(0.5)  # Small delay for async processing
        
        response = client.get('/api/pipeline/status')
        data = json.loads(response.data)
        
        # Check for results if completed
        if data['status'] == 'completed':
            assert 'totalProcessed' in data
            assert 'accuracy' in data
            assert data['totalProcessed'] >= 0
            assert 0.0 <= data['accuracy'] <= 1.0


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
    
    def test_pipeline_logs_after_start(self, client):
        """Test logs are generated after starting pipeline"""
        # Start pipeline
        client.post('/api/pipeline/start')
        
        # Get logs
        response = client.get('/api/pipeline/logs')
        data = json.loads(response.data)
        
        # Should have at least startup logs
        assert len(data) > 0


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
        
        # Verify defaults
        assert data['hiddenDim'] > 0
        assert data['epochs'] > 0
        assert data['learningRate'] > 0
        assert data['lambdaReg'] >= 0
        assert data['batchSize'] > 0
    
    def test_update_pipeline_config(self, client):
        """Test updating pipeline configuration"""
        new_config = {
            'hiddenDim': 64,
            'epochs': 200,
            'learningRate': 0.002,
            'lambdaReg': 0.005,
            'batchSize': 64
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


class TestPipelineWorkflow:
    """Test complete pipeline workflows"""
    
    def test_complete_pipeline_workflow(self, client):
        """Test start -> status -> stop workflow"""
        # 1. Start pipeline
        start_response = client.post('/api/pipeline/start')
        assert start_response.status_code == 200
        
        # 2. Check status
        status_response = client.get('/api/pipeline/status')
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        assert status_data['running'] == True
        
        # 3. Stop pipeline
        stop_response = client.post('/api/pipeline/stop')
        assert stop_response.status_code == 200
        
        # 4. Verify stopped
        final_status = client.get('/api/pipeline/status')
        final_data = json.loads(final_status.data)
        assert final_data['status'] in ['stopped', 'idle']
    
    def test_config_then_start_workflow(self, client):
        """Test updating config before starting pipeline"""
        # 1. Update config
        config = {'hiddenDim': 32, 'epochs': 100}
        config_response = client.put('/api/pipeline/config',
                                     data=json.dumps(config),
                                     content_type='application/json')
        assert config_response.status_code == 200
        
        # 2. Start with new config
        start_response = client.post('/api/pipeline/start')
        assert start_response.status_code == 200
        
        # 3. Verify config persists
        status_response = client.get('/api/pipeline/config')
        status_data = json.loads(status_response.data)
        assert status_data['hiddenDim'] == 32


class TestErrorHandling:
    """Test error handling for pipeline endpoints"""
    
    def test_invalid_config_format(self, client):
        """Test pipeline handles invalid config format"""
        invalid_config = "not a json"
        
        response = client.put('/api/pipeline/config',
                            data=invalid_config,
                            content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    def test_pipeline_endpoints_with_errors(self, client):
        """Test all endpoints handle errors gracefully"""
        endpoints = [
            ('/api/pipeline/status', 'GET'),
            ('/api/pipeline/logs', 'GET'),
            ('/api/pipeline/config', 'GET')
        ]
        
        for endpoint, method in endpoints:
            if method == 'GET':
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            
            # Should return 200 with valid data
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])