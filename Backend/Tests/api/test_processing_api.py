"""
Processing Pipeline API Tests
Essential tests for ML pipeline control and monitoring endpoints
"""
import pytest
import json
import time
from pathlib import Path
import sys

# ==================== Tests ====================

class TestPipelineControl:
    """Test pipeline start/stop endpoints"""
    
    def test_start_pipeline_success(self, client):
        """Test starting the ML pipeline"""
        response = client.post('/api/pipeline/start')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify response structure
        assert 'success' in data
        assert data['success'] is True
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
        assert data['success'] is True
    
    def test_start_pipeline_empty_config(self, client):
        """Test starting pipeline with empty config (should use defaults)"""
        response = client.post('/api/pipeline/start',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_start_pipeline_already_running(self, client):
        """Test starting pipeline when already running"""
        # Start first time
        client.post('/api/pipeline/start')
        
        # Try to start again immediately
        response = client.post('/api/pipeline/start')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data
        assert 'already running' in data['error'].lower()
    
    def test_stop_pipeline(self, client):
        """Test stopping the pipeline"""
        # Start pipeline first
        client.post('/api/pipeline/start')
        
        # Small delay to ensure it's started
        time.sleep(0.1)
        
        # Stop it
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
    
    def test_get_pipeline_status_running(self, client):
        """Test getting pipeline status while running"""
        # Start pipeline
        client.post('/api/pipeline/start')
        
        # Small delay
        time.sleep(0.1)
        
        # Get status
        response = client.get('/api/pipeline/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should be running
        assert data['running'] is True
        assert data['status'] in ['running', 'processing', 'started']
        assert data['progress'] >= 0
        assert data['progress'] <= 100
    
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
        
        # If steps exist, verify structure
        for step in data['steps']:
            if isinstance(step, dict):
                assert 'name' in step or 'status' in step


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
    
    def test_pipeline_logs_after_start(self, client):
        """Test logs are generated after starting pipeline"""
        # Start pipeline
        client.post('/api/pipeline/start')
        
        # Small delay
        time.sleep(0.1)
        
        # Get logs
        response = client.get('/api/pipeline/logs')
        data = json.loads(response.data)
        
        # Should have at least startup logs
        assert len(data) > 0
    
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
        
        # Verify defaults are positive
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
    
    def test_config_validation(self, client):
        """Test config validates reasonable values"""
        response = client.get('/api/pipeline/config')
        data = json.loads(response.data)
        
        # Values should be in reasonable ranges
        assert 1 <= data['hiddenDim'] <= 1024
        assert 1 <= data['epochs'] <= 10000
        assert 0.0 < data['learningRate'] <= 1.0
        assert 0.0 <= data['lambdaReg'] <= 1.0


class TestPipelineWorkflow:
    """Test complete pipeline workflows"""
    
    def test_complete_pipeline_workflow(self, client):
        """Test start -> status -> stop workflow"""
        # 1. Start pipeline
        start_response = client.post('/api/pipeline/start')
        assert start_response.status_code == 200
        
        # Small delay
        time.sleep(0.1)
        
        # 2. Check status
        status_response = client.get('/api/pipeline/status')
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        assert status_data['running'] is True
        
        # 3. Stop pipeline
        stop_response = client.post('/api/pipeline/stop')
        assert stop_response.status_code == 200
        
        # Small delay
        time.sleep(0.1)
        
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
    
    def test_multiple_status_checks(self, client):
        """Test multiple status checks during pipeline execution"""
        # Start pipeline
        client.post('/api/pipeline/start')
        
        # Check status multiple times
        for _ in range(3):
            response = client.get('/api/pipeline/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'status' in data
            time.sleep(0.1)


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
    
    def test_invalid_http_methods(self, client):
        """Test endpoints reject invalid HTTP methods"""
        # Start should not accept GET
        response = client.get('/api/pipeline/start')
        assert response.status_code in [405, 404]
        
        # Status should not accept POST
        response = client.post('/api/pipeline/status')
        assert response.status_code in [405, 404]
    
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
    
    def test_endpoints_return_json(self, client):
        """Test all endpoints return valid JSON"""
        endpoints = [
            ('/api/pipeline/status', 'GET'),
            ('/api/pipeline/logs', 'GET'),
            ('/api/pipeline/config', 'GET'),
            ('/api/pipeline/start', 'POST'),
            ('/api/pipeline/stop', 'POST')
        ]
        
        for endpoint, method in endpoints:
            if method == 'GET':
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)
            
            # Should return JSON
            assert response.content_type == 'application/json'
            
            # Should be valid JSON
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON from {endpoint}")


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    def test_status_endpoint_performance(self, client):
        """Test status endpoint responds quickly"""
        start = time.time()
        response = client.get('/api/pipeline/status')
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should be fast
    
    def test_config_endpoint_performance(self, client):
        """Test config endpoint responds quickly"""
        start = time.time()
        response = client.get('/api/pipeline/config')
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0


class TestPipelineEdgeCases:
    """Test edge cases and unusual scenarios"""
    
    def test_rapid_start_stop_requests(self, client):
        """Test handling rapid start/stop requests"""
        # Start
        response1 = client.post('/api/pipeline/start')
        assert response1.status_code == 200
        
        # Stop immediately
        response2 = client.post('/api/pipeline/stop')
        assert response2.status_code == 200
        
        # Start again
        time.sleep(0.2)  # Small delay
        response3 = client.post('/api/pipeline/start')
        assert response3.status_code == 200
    
    def test_concurrent_status_requests(self, client):
        """Test handling concurrent status requests"""
        import threading
        
        results = []
        
        def get_status():
            response = client.get('/api/pipeline/status')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = [threading.Thread(target=get_status) for _ in range(5)]
        
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
