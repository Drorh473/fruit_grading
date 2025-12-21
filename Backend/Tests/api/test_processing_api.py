import pytest
import json
from Tests.test_config import TestConfig


class TestProcessingAPI:
    """Test processing pipeline endpoints"""
    
    def test_start_processing(self, client):
        """Test /api/pipeline/start endpoint"""
        response = client.post('/api/pipeline/start')
        
        assert response.status_code in [200, 201]
        data = json.loads(response.data)
        
        assert 'success' in data or 'status' in data
    
    def test_stop_processing(self, client):
        """Test /api/pipeline/stop endpoint"""
        response = client.post('/api/pipeline/stop')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'success' in data or 'status' in data
    
    def test_get_processing_status(self, client):
        """Test /api/pipeline/status endpoint"""
        response = client.get('/api/pipeline/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'status' in data or 'state' in data
    
    def test_processing_progress_tracking(self, client):
        """Test real-time progress updates"""
        # Start processing
        client.post('/api/pipeline/start')
        
        # Get status
        response = client.get('/api/pipeline/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have progress information
        assert 'progress' in data or 'status' in data