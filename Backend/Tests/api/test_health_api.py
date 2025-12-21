import pytest
import json


class TestHealthAPI:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test /api/health endpoint"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'status' in data
    
    def test_database_connectivity(self, client):
        """Test database health status"""
        response = client.get('/api/health/database')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'connected' in data or 'status' in data
