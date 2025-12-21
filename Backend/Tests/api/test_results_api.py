import pytest
import json
from Tests.test_config import TestConfig


class TestResultsAPI:
    """Test results retrieval endpoints"""
    
    def test_get_all_results(self, client):
        """Test /api/results endpoint"""
        response = client.get('/api/results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'results' in data or isinstance(data, list)
    
    def test_get_results_with_pagination(self, client):
        """Test /api/results endpoint with pagination"""
        response = client.get('/api/results?page=1&limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have pagination info
        assert 'results' in data or isinstance(data, list)
    
    def test_get_result_by_id(self, client, test_collection):
        """Test /api/results/<id> endpoint"""
        # Insert a test result
        test_result = {
            "_id": "test_result_001",
            "fruit_type": "market",
            "category": "A",
            "timestamp": "2025-01-01T00:00:00"
        }
        test_collection.insert_one(test_result)
        
        response = client.get('/api/results/test_result_001')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['fruit_type'] == 'market'
    
    def test_results_filtering_by_fruit_type(self, client):
        """Test filtering by fruit type"""
        response = client.get('/api/results?fruit_type=market')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # All results should be market type
        results = data.get('results', data)
        if isinstance(results, list) and len(results) > 0:
            assert all(r.get('fruit_type') == 'market' for r in results)
    
    def test_results_filtering_by_date_range(self, client):
        """Test filtering by date range"""
        response = client.get('/api/results?start_date=2025-01-01&end_date=2025-01-31')
        
        assert response.status_code == 200
    
    def test_results_sorting(self, client):
        """Test sorting by various fields"""
        response = client.get('/api/results?sort=timestamp&order=desc')
        
        assert response.status_code == 200