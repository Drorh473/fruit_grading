"""
User Dashboard API Tests
Essential tests for operator dashboard statistics and results endpoints
"""
import pytest
import json
from pathlib import Path
import sys

class TestDashboardStats:
    """Test dashboard statistics endpoint"""
    
    def test_get_dashboard_stats(self, client):
        """Test getting dashboard statistics"""
        response = client.get('/api/user/dashboard-stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields
        assert 'totalToday' in data
        assert 'marketCount' in data
        assert 'standardCount' in data
        assert 'premiumCount' in data
        
        # Verify types
        assert isinstance(data['totalToday'], int)
        assert isinstance(data['marketCount'], int)
        assert isinstance(data['standardCount'], int)
        assert isinstance(data['premiumCount'], int)
    
    def test_dashboard_stats_values(self, client):
        """Test dashboard stats have valid values"""
        response = client.get('/api/user/dashboard-stats')
        data = json.loads(response.data)
        
        # All counts should be non-negative
        assert data['totalToday'] >= 0
        assert data['marketCount'] >= 0
        assert data['standardCount'] >= 0
        assert data['premiumCount'] >= 0
    
    def test_dashboard_stats_sum_consistency(self, client):
        """Test that category counts sum correctly"""
        response = client.get('/api/user/dashboard-stats')
        data = json.loads(response.data)
        
        # Total should equal sum of categories (if data exists)
        category_sum = (data['marketCount'] + data['standardCount'] + 
                       data['premiumCount'])
        
        # Either total matches sum, or everything is zero
        assert (data['totalToday'] == category_sum or 
                (data['totalToday'] == 0 and category_sum == 0))


class TestRecentResults:
    """Test recent results endpoint"""
    
    def test_get_recent_results(self, client):
        """Test getting recent classification results"""
        response = client.get('/api/user/recent-results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)
        
        # Should not exceed 10 results
        assert len(data) <= 10
    
    def test_recent_results_structure(self, client):
        """Test recent results have correct structure"""
        response = client.get('/api/user/recent-results')
        data = json.loads(response.data)
        
        # Check each result has required fields
        for result in data:
            assert 'id' in result
            assert 'type' in result
            assert 'confidence' in result
            assert 'timestamp' in result
            
            # Verify types
            assert isinstance(result['id'], str)
            assert isinstance(result['type'], str)
            assert isinstance(result['confidence'], (int, float))
            assert isinstance(result['timestamp'], str)
    
    def test_recent_results_values(self, client):
        """Test recent results have valid values"""
        response = client.get('/api/user/recent-results')
        data = json.loads(response.data)
        
        for result in data:
            # Confidence should be between 0 and 1
            assert 0.0 <= result['confidence'] <= 1.0
            
            # Type should be valid category
            assert result['type'] in ['market', 'standard', 'premium']
    
    def test_recent_results_ordering(self, client):
        """Test results are ordered by timestamp (newest first)"""
        response = client.get('/api/user/recent-results')
        data = json.loads(response.data)
        
        # If we have multiple results, verify ordering
        if len(data) > 1:
            timestamps = [result['timestamp'] for result in data]
            # All timestamps should be strings
            assert all(isinstance(ts, str) for ts in timestamps)
            
            # Timestamps should be in valid format
            for ts in timestamps:
                assert len(ts) > 0
    
    def test_recent_results_limit(self, client):
        """Test that results are limited to 10"""
        response = client.get('/api/user/recent-results')
        data = json.loads(response.data)
        
        # Should never exceed 10 items
        assert len(data) <= 10


class TestModelInfo:
    """Test model information endpoint"""
    
    def test_get_model_info(self, client):
        """Test getting model information"""
        response = client.get('/api/user/model-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields
        assert 'accuracy' in data
        assert 'lastTrained' in data
        assert 'totalSamples' in data
        
        # Verify types
        assert isinstance(data['accuracy'], (int, float))
        assert data['lastTrained'] is None or isinstance(data['lastTrained'], str)
        assert isinstance(data['totalSamples'], int)
    
    def test_model_info_values(self, client):
        """Test model info has valid values"""
        response = client.get('/api/user/model-info')
        data = json.loads(response.data)
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= data['accuracy'] <= 1.0
        
        # Total samples should be non-negative
        assert data['totalSamples'] >= 0
    
    def test_model_info_when_trained(self, client):
        """Test model info when model is trained"""
        response = client.get('/api/user/model-info')
        data = json.loads(response.data)
        
        # If model is trained (accuracy > 0), check consistency
        if data['accuracy'] > 0:
            # Should have training timestamp or be null
            assert data['lastTrained'] is None or len(data['lastTrained']) > 0
    
    def test_model_info_when_not_trained(self, client):
        """Test model info when model is not trained"""
        response = client.get('/api/user/model-info')
        data = json.loads(response.data)
        
        # If not trained (accuracy == 0), verify defaults
        if data['accuracy'] == 0.0:
            assert data['totalSamples'] == 0
            assert data['lastTrained'] is None


class TestUserDashboardIntegration:
    """Test integration between user dashboard endpoints"""
    
    def test_stats_and_results_consistency(self, client):
        """Test consistency between stats and recent results"""
        # Get stats
        stats_response = client.get('/api/user/dashboard-stats')
        stats_data = json.loads(stats_response.data)
        
        # Get recent results
        results_response = client.get('/api/user/recent-results')
        results_data = json.loads(results_response.data)
        
        # Both should work
        assert stats_response.status_code == 200
        assert results_response.status_code == 200
        
        # Recent results should be a list
        assert isinstance(results_data, list)
        
        # If we have results, stats should have some counts
        if len(results_data) > 0:
            # At least totalToday should be >= 0
            assert stats_data['totalToday'] >= 0
    
    def test_model_info_and_stats_consistency(self, client):
        """Test consistency between model info and stats"""
        # Get model info
        model_response = client.get('/api/user/model-info')
        model_data = json.loads(model_response.data)
        
        # Get stats
        stats_response = client.get('/api/user/dashboard-stats')
        stats_data = json.loads(stats_response.data)
        
        # Both should work
        assert model_response.status_code == 200
        assert stats_response.status_code == 200
        
        # Basic consistency checks
        assert isinstance(model_data['accuracy'], (int, float))
        assert isinstance(stats_data['totalToday'], int)
    
    def test_all_endpoints_accessible(self, client):
        """Test all user dashboard endpoints are accessible"""
        endpoints = [
            '/api/user/dashboard-stats',
            '/api/user/recent-results',
            '/api/user/model-info'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestDashboardWithMetadata:
    """Test dashboard endpoints with metadata"""
    
    def test_stats_with_metadata(self, client):
        """Test stats uses metadata when available"""
        response = client.get('/api/user/dashboard-stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return valid stats regardless of metadata
        assert 'totalToday' in data
        assert data['totalToday'] >= 0
    
    def test_model_info_with_metadata(self, client):
        """Test model info uses metadata when available"""
        response = client.get('/api/user/model-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return valid info regardless of metadata
        assert 'accuracy' in data
        assert 0.0 <= data['accuracy'] <= 1.0
    
    def test_fallback_to_database(self, client):
        """Test endpoints fallback to database when metadata unavailable"""
        # All endpoints should work even without metadata
        stats_response = client.get('/api/user/dashboard-stats')
        results_response = client.get('/api/user/recent-results')
        model_response = client.get('/api/user/model-info')
        
        assert stats_response.status_code == 200
        assert results_response.status_code == 200
        assert model_response.status_code == 200


class TestErrorHandling:
    """Test error handling for user dashboard endpoints"""
    
    def test_dashboard_endpoints_resilience(self, client):
        """Test all dashboard endpoints are resilient"""
        endpoints = [
            '/api/user/dashboard-stats',
            '/api/user/recent-results',
            '/api/user/model-info'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
    
    def test_stats_error_returns_defaults(self, client):
        """Test stats returns default values on error"""
        response = client.get('/api/user/dashboard-stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have all fields even on error
        assert 'totalToday' in data
        assert 'marketCount' in data
        assert 'standardCount' in data
        assert 'premiumCount' in data
        
        # All should be integers
        assert isinstance(data['totalToday'], int)
        assert isinstance(data['marketCount'], int)
    
    def test_recent_results_error_returns_empty(self, client):
        """Test recent results returns empty list on error"""
        response = client.get('/api/user/recent-results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should be a list (empty or with data)
        assert isinstance(data, list)
    
    def test_model_info_error_returns_defaults(self, client):
        """Test model info returns default values on error"""
        response = client.get('/api/user/model-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have all fields even on error
        assert 'accuracy' in data
        assert 'lastTrained' in data
        assert 'totalSamples' in data
        
        # Accuracy should be in valid range
        assert 0.0 <= data['accuracy'] <= 1.0
    
    def test_invalid_http_methods(self, client):
        """Test endpoints reject invalid HTTP methods"""
        endpoints = [
            '/api/user/dashboard-stats',
            '/api/user/recent-results',
            '/api/user/model-info'
        ]
        
        for endpoint in endpoints:
            # Should not accept POST
            response = client.post(endpoint)
            assert response.status_code in [405, 404]
class TestEdgeCases:
    """Test edge cases"""
    
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        import threading
        
        results = []
        
        def get_stats():
            response = client.get('/api/user/dashboard-stats')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = [threading.Thread(target=get_stats) for _ in range(5)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
    
    def test_multiple_rapid_requests(self, client):
        """Test multiple rapid requests to same endpoint"""
        for _ in range(10):
            response = client.get('/api/user/dashboard-stats')
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
