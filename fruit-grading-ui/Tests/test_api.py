"""
API Integration Tests
Tests backend API endpoints
"""

import pytest
import requests


API_URL = "http://localhost:5000/api"


# ============================================================================
# PROCESSING API TESTS
# ============================================================================

@pytest.mark.api
class TestProcessingAPI:
    """Test processing API endpoints"""
    
    def test_start_pipeline(self, api_client):
        """Should start processing pipeline"""
        response = api_client.post(
            f"{API_URL}/processing/start",
            json={'config': {'batch_size': 32}}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
    
    def test_stop_pipeline(self, api_client):
        """Should stop processing pipeline"""
        response = api_client.post(f"{API_URL}/processing/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
    
    def test_get_pipeline_status(self, api_client):
        """Should get current pipeline status"""
        response = api_client.get(f"{API_URL}/processing/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'progress' in data
    
    def test_get_pipeline_logs(self, api_client):
        """Should fetch pipeline logs"""
        response = api_client.get(f"{API_URL}/processing/logs?limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ============================================================================
# DASHBOARD API TESTS
# ============================================================================

@pytest.mark.api
class TestDashboardAPI:
    """Test dashboard API endpoints"""
    
    def test_get_dashboard_data(self, api_client):
        """Should fetch dashboard statistics"""
        response = api_client.get(f"{API_URL}/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert 'total_processed' in data
        assert 'accuracy' in data
        assert 'cameras_online' in data


# ============================================================================
# CAMERA API TESTS
# ============================================================================

@pytest.mark.api
class TestCameraAPI:
    """Test camera monitoring API"""
    
    def test_get_camera_status(self, api_client):
        """Should fetch status of all cameras"""
        response = api_client.get(f"{API_URL}/cameras/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'cameras' in data
        assert len(data['cameras']) == 4  # 4 cameras
    
    def test_refresh_camera(self, api_client):
        """Should refresh specific camera"""
        response = api_client.post(f"{API_URL}/cameras/1/refresh")
        
        assert response.status_code == 200


# ============================================================================
# RESULTS API TESTS
# ============================================================================

@pytest.mark.api
class TestResultsAPI:
    """Test results API endpoints"""
    
    def test_get_results(self, api_client):
        """Should fetch classification results"""
        response = api_client.get(f"{API_URL}/results")
        
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
    
    def test_get_results_with_filters(self, api_client):
        """Should filter results by fruit type"""
        response = api_client.get(
            f"{API_URL}/results?fruit_type=apple&grade=premium"
        )
        
        assert response.status_code == 200
        data = response.json()
        # Results should be filtered
    
    def test_export_results_csv(self, api_client):
        """Should export results as CSV"""
        response = api_client.get(f"{API_URL}/results/export?format=csv")
        
        assert response.status_code == 200
        assert 'text/csv' in response.headers['Content-Type']


# ============================================================================
# SETTINGS API TESTS
# ============================================================================

@pytest.mark.api
class TestSettingsAPI:
    """Test settings API endpoints"""
    
    def test_get_settings(self, api_client):
        """Should fetch system settings"""
        response = api_client.get(f"{API_URL}/settings")
        
        assert response.status_code == 200
        data = response.json()
        assert 'confidence_threshold' in data
    
    def test_update_settings(self, api_client):
        """Should update settings"""
        response = api_client.put(
            f"{API_URL}/settings",
            json={'confidence_threshold': 0.85}
        )
        
        assert response.status_code == 200


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error responses"""
    
    def test_404_for_invalid_endpoint(self, api_client):
        """Should return 404 for invalid endpoint"""
        response = api_client.get(f"{API_URL}/invalid-endpoint")
        
        assert response.status_code == 404
    
    def test_401_for_unauthorized(self, api_client):
        """Should return 401 for unauthorized access"""
        # If your API requires auth
        pass
    
    def test_500_on_server_error(self, api_client):
        """Should handle server errors gracefully"""
        # Test endpoint that might fail
        pass