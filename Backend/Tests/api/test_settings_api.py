import pytest
import json


class TestSettingsAPI:
    """Test settings management endpoints"""
    
    def test_get_settings(self, client):
        """Test /api/settings endpoint"""
        response = client.get('/api/settings')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have settings object
        assert isinstance(data, dict)
    
    def test_update_settings(self, client):
        """Test PUT /api/settings endpoint"""
        payload = {
            "camera_fps": 25,
            "batch_size": 16
        }
        
        response = client.put('/api/settings',
                            data=json.dumps(payload),
                            content_type='application/json')
        
        assert response.status_code in [200, 204]
    
    def test_settings_validation(self, client):
        """Test invalid setting values rejection"""
        payload = {
            "camera_fps": -1,  # Invalid value
            "batch_size": 0     # Invalid value
        }
        
        response = client.put('/api/settings',
                            data=json.dumps(payload),
                            content_type='application/json')
        
        # Should reject invalid values
        assert response.status_code in [400, 422]
