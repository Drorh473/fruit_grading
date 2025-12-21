import pytest
import json
from Tests.test_config import TestConfig


class TestCameraAPI:
    """Test camera monitoring endpoints"""
    
    def test_get_all_camera_statuses(self, client):
        """Test /api/cameras/status endpoint"""
        response = client.get('/api/cameras/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'cameras' in data
        assert len(data['cameras']) == TestConfig.NUM_OF_CAMERAS
        
        # Verify camera structure
        first_camera = data['cameras'][0]
        assert 'id' in first_camera
        assert 'name' in first_camera
        assert 'status' in first_camera
        assert 'angle' in first_camera
        assert 'fps' in first_camera
    
    def test_get_camera_details(self, client):
        """Test /api/cameras/<id> endpoint"""
        camera_id = 0
        response = client.get(f'/api/cameras/{camera_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['id'] == camera_id
        assert 'name' in data
        assert 'status' in data
        assert 'preprocessing' in data
    
    def test_camera_refresh(self, client):
        """Test /api/cameras/<id>/refresh endpoint"""
        camera_id = 0
        response = client.post(f'/api/cameras/{camera_id}/refresh')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['cameraId'] == camera_id
    
    def test_camera_config(self, client):
        """Test /api/cameras/config endpoint"""
        response = client.get('/api/cameras/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'fps' in data
        assert 'numCameras' in data
        assert data['numCameras'] == TestConfig.NUM_OF_CAMERAS
    
    def test_invalid_camera_id(self, client):
        """Test 404 handling for non-existent camera"""
        response = client.get('/api/cameras/999')
        
        # Should return error or 404
        assert response.status_code in [404, 500]
    
    def test_refresh_all_cameras(self, client):
        """Test /api/cameras/refresh-all endpoint"""
        response = client.post('/api/cameras/refresh-all')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert 'refreshedCount' in data