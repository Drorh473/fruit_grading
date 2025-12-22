import pytest
import json


class TestCameraStatus:
    """Test camera status endpoint"""
    
    def test_get_all_camera_statuses(self, client):
        """Test getting all camera statuses with correct structure"""
        response = client.get('/api/cameras/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify structure
        assert 'cameras' in data
        assert isinstance(data['cameras'], list)
        assert len(data['cameras']) == 4
        
        # Verify first camera structure
        camera = data['cameras'][0]
        assert 'id' in camera
        assert 'name' in camera
        assert 'status' in camera
        assert 'angle' in camera
        assert 'fps' in camera
        assert 'resolution' in camera
        assert 'health' in camera
    
    def test_camera_angles_correct(self, client):
        """Test camera angles are assigned correctly"""
        response = client.get('/api/cameras/status')
        data = json.loads(response.data)
        
        expected_angles = ['Front View', 'Right View', 'Back View', 'Left View']
        
        for i, camera in enumerate(data['cameras']):
            assert camera['angle'] == expected_angles[i]
            assert camera['id'] == i


class TestCameraDetails:
    """Test individual camera details endpoint"""
    
    def test_get_camera_details(self, client):
        """Test getting specific camera details"""
        response = client.get('/api/cameras/0')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all fields
        assert data['id'] == 0
        assert data['name'] == 'Camera 0'
        assert data['angle'] == 'Front View'
        assert 'preprocessing' in data
        assert 'uptime' in data
        assert 'framesProcessed' in data
        
        # Verify values
        assert data['resolution'] == '224x224'
        assert data['fps'] > 0
        assert data['framesProcessed'] >= 0
    
    def test_invalid_camera_id(self, client):
        """Test getting details for non-existent camera"""
        response = client.get('/api/cameras/999')
        assert response.status_code == 404


class TestCameraRefresh:
    """Test camera refresh endpoints"""
    
    def test_refresh_single_camera(self, client):
        """Test refreshing a single camera"""
        response = client.post('/api/cameras/0/refresh')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['cameraId'] == 0
        assert 'message' in data
    
    def test_refresh_all_cameras(self, client):
        """Test refreshing all cameras"""
        response = client.post('/api/cameras/refresh-all')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['success'] == True
        assert data['refreshedCount'] == 4
        assert 'message' in data


class TestCameraConfig:
    """Test camera configuration endpoint"""
    
    def test_get_camera_config(self, client):
        """Test getting camera configuration"""
        response = client.get('/api/cameras/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all fields
        assert data['fps'] == 30
        assert data['numCameras'] == 4
        assert data['imageSize'] == '224x224'
        assert 'preprocessing' in data
        assert len(data['angles']) == 4
        
        # Verify angles
        expected_angles = ['Front View', 'Right View', 'Back View', 'Left View']
        assert data['angles'] == expected_angles


class TestCameraIntegration:
    """Test camera endpoint integration"""
    
    def test_status_and_details_consistency(self, client):
        """Test consistency between status and details endpoints"""
        # Get all statuses
        status_response = client.get('/api/cameras/status')
        status_data = json.loads(status_response.data)
        
        # Compare with individual details
        for camera in status_data['cameras']:
            details_response = client.get(f'/api/cameras/{camera["id"]}')
            details_data = json.loads(details_response.data)
            
            assert camera['id'] == details_data['id']
            assert camera['name'] == details_data['name']
            assert camera['angle'] == details_data['angle']
            assert camera['fps'] == details_data['fps']
    
    def test_config_and_status_consistency(self, client):
        """Test consistency between config and status"""
        config_response = client.get('/api/cameras/config')
        config_data = json.loads(config_response.data)
        
        status_response = client.get('/api/cameras/status')
        status_data = json.loads(status_response.data)
        
        # Number of cameras should match
        assert config_data['numCameras'] == len(status_data['cameras'])
        
        # FPS should match
        for camera in status_data['cameras']:
            assert camera['fps'] == config_data['fps']


class TestErrorHandling:
    """Test error handling for camera endpoints"""
    
    def test_invalid_camera_routes(self, client):
        """Test invalid camera routes return 404"""
        invalid_routes = ['/api/cameras/-1', '/api/cameras/100']
        
        for route in invalid_routes:
            response = client.get(route)
            assert response.status_code == 404
    
    def test_wrong_http_methods(self, client):
        """Test endpoints reject wrong HTTP methods"""
        # Refresh should not accept GET
        response = client.get('/api/cameras/0/refresh')
        assert response.status_code in [405, 404]
        
        # Status should not accept POST
        response = client.post('/api/cameras/status')
        assert response.status_code in [405, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])