"""
Add Fruit API Tests
Complete tests for fruit addition endpoints
"""
import pytest
import json
import io
from pathlib import Path


class TestAddFruitAPI:
    """Test fruit addition endpoints"""
    
    def test_add_fruit_success(self, client, valid_image_path):
        """Test successful fruit addition"""
        # Create multipart form data
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0001',
            'category': 'A',
            'camera_0': (open(valid_image_path, 'rb'), 'camera_0.png'),
            'camera_1': (open(valid_image_path, 'rb'), 'camera_1.png'),
            'camera_2': (open(valid_image_path, 'rb'), 'camera_2.png'),
            'camera_3': (open(valid_image_path, 'rb'), 'camera_3.png')
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code in [200, 201]
        response_data = json.loads(response.data)
        assert 'success' in response_data or 'id' in response_data
    
    def test_add_fruit_validation_missing_object_id(self, client, valid_image_path):
        """Test validation of required object_id field"""
        data = {
            'fruit_type': 'market',
            'category': 'A',
            'camera_0': (open(valid_image_path, 'rb'), 'camera_0.png')
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_add_fruit_missing_camera_images(self, client):
        """Test validation when camera images are missing"""
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0001',
            'category': 'A'
            # No camera images
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
   
    def test_add_fruit_invalid_image_format(self, client, tmp_path):
        """Test rejection of invalid image formats"""
        # Create a text file instead of image
        text_file = tmp_path / "not_image.txt"
        text_file.write_text("This is not an image")
        
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0003',
            'category': 'A',
            'camera_0': (open(text_file, 'rb'), 'camera_0.txt')
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should reject invalid format
        assert response.status_code in [400, 415, 422]
    
    def test_add_fruit_large_file(self, client, tmp_path):
        """Test handling of large file uploads"""
        # Create a large file (simulated)
        import numpy as np
        import cv2
        
        # Create very large image
        large_image = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        large_path = tmp_path / "large_image.png"
        cv2.imwrite(str(large_path), large_image)
        
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0004',
            'category': 'A',
            'camera_0': (open(large_path, 'rb'), 'large_image.png')
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Should either accept or reject based on size limits
        assert response.status_code in [200, 201, 413, 422]
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])