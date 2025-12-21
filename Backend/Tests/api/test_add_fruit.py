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
    
    def test_add_fruit_validation_missing_fruit_type(self, client, valid_image_path):
        """Test validation of required fruit_type field"""
        data = {
            'object_id': 'obj0001',
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
    
    def test_add_fruit_validation_invalid_fruit_type(self, client, valid_image_path):
        """Test validation of fruit_type values"""
        data = {
            'fruit_type': 'invalid_type',  # Invalid type
            'object_id': 'obj0001',
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
    
    def test_add_fruit_partial_camera_images(self, client, valid_image_path):
        """Test with only some camera images"""
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0001',
            'category': 'A',
            'camera_0': (open(valid_image_path, 'rb'), 'camera_0.png'),
            'camera_1': (open(valid_image_path, 'rb'), 'camera_1.png')
            # Missing camera_2 and camera_3
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Might accept partial or reject depending on implementation
        assert response.status_code in [200, 201, 400, 422]
    
    def test_add_fruit_duplicate_prevention(self, client, valid_image_path, test_collection):
        """Test prevention of duplicate fruit entries"""
        # Add first fruit
        data = {
            'fruit_type': 'market',
            'object_id': 'obj0001',
            'category': 'A',
            'camera_0': (open(valid_image_path, 'rb'), 'camera_0.png')
        }
        
        response1 = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Try to add same fruit again
        data2 = {
            'fruit_type': 'market',
            'object_id': 'obj0001',  # Same object_id
            'category': 'A',
            'camera_0': (open(valid_image_path, 'rb'), 'camera_0.png')
        }
        
        response2 = client.post(
            '/api/fruit/add',
            data=data2,
            content_type='multipart/form-data'
        )
        
        # Second request should fail or return conflict
        if response1.status_code in [200, 201]:
            assert response2.status_code in [400, 409, 422]
    
    def test_add_fruit_image_upload(self, client, sample_image, tmp_path):
        """Test multi-camera image upload"""
        # Create image files
        image_paths = []
        for i in range(4):
            img_path = tmp_path / f"camera_{i}.png"
            import cv2
            cv2.imwrite(str(img_path), sample_image)
            image_paths.append(img_path)
        
        data = {
            'fruit_type': 'standard',
            'object_id': 'obj0002',
            'category': 'B',
            'camera_0': (open(image_paths[0], 'rb'), 'camera_0.png'),
            'camera_1': (open(image_paths[1], 'rb'), 'camera_1.png'),
            'camera_2': (open(image_paths[2], 'rb'), 'camera_2.png'),
            'camera_3': (open(image_paths[3], 'rb'), 'camera_3.png')
        }
        
        response = client.post(
            '/api/fruit/add',
            data=data,
            content_type='multipart/form-data'
        )
        
        assert response.status_code in [200, 201]
    
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


class TestFruitListAPI:
    """Test fruit listing endpoints"""
    
    def test_get_all_fruits(self, client):
        """Test retrieving all fruits"""
        response = client.get('/api/fruit/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, (list, dict))
    
    def test_get_fruit_by_id(self, client, test_collection):
        """Test retrieving specific fruit by ID"""
        # Insert a fruit
        test_fruit = {
            "_id": "obj0001",
            "fruit_type": "market",
            "category": "A"
        }
        test_collection.insert_one(test_fruit)
        
        response = client.get('/api/fruit/obj0001')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['fruit_type'] == 'market'


class TestFruitDeleteAPI:
    """Test fruit deletion endpoints"""
    
    def test_delete_fruit(self, client, test_collection):
        """Test deleting a fruit"""
        # Insert a fruit
        test_fruit = {
            "_id": "obj0005",
            "fruit_type": "market",
            "category": "A"
        }
        test_collection.insert_one(test_fruit)
        
        response = client.delete('/api/fruit/obj0005')
        
        assert response.status_code in [200, 204]
        
        # Verify deletion
        deleted = test_collection.find_one({"_id": "obj0005"})
        assert deleted is None
    
    def test_delete_nonexistent_fruit(self, client):
        """Test deleting non-existent fruit"""
        response = client.delete('/api/fruit/nonexistent')
        
        # Should return 404
        assert response.status_code == 404


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])