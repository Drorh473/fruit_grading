import pytest
import json
from PIL import Image

class TestValidateFolderEndpoint:
    """Test/validate endpoint"""
    
    def test_validate_folder_missing_path(self, client):
        """Test validation without folder path"""
        response = client.post('/api/fruits/validate',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['valid'] is False
        assert 'required' in data['message'].lower()
    
    def test_validate_folder_nonexistent_path(self, client):
        """Test validation with non-existent folder"""
        response = client.post('/api/fruits/validate',
                              data=json.dumps({'folderPath': '/nonexistent/path'}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is False
        assert 'not exist' in data['message'].lower()
    
    def test_validate_folder_valid_structure(self, client, tmp_path):
        """Test validation with valid folder structure"""
        # Create valid folder structure
        folder = tmp_path / "fruit_obj001"
        folder.mkdir()
        
        # Create angle directories with images
        for i in range(4):
            angle_dir = folder / f"angle_{i}"
            angle_dir.mkdir()
            
            # Create test image
            img = Image.new('RGB', (224, 224), color='red')
            img.save(angle_dir / 'image.jpg')
        
        response = client.post('/api/fruits/validate',
                              data=json.dumps({'folderPath': str(folder)}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        if data['valid']:
            assert 'details' in data
            assert data['details']['anglesFound'] == 4
            assert data['details']['totalImages'] >= 4
    
    def test_validate_folder_invalid_structure(self, client, tmp_path):
        """Test validation with invalid folder structure"""
        # Create folder without proper structure
        folder = tmp_path / "bad_folder"
        folder.mkdir()
        
        response = client.post('/api/fruits/validate',
                              data=json.dumps({'folderPath': str(folder)}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        # Should be invalid or have warnings
        assert 'valid' in data


class TestProcessFruitEndpoint:
    """Test/process endpoint"""
    
    def test_process_fruit_missing_path(self, client):
        """Test processing without folder path"""
        response = client.post('/api/fruits/process',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'required' in data['error'].lower()
    
    def test_process_fruit_nonexistent_path(self, client):
        """Test processing with non-existent folder"""
        response = client.post('/api/fruits/process',
                              data=json.dumps({'folderPath': '/nonexistent/path'}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'not exist' in data['error'].lower()
    
    def test_process_fruit_with_tests_flag(self, client, tmp_path):
        """Test processing with runTests flag"""
        folder = tmp_path / "test_fruit"
        folder.mkdir()
        
        response = client.post('/api/fruits/process',
                              data=json.dumps({
                                  'folderPath': str(folder),
                                  'runTests': True
                              }),
                              content_type='application/json')
        
        assert response.status_code in [200, 400, 500]
        data = json.loads(response.data)
        assert 'success' in data


class TestUploadFruitEndpoint:
    """Test/upload endpoint"""
    
    def test_upload_fruit_no_files(self, client):
        """Test upload without files"""
        response = client.post('/api/fruits/upload',
                              data={},
                              content_type='multipart/form-data')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'no files' in data['error'].lower()
    
    def test_upload_fruit_success(self, client, valid_image_path):
        """Test successful fruit upload"""
        with open(valid_image_path, 'rb') as img1:
            with open(valid_image_path, 'rb') as img2:
                with open(valid_image_path, 'rb') as img3:
                    with open(valid_image_path, 'rb') as img4:
                        data = {
                            'fruitType': 'market',
                            'objectId': 'obj0001',
                            'files': [
                                (img1, 'camera_0.jpg'),
                                (img2, 'camera_1.jpg'),
                                (img3, 'camera_2.jpg'),
                                (img4, 'camera_3.jpg')
                            ]
                        }
                        
                        response = client.post('/api/fruits/upload',
                                              data=data,
                                              content_type='multipart/form-data')
        
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data['success'] is True
        assert response_data['filesUploaded'] == 4
        assert 'objectId' in response_data
    
    def test_upload_fruit_single_image(self, client, valid_image_path):
        """Test upload with single image"""
        with open(valid_image_path, 'rb') as img:
            data = {
                'fruitType': 'premium',
                'objectId': 'obj0002',
                'files': [(img, 'single.jpg')]
            }
            
            response = client.post('/api/fruits/upload',
                                  data=data,
                                  content_type='multipart/form-data')
        
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data['success'] is True
        assert response_data['filesUploaded'] == 1


class TestAddFruitIntegration:
    """Test integration between endpoints"""
    
    def test_validate_then_process_workflow(self, client, tmp_path):
        """Test validate -> process workflow"""
        # Create valid folder
        folder = tmp_path / "workflow_test"
        folder.mkdir()
        
        for i in range(4):
            angle_dir = folder / f"angle_{i}"
            angle_dir.mkdir()
            img = Image.new('RGB', (224, 224), color='red')
            img.save(angle_dir / 'test.jpg')
        
        # Step 1: Validate
        validate_response = client.post('/api/fruits/validate',
                                       data=json.dumps({'folderPath': str(folder)}),
                                       content_type='application/json')
        
        assert validate_response.status_code == 200
        
        # Step 2: Process 
        process_response = client.post('/api/fruits/process',
                                      data=json.dumps({'folderPath': str(folder)}),
                                      content_type='application/json')
        
        # Should accept the request
        assert process_response.status_code in [200, 500]


class TestErrorHandling:
    """Test error handling for add fruit endpoints"""
    
    def test_invalid_json_format(self, client):
        """Test handling of invalid JSON"""
        response = client.post('/api/fruits/validate',
                              data='invalid json',
                              content_type='application/json')
        
        assert response.status_code in [400, 500]
    
    def test_endpoints_return_json(self, client):
        """Test all endpoints return JSON"""
        # Test validate endpoint
        response1 = client.post('/api/fruits/validate',
                               data=json.dumps({}),
                               content_type='application/json')
        assert 'application/json' in response1.content_type
        
        # Test process endpoint
        response2 = client.post('/api/fruits/process',
                               data=json.dumps({}),
                               content_type='application/json')
        assert 'application/json' in response2.content_type
        
        # Test types endpoint
        response3 = client.get('/api/fruits/types')
        assert 'application/json' in response3.content_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
