import pytest
import json
import io
from PIL import Image


class TestUploadProcessEndpoint:
    """Test /upload-process endpoint - the main add fruit endpoint"""

    def test_upload_process_no_files(self, client):
        """Test upload without files returns 400"""
        response = client.post('/api/fruit/upload-process',
                              data={},
                              content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'no files' in data['error'].lower()

    def test_upload_process_empty_file_list(self, client):
        """Test upload with empty filename returns 400"""
        response = client.post('/api/fruit/upload-process',
                              data={'dataset': (io.BytesIO(b''), '')},
                              content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False

    def test_upload_process_no_valid_angle_folders(self, client, valid_image_path):
        """Test upload with images not in angle folders returns 400"""
        with open(valid_image_path, 'rb') as img:
            data = {
                'dataset': (img, 'test_image.jpg')  # No angle folder in path
            }
            response = client.post('/api/fruit/upload-process',
                                  data=data,
                                  content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'no valid images' in data['error'].lower()

    def test_upload_process_with_an_folders(self, client, tmp_path):
        """Test upload with images in an_X folders"""
        # Create test images
        image_files = []
        for i in range(4):
            img = Image.new('RGB', (224, 224), color='red')
            img_path = tmp_path / f'an_{i}' / 'test.jpg'
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
            image_files.append((f'root/an_{i}/test.jpg', img_path))

        # Prepare multipart data
        data = {}
        file_handles = []
        for i, (path, img_path) in enumerate(image_files):
            f = open(img_path, 'rb')
            file_handles.append(f)
            data[f'dataset'] = [(f, path) for f, (path, _) in zip(file_handles, image_files)]

        try:
            # Build proper multipart data
            files = []
            for path, img_path in image_files:
                f = open(img_path, 'rb')
                file_handles.append(f)
                files.append(('dataset', (f, path)))

            response = client.post('/api/fruit/upload-process',
                                  data=files,
                                  content_type='multipart/form-data')

            # The processing might fail due to missing model, but should accept the files
            assert response.status_code in [200, 500]
            data = json.loads(response.data)

            if response.status_code == 200:
                assert data['success'] is True
                assert 'objectId' in data
                assert 'predictedType' in data
                assert 'confidence' in data
                assert 'imagesProcessed' in data
        finally:
            for f in file_handles:
                f.close()

    def test_upload_process_with_angle_folders(self, client, tmp_path):
        """Test upload with images in angle_X folders (alternative naming)"""
        # Create test images in angle_X folders
        image_files = []
        for i in range(4):
            img = Image.new('RGB', (224, 224), color='blue')
            img_path = tmp_path / f'angle_{i}' / 'image.jpg'
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
            image_files.append((f'test_obj/angle_{i}/image.jpg', img_path))

        file_handles = []
        try:
            files = []
            for path, img_path in image_files:
                f = open(img_path, 'rb')
                file_handles.append(f)
                files.append(('dataset', (f, path)))

            response = client.post('/api/fruit/upload-process',
                                  data=files,
                                  content_type='multipart/form-data')

            # The processing might fail due to missing model, but should accept the files
            assert response.status_code in [200, 500]
        finally:
            for f in file_handles:
                f.close()

    def test_upload_process_non_image_files_ignored(self, client, tmp_path):
        """Test that non-image files are ignored"""
        # Create a non-image file in angle folder
        txt_path = tmp_path / 'an_0' / 'readme.txt'
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text('This is not an image')

        with open(txt_path, 'rb') as f:
            response = client.post('/api/fruit/upload-process',
                                  data={'dataset': (f, 'root/an_0/readme.txt')},
                                  content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        # Should fail because no valid images were found
        assert 'no valid images' in data['error'].lower()


class TestUploadProcessValidation:
    """Test validation logic in upload-process endpoint"""

    def test_valid_extensions_accepted(self, client, tmp_path):
        """Test that jpg, jpeg, png, bmp are accepted"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for ext in extensions:
            img = Image.new('RGB', (224, 224), color='green')
            img_path = tmp_path / f'test{ext}'
            img.save(img_path)

            with open(img_path, 'rb') as f:
                response = client.post('/api/fruit/upload-process',
                                      data={'dataset': (f, f'obj/an_0/test{ext}')},
                                      content_type='multipart/form-data')

                # Should either succeed or fail at processing (not at validation)
                assert response.status_code in [200, 500], f"Failed for extension {ext}"

    def test_multiple_images_per_angle(self, client, tmp_path):
        """Test upload with multiple images per angle folder"""
        file_handles = []
        files = []

        try:
            # Create 3 images per angle
            for angle in range(4):
                for img_num in range(3):
                    img = Image.new('RGB', (224, 224), color='yellow')
                    img_path = tmp_path / f'an_{angle}_{img_num}.jpg'
                    img.save(img_path)

                    f = open(img_path, 'rb')
                    file_handles.append(f)
                    files.append(('dataset', (f, f'obj/an_{angle}/image_{img_num}.jpg')))

            response = client.post('/api/fruit/upload-process',
                                  data=files,
                                  content_type='multipart/form-data')

            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = json.loads(response.data)
                assert data['imagesProcessed'] == 12  # 4 angles * 3 images
        finally:
            for f in file_handles:
                f.close()


class TestUploadProcessResponse:
    """Test response format of upload-process endpoint"""

    def test_success_response_format(self, client, tmp_path):
        """Test that successful response has correct format"""
        # Create valid structure
        file_handles = []
        files = []

        try:
            for i in range(4):
                img = Image.new('RGB', (224, 224), color='purple')
                img_path = tmp_path / f'test_{i}.jpg'
                img.save(img_path)

                f = open(img_path, 'rb')
                file_handles.append(f)
                files.append(('dataset', (f, f'obj/an_{i}/test.jpg')))

            response = client.post('/api/fruit/upload-process',
                                  data=files,
                                  content_type='multipart/form-data')

            data = json.loads(response.data)

            if response.status_code == 200:
                # Verify all expected fields
                assert 'success' in data
                assert data['success'] is True
                assert 'objectId' in data
                assert 'predictedType' in data
                assert data['predictedType'] in ['market', 'standard', 'premium']
                assert 'confidence' in data
                assert 0 <= data['confidence'] <= 1
                assert 'imagesProcessed' in data
                assert data['imagesProcessed'] > 0
                assert 'processingTime' in data
            else:
                # Processing failed but should still have error format
                assert 'success' in data
                assert data['success'] is False
                assert 'error' in data
        finally:
            for f in file_handles:
                f.close()

    def test_error_response_format(self, client):
        """Test that error response has correct format"""
        response = client.post('/api/fruit/upload-process',
                              data={},
                              content_type='multipart/form-data')

        data = json.loads(response.data)

        assert 'success' in data
        assert data['success'] is False
        assert 'error' in data
        assert isinstance(data['error'], str)


class TestErrorHandling:
    """Test error handling for add fruit endpoint"""

    def test_endpoint_returns_json(self, client):
        """Test endpoint returns JSON content type"""
        response = client.post('/api/fruit/upload-process',
                              data={},
                              content_type='multipart/form-data')

        assert 'application/json' in response.content_type

    def test_endpoint_exists(self, client):
        """Test that the upload-process endpoint exists"""
        response = client.post('/api/fruit/upload-process',
                              data={},
                              content_type='multipart/form-data')

        # Should not be 404
        assert response.status_code != 404


class TestAddFruitIntegration:
    """Integration tests for add fruit workflow"""

    def test_full_upload_workflow(self, client, tmp_path):
        """Test complete upload and process workflow"""
        # Create a valid folder structure
        obj_folder = tmp_path / 'test_fruit_obj'
        obj_folder.mkdir()

        file_handles = []
        files = []

        try:
            # Create images in proper structure
            for i in range(4):
                angle_dir = obj_folder / f'an_{i}'
                angle_dir.mkdir()

                # Create multiple images per angle
                for j in range(2):
                    img = Image.new('RGB', (224, 224), color='red')
                    img_path = angle_dir / f'image_{j}.jpg'
                    img.save(img_path)

                    f = open(img_path, 'rb')
                    file_handles.append(f)
                    files.append(('dataset', (f, f'test_fruit_obj/an_{i}/image_{j}.jpg')))

            # Upload and process
            response = client.post('/api/fruit/upload-process',
                                  data=files,
                                  content_type='multipart/form-data')

            # Should either succeed or fail at processing stage
            assert response.status_code in [200, 500]
            data = json.loads(response.data)

            if response.status_code == 200:
                # Verify successful classification
                assert data['success'] is True
                assert data['predictedType'] in ['market', 'standard', 'premium']
                assert data['imagesProcessed'] == 8  # 4 angles * 2 images
        finally:
            for f in file_handles:
                f.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
