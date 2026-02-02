import pytest
import json
import io
import sys
from PIL import Image
from unittest.mock import patch

# -------------------------------------------------------------------------
# TARGET PATCH PATH
# Based on your file path: Backend/routes/add_fruit.py
# -------------------------------------------------------------------------
# We try two likely variations to be safe:
# 1. 'routes.add_fruit.process_new_fruit_folder' (if running from Backend/)
# 2. 'backend.routes.add_fruit.process_new_fruit_folder' (if running from parent)

# We will use a try-except block in the test setup if needed, but for now, 
# 'routes.add_fruit.process_new_fruit_folder' is the most probable correct path
# because your previous errors showed `backend` as the rootdir.

TARGET_PATCH = 'routes.add_fruit.process_new_fruit_folder'

class TestUploadProcessEndpoint:
    """Test /upload-process endpoint with MOCKED backend."""

    def test_upload_process_no_files(self, client):
        response = client.post('/api/fruit/upload-process',
                             data={},
                             content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False

    def test_upload_process_empty_file_list(self, client):
        response = client.post('/api/fruit/upload-process',
                             data={'dataset': (io.BytesIO(b''), '')},
                             content_type='multipart/form-data')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False

    def test_upload_process_no_valid_angle_folders(self, client, valid_image_path):
        with open(valid_image_path, 'rb') as img:
            data = {'dataset': (img, 'test_image.jpg')}
            response = client.post('/api/fruit/upload-process',
                                 data=data,
                                 content_type='multipart/form-data')
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'no valid images' in data['error'].lower()

    def test_upload_process_with_an_folders(self, client, tmp_path):
        # MOCK THE PROCESSOR
        with patch(TARGET_PATCH) as mock_process:
            mock_process.return_value = {
                'object_id': 'mock_123',
                'predicted_type': 'premium',
                'confidence': 0.99,
                'images_count': 4,
                'processing_time': 0.1
            }

            # Create test images
            image_files = []
            for i in range(4):
                img = Image.new('RGB', (224, 224), color='red')
                img_path = tmp_path / f'an_{i}' / 'test.jpg'
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)
                image_files.append((f'root/an_{i}/test.jpg', img_path))

            file_handles = []
            try:
                file_list = []
                for path, img_path in image_files:
                    f = open(img_path, 'rb')
                    file_handles.append(f)
                    file_list.append((f, path))

                response = client.post('/api/fruit/upload-process',
                                     data={'dataset': file_list},
                                     content_type='multipart/form-data')

                # DEBUG: Print error if it fails
                if response.status_code != 200:
                    print(f"DEBUG ERROR: {response.data}")

                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['success'] is True
                assert data['objectId'] == 'mock_123'

            finally:
                for f in file_handles:
                    f.close()

    def test_upload_process_with_angle_folders(self, client, tmp_path):
        with patch(TARGET_PATCH) as mock_process:
            mock_process.return_value = {
                'object_id': 'mock_456',
                'predicted_type': 'standard',
                'confidence': 0.85,
                'images_count': 4,
                'processing_time': 0.1
            }

            image_files = []
            for i in range(4):
                img = Image.new('RGB', (224, 224), color='blue')
                img_path = tmp_path / f'angle_{i}' / 'image.jpg'
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)
                image_files.append((f'test_obj/angle_{i}/image.jpg', img_path))

            file_handles = []
            try:
                file_list = []
                for path, img_path in image_files:
                    f = open(img_path, 'rb')
                    file_handles.append(f)
                    file_list.append((f, path))

                response = client.post('/api/fruit/upload-process',
                                     data={'dataset': file_list},
                                     content_type='multipart/form-data')

                assert response.status_code == 200

            finally:
                for f in file_handles:
                    f.close()

    def test_upload_process_non_image_files_ignored(self, client, tmp_path):
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


class TestUploadProcessValidation:

    def test_valid_extensions_accepted(self, client, tmp_path):
        with patch(TARGET_PATCH) as mock_process:
            mock_process.return_value = {'object_id': 'id', 'predicted_type': 'p', 'confidence': 1.0}

            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in extensions:
                img = Image.new('RGB', (224, 224), color='green')
                img_path = tmp_path / f'test{ext}'
                img.save(img_path)

                with open(img_path, 'rb') as f:
                    response = client.post('/api/fruit/upload-process',
                                         data={'dataset': (f, f'obj/an_0/test{ext}')},
                                         content_type='multipart/form-data')

                assert response.status_code == 200, f"Failed for extension {ext}"

    def test_multiple_images_per_angle(self, client, tmp_path):
        with patch(TARGET_PATCH) as mock_process:
            mock_process.return_value = {
                'object_id': 'mock_multi',
                'predicted_type': 'market',
                'confidence': 0.99,
                'images_count': 12,
                'processing_time': 0.5
            }

            file_handles = []
            file_list = []
            try:
                for angle in range(4):
                    for img_num in range(3):
                        img = Image.new('RGB', (224, 224), color='yellow')
                        img_path = tmp_path / f'an_{angle}_{img_num}.jpg'
                        img.save(img_path)
                        f = open(img_path, 'rb')
                        file_handles.append(f)
                        file_list.append((f, f'obj/an_{angle}/image_{img_num}.jpg'))

                response = client.post('/api/fruit/upload-process',
                                     data={'dataset': file_list},
                                     content_type='multipart/form-data')

                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['imagesProcessed'] == 12 

            finally:
                for f in file_handles:
                    f.close()


class TestUploadProcessResponse:

    def test_success_response_format(self, client, tmp_path):
        with patch(TARGET_PATCH) as mock_process:
            mock_process.return_value = {
                'object_id': 'id_789',
                'predicted_type': 'premium',
                'confidence': 0.88,
                'images_count': 4,
                'processing_time': 0.2
            }

            file_handles = []
            file_list = []
            try:
                for i in range(4):
                    img = Image.new('RGB', (224, 224), color='purple')
                    img_path = tmp_path / f'test_{i}.jpg'
                    img.save(img_path)
                    f = open(img_path, 'rb')
                    file_handles.append(f)
                    file_list.append((f, f'obj/an_{i}/test.jpg'))

                response = client.post('/api/fruit/upload-process',
                                     data={'dataset': file_list},
                                     content_type='multipart/form-data')

                data = json.loads(response.data)
                assert response.status_code == 200
                assert data['success'] is True
                assert 'objectId' in data

            finally:
                for f in file_handles:
                    f.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
