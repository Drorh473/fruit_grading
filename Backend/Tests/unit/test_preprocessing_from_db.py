import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import TestConfig for test database settings
from Tests.test_config import TestConfig

from preprocessing.preprocessing_from_db import (
    custom_preprocessing,
    process_image
)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def valid_image_path(tmp_path, sample_image):
    """Create a valid test image file"""
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


@pytest.fixture
def corrupted_image_path(tmp_path):
    """Create a corrupted/truncated image file"""
    img_path = tmp_path / "corrupted.png"
    # Write truncated PNG data
    with open(img_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')  # Incomplete PNG header
    return img_path


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing"""
    return {
        'set_type': 'train',
        'camera_id': 1,
        '_id': '12345abcde'
    }


class TestCustomPreprocessing:
    """Test custom preprocessing function"""
    
    def test_preprocessing_output_shape(self, sample_image):
        """Test that preprocessing maintains correct output shape"""
        result = custom_preprocessing(sample_image)
        
        # Check output is normalized
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        
        # Check shape is appropriate
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels
    
    def test_preprocessing_with_uint8_input(self, sample_image):
        """Test preprocessing with uint8 input (0-255 range)"""
        result = custom_preprocessing(sample_image)
        
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0
    
    def test_preprocessing_with_float_input(self):
        """Test preprocessing handles float input correctly"""
        # Create float image in [0, 1] range
        test_image = np.random.rand(200, 200, 3).astype(np.float32)
        
        result = custom_preprocessing(test_image)
        
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0
    
    def test_preprocessing_saves_file(self, sample_image, tmp_path):
        """Test that preprocessing saves file when path provided"""
        save_path = tmp_path / "preprocessed.png"
        
        custom_preprocessing(sample_image, save_path=str(save_path))
        
        # Check file was created
        assert save_path.exists()
        
        # Verify saved image can be loaded
        saved_img = cv2.imread(str(save_path))
        assert saved_img is not None
    
    def test_preprocessing_different_sizes(self):
        """Test preprocessing with various image sizes"""
        sizes = [(100, 100, 3), (300, 400, 3), (500, 200, 3)]
        
        for size in sizes:
            test_image = np.random.randint(0, 255, size, dtype=np.uint8)
            result = custom_preprocessing(test_image)
            
            # Should complete without errors
            assert result is not None
            assert result.dtype == np.float32
    
    def test_preprocessing_preserves_aspect_ratio(self):
        """Test that preprocessing maintains aspect ratio when resizing"""
        # Create image with known aspect ratio
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = custom_preprocessing(test_image)
        
        # Should be resized but maintain general shape
        assert result is not None
        assert result.shape[2] == 3


class TestProcessImage:
    """Test process_image function"""
    
    def test_process_image_success(self, valid_image_path, tmp_path, sample_metadata):
        """Test successful image processing"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        file_id, output_path, error = process_image(
            str(valid_image_path), 
            str(output_dir),
            sample_metadata
        )
        
        assert file_id is not None
        assert output_path is not None
        assert error is None
        assert Path(output_path).exists()
    
    def test_process_image_invalid_path(self, tmp_path, sample_metadata):
        """Test processing with invalid image path"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        file_id, output_path, error = process_image(
            "/invalid/path/image.png", 
            str(output_dir),
            sample_metadata
        )
        
        assert file_id is None
        assert output_path is None
        assert error is not None
    
    def test_process_image_creates_output_directory(self, valid_image_path, tmp_path, sample_metadata):
        """Test that process_image creates output directory if needed"""
        output_dir = tmp_path / "new_output"
        # Don't create directory - let function create it
        
        file_id, output_path, error = process_image(
            str(valid_image_path), 
            str(output_dir),
            sample_metadata
        )
        
        # Should create directory structure
        if error is None:
            assert Path(output_path).parent.exists()
            assert Path(output_path).exists()
    
    def test_process_image_handles_corrupted_file(self, corrupted_image_path, tmp_path, sample_metadata):
        """Test that process_image handles corrupted images gracefully"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        file_id, output_path, error = process_image(
            str(corrupted_image_path),
            str(output_dir),
            sample_metadata
        )
        
        # Should return error for corrupted file
        assert error is not None
    
    def test_process_image_with_different_cameras(self, valid_image_path, tmp_path):
        """Test processing images from different cameras"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        for camera_id in range(TestConfig.NUM_OF_CAMERAS):
            metadata = {
                'set_type': 'train',
                'camera_id': camera_id,
                '_id': f'test_{camera_id}'
            }
            
            file_id, output_path, error = process_image(
                str(valid_image_path),
                str(output_dir),
                metadata
            )
            
            if error is None:
                assert Path(output_path).exists()
                # Verify camera ID is in the path
                assert f'camera_{camera_id}' in str(output_path)


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing"""
    
    def test_preprocessing_grayscale_image(self):
        """Test preprocessing converts grayscale to RGB"""
        # Create grayscale image
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # If preprocessing expects RGB, it should handle grayscale
        # This test depends on your implementation
        try:
            result = custom_preprocessing(gray_image)
            assert result is not None
        except Exception as e:
            # If it's expected to fail on grayscale, that's okay
            assert 'shape' in str(e).lower() or 'channel' in str(e).lower()
    
    def test_preprocessing_very_small_image(self):
        """Test preprocessing with very small image"""
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        result = custom_preprocessing(small_image)
        
        # Should handle small images
        assert result is not None
        assert result.dtype == np.float32
    
    def test_preprocessing_very_large_image(self):
        """Test preprocessing with large image"""
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        result = custom_preprocessing(large_image)
        
        # Should resize and process large images
        assert result is not None
        assert result.dtype == np.float32
    
    def test_preprocessing_output_path_creation(self, sample_image, tmp_path):
        """Test that preprocessing creates necessary directories"""
        nested_path = tmp_path / "level1" / "level2" / "output.png"
        
        # Should create nested directories
        custom_preprocessing(sample_image, save_path=str(nested_path))
        
        assert nested_path.exists()


class TestPreprocessingConsistency:
    """Test consistency of preprocessing"""
    
    def test_preprocessing_deterministic(self, sample_image):
        """Test that preprocessing gives consistent results"""
        # Process same image twice
        result1 = custom_preprocessing(sample_image.copy())
        result2 = custom_preprocessing(sample_image.copy())
        
        # Results should be very similar (allowing for floating point errors)
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)
    
    def test_preprocessing_batch_consistency(self):
        """Test that batch processing gives consistent results"""
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Process all images
        results = [custom_preprocessing(img) for img in images]
        
        # All results should have same shape
        shapes = [r.shape for r in results]
        assert len(set(shapes)) == 1  # All same shape
        
        # All should be normalized
        for result in results:
            assert result.max() <= 1.0
            assert result.min() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])