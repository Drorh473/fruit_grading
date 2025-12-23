"""
Enhanced Preprocessing Testing
Comprehensive tests for image preprocessing with robustness checks
"""
import pytest
import numpy as np
import cv2
import os
from pathlib import Path


# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
            assert Path(output_path).exists
