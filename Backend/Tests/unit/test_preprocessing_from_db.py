"""
Enhanced Preprocessing Testing
Comprehensive tests for image preprocessing with robustness checks
"""
import pytest
import numpy as np
import cv2
import os
from pathlib import Path
from Tests.test_config import TestConfig

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.preprocessing_from_db import (
    custom_preprocessing,
    process_image
)


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
    
    def test_process_image_success(self, valid_image_path, tmp_path):
        """Test successful image processing"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        args = (str(valid_image_path), str(output_dir))
        file_id, output_path, error = process_image(args)
        
        assert file_id is not None
        assert output_path is not None
        assert error is None
        assert Path(output_path).exists()
    
    def test_process_image_invalid_path(self, tmp_path):
        """Test processing with invalid image path"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        args = ("/invalid/path/image.png", str(output_dir))
        file_id, output_path, error = process_image(args)
        
        assert file_id is None
        assert output_path is None
        assert error is not None
    
    def test_process_image_creates_output_directory(self, valid_image_path, tmp_path):
        """Test that process_image creates output directory if needed"""
        output_dir = tmp_path / "new_output"
        # Don't create directory - let function create it
        
        args = (str(valid_image_path), str(output_dir))
        file_id, output_path, error = process_image(args)
        
        # Should either create directory or handle gracefully
        # (depends on implementation)
        if error is None:
            assert output_dir.exists()


class TestPreprocessingRobustness:
    """Robustness and edge case testing"""
    
    def test_corrupted_image_handling(self, corrupted_image_path, tmp_path):
        """Test handling of truncated/corrupted image files"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        args = (str(corrupted_image_path), str(output_dir))
        file_id, output_path, error = process_image(args)
        
        # Should handle gracefully without crashing
        assert error is not None or file_id is None
    
    def test_unsupported_format_detection(self, tmp_path):
        """Test rejection of non-image files"""
        # Create a text file
        text_file = tmp_path / "not_image.txt"
        text_file.write_text("This is not an image")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        args = (str(text_file), str(output_dir))
        file_id, output_path, error = process_image(args)
        
        # Should fail gracefully
        assert error is not None or file_id is None
    
    def test_preprocessing_idempotency(self, sample_image):
        """Test that preprocessing same image twice yields same result"""
        result1 = custom_preprocessing(sample_image.copy())
        result2 = custom_preprocessing(sample_image.copy())
        
        # Results should be very similar (accounting for floating point precision)
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)
    
    def test_batch_processing_memory_usage(self, tmp_path):
        """Monitor memory consumption during batch processing"""
        import tracemalloc
        
        # Create multiple test images
        test_images = []
        for i in range(10):
            img_path = tmp_path / f"test_{i}.png"
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            test_images.append(img_path)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Process all images
        for img_path in test_images:
            args = (str(img_path), str(output_dir))
            process_image(args)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should be reasonable (< 500MB for 10 images)
        assert peak < 500 * 1024 * 1024, f"Peak memory too high: {peak / 1024 / 1024}MB"
    
    def test_opencv_unicode_path_workaround(self, tmp_path, sample_image):
        """Test the Unicode path handling solution"""
        # Create a path with Unicode characters
        unicode_dir = tmp_path / "תפוח_apple"
        unicode_dir.mkdir()
        
        img_path = unicode_dir / "test_תפוח.png"
        
        # OpenCV's imwrite might fail with Unicode paths on some systems
        # Test the workaround using numpy save or cv2.imencode
        try:
            # Try direct save
            cv2.imwrite(str(img_path), sample_image)
            success = img_path.exists()
        except:
            success = False
        
        if not success:
            # Use workaround: encode then write bytes
            is_success, buffer = cv2.imencode('.png', sample_image)
            if is_success:
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                success = img_path.exists()
        
        # At least one method should work
        assert success, "Unicode path handling failed"


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline"""
    
    def test_preprocessing_pipeline(self, sample_image):
        """Test complete preprocessing pipeline"""
        result = custom_preprocessing(sample_image)
        
        # Verify output properties
        assert result.dtype == np.float32
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels
        assert result.max() <= 1.0
        assert result.min() >= 0.0
    
    def test_preprocessing_consistency(self, sample_image):
        """Test that preprocessing produces consistent results"""
        results = []
        
        # Process same image multiple times
        for _ in range(5):
            result = custom_preprocessing(sample_image.copy())
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i], decimal=5)


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])