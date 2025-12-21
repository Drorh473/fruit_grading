
import pytest
import numpy as np
import time
from Tests.test_config import TestConfig
from preprocessing.preprocessing_from_db import custom_preprocessing


class TestPreprocessingPerformance:
    """Benchmark preprocessing operations"""
    
    def test_single_image_preprocessing_speed(self, sample_image, benchmark):
        """Benchmark: < 100ms per image"""
        result = benchmark(custom_preprocessing, sample_image)
        
        assert result is not None
        # Benchmark automatically measures time
    
    def test_batch_preprocessing_throughput(self, tmp_path):
        """Benchmark: > 10 images/second"""
        import cv2
        
        # Create 100 test images
        images = []
        for i in range(100):
            img_path = tmp_path / f"test_{i}.png"
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            images.append(img)
        
        # Time batch processing
        start = time.time()
        for img in images:
            custom_preprocessing(img)
        elapsed = time.time() - start
        
        throughput = len(images) / elapsed
        
        print(f"Throughput: {throughput:.2f} images/second")
        assert throughput > 10, f"Throughput too low: {throughput:.2f} images/sec"
    
    @pytest.mark.slow
    def test_memory_usage_during_batch_processing(self):
        """Benchmark: < 2GB for 1000 images"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Process 1000 images
        for i in range(1000):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            result = custom_preprocessing(img)
            del result
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"Peak memory: {peak_mb:.2f} MB")
        
        assert peak_mb < 2048, f"Memory usage too high: {peak_mb:.2f} MB"
