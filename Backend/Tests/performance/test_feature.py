"""
Feature Extraction Performance Tests
Benchmark feature extraction operations
"""
import pytest
import numpy as np
import time
from Tests.test_config import TestConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.pre_trained_feature_map import (
    flatten_features,
    temporal_pooling,
    multi_view_fusion
)


class TestFeatureExtractionPerformance:
    """Benchmark feature extraction operations"""
    
    def test_flatten_features_speed(self, benchmark):
        """Benchmark feature flattening speed"""
        # Create large feature map
        feature_map = {}
        for i in range(100):
            feature_map[f'obj{i:03d}_0'] = [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
            ]
        
        result = benchmark(flatten_features, feature_map)
        
        assert len(result) == 100
    
    def test_temporal_pooling_speed(self, benchmark):
        """Benchmark temporal pooling speed"""
        # Create flattened features with multiple timesteps
        flattened = {}
        for i in range(50):
            for t in range(10):  # 10 timesteps
                key = f'obj{i:03d}_0_t{t}'
                flattened[key] = {
                    'features': np.random.rand(50176),
                    'group_key': f'obj{i:03d}_0'
                }
        
        result = benchmark(temporal_pooling, flattened)
        
        assert len(result) == 50
    
    def test_multi_view_fusion_speed(self, benchmark):
        """Benchmark multi-view fusion speed"""
        # Create pooled features from multiple cameras
        pooled = {}
        for i in range(50):
            for cam in range(4):
                key = f'obj{i:03d}_{cam}'
                pooled[key] = np.random.rand(50176)
        
        result = benchmark(multi_view_fusion, pooled)
        
        assert len(result) == 50


class TestFeatureExtractionThroughput:
    """Test throughput of feature extraction"""
    
    @pytest.mark.slow
    def test_large_batch_processing_throughput(self):
        """Benchmark processing large batch of features"""
        # Create 1000 feature maps
        feature_map = {}
        for i in range(1000):
            feature_map[f'obj{i:04d}_0'] = [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': i % 3}
            ]
        
        start = time.time()
        
        # Process through pipeline
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        elapsed = time.time() - start
        
        throughput = len(feature_map) / elapsed
        
        print(f"Throughput: {throughput:.2f} objects/second")
        
        # Should process > 100 objects per second
        assert throughput > 100, f"Throughput too low: {throughput:.2f} objects/sec"
    
    def test_multi_camera_fusion_throughput(self):
        """Benchmark multi-camera fusion throughput"""
        # Create features for 100 objects, 4 cameras each
        pooled = {}
        for i in range(100):
            for cam in range(4):
                key = f'obj{i:03d}_{cam}'
                pooled[key] = np.random.rand(50176)
        
        start = time.time()
        fused = multi_view_fusion(pooled)
        elapsed = time.time() - start
        
        throughput = len(fused) / elapsed
        
        print(f"Fusion throughput: {throughput:.2f} objects/second")
        
        # Should be very fast (> 1000 objects/sec)
        assert throughput > 1000


class TestFeatureExtractionMemory:
    """Test memory usage of feature extraction"""
    
    @pytest.mark.slow
    def test_memory_usage_large_batch(self):
        """Monitor memory usage for large batch"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Process large batch
        feature_map = {}
        for i in range(500):
            feature_map[f'obj{i:04d}_0'] = [
                {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
            ]
        
        flattened = flatten_features(feature_map)
        pooled = temporal_pooling(flattened)
        fused = multi_view_fusion(pooled)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"Peak memory: {peak_mb:.2f} MB")
        
        # Should use < 500MB for 500 objects
        assert peak_mb < 500, f"Memory usage too high: {peak_mb:.2f} MB"


class TestPipelineOptimization:
    """Test optimizations in the pipeline"""
    
    def test_vectorized_operations(self):
        """Test that vectorized operations are faster than loops"""
        # Create test data
        pooled = {}
        for i in range(100):
            for cam in range(4):
                pooled[f'obj{i:03d}_{cam}'] = np.random.rand(1000)
        
        # Time the fusion
        start = time.time()
        fused = multi_view_fusion(pooled)
        vectorized_time = time.time() - start
        
        print(f"Vectorized fusion: {vectorized_time:.4f}s")
        
        # Should be very fast (< 0.1s for 100 objects)
        assert vectorized_time < 0.1
    
    def test_memory_efficient_processing(self):
        """Test that processing is memory efficient"""
        import tracemalloc
        
        # Process in chunks to test memory efficiency
        tracemalloc.start()
        
        for chunk in range(10):
            feature_map = {}
            for i in range(100):
                obj_id = f'obj{chunk*100 + i:04d}_0'
                feature_map[obj_id] = [
                    {'features': np.random.rand(7, 7, 1024), 'timestamp': 't0', 'label': 0}
                ]
            
            flattened = flatten_features(feature_map)
            pooled = temporal_pooling(flattened)
            fused = multi_view_fusion(pooled)
            
            del feature_map, flattened, pooled, fused
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        # Memory should stay relatively constant (< 200MB)
        assert peak_mb < 200


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])