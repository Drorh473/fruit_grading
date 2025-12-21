"""
Model Training Performance Tests
Benchmark model training and inference operations
"""
import pytest
import numpy as np
import time
from tests.test_config import TestConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cnn.fully_connected_layer import (
    initialize_parameters,
    forward_pass,
    backward_pass,
    update_parameters,
    train_step,
    predict
)


class TestModelInferencePerformance:
    """Benchmark model inference speed"""
    
    def test_forward_pass_speed(self, benchmark, mock_model_parameters):
        """Benchmark forward pass speed"""
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        
        result = benchmark(forward_pass, X, mock_model_parameters)
        
        assert 'A2' in result
    
    def test_prediction_speed(self, benchmark, mock_model_parameters):
        """Benchmark prediction speed"""
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        
        predictions = benchmark(predict, X, mock_model_parameters)
        
        assert len(predictions) == 100
    
    def test_single_sample_inference_latency(self, mock_model_parameters):
        """Test inference latency for single sample"""
        X = np.random.rand(1, TestConfig.FEATURE_DIM)
        
        start = time.time()
        for _ in range(1000):
            _ = forward_pass(X, mock_model_parameters)
        elapsed = time.time() - start
        
        avg_latency = (elapsed / 1000) * 1000  # Convert to ms
        
        print(f"Average inference latency: {avg_latency:.2f}ms")
        
        # Should be < 1ms per sample
        assert avg_latency < 1.0


class TestTrainingPerformance:
    """Benchmark training operations"""
    
    def test_backward_pass_speed(self, benchmark, mock_model_parameters):
        """Benchmark backward pass speed"""
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 100)
        
        cache = forward_pass(X, mock_model_parameters)
        
        result = benchmark(backward_pass, X, y, cache, mock_model_parameters)
        
        assert 'dW1' in result
    
    def test_training_step_speed(self, benchmark, mock_model_parameters):
        """Benchmark complete training step"""
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 100)
        
        params, loss = benchmark(train_step, X, y, mock_model_parameters, 0.01)
        
        assert loss >= 0
    
    @pytest.mark.slow
    def test_single_epoch_training_time(self):
        """Benchmark single epoch training time"""
        num_samples = 100
        X = np.random.rand(num_samples, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, num_samples)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        start = time.time()
        params, loss = train_step(X, y, params, learning_rate=0.01)
        elapsed = time.time() - start
        
        print(f"Single epoch time: {elapsed:.4f}s for {num_samples} samples")
        
        # Should be < 1s for 100 samples
        assert elapsed < 1.0


class TestBatchProcessingPerformance:
    """Benchmark batch processing"""
    
    def test_batch_size_scaling(self, mock_model_parameters):
        """Test how performance scales with batch size"""
        batch_sizes = [10, 50, 100, 200]
        times = []
        
        for batch_size in batch_sizes:
            X = np.random.rand(batch_size, TestConfig.FEATURE_DIM)
            
            start = time.time()
            _ = forward_pass(X, mock_model_parameters)
            elapsed = time.time() - start
            
            times.append(elapsed)
            print(f"Batch size {batch_size}: {elapsed:.4f}s")
        
        # Larger batches should have better throughput (not linear scaling)
        throughputs = [bs / t for bs, t in zip(batch_sizes, times)]
        
        # Throughput should generally increase with batch size
        assert throughputs[-1] > throughputs[0]
    
    def test_optimal_batch_size(self):
        """Find optimal batch size for training"""
        batch_sizes = [8, 16, 32, 64, 128]
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        results = []
        
        for batch_size in batch_sizes:
            X = np.random.rand(batch_size, TestConfig.FEATURE_DIM)
            y = np.random.randint(0, TestConfig.NUM_CLASSES, batch_size)
            
            start = time.time()
            _, _ = train_step(X, y, params, learning_rate=0.01)
            elapsed = time.time() - start
            
            throughput = batch_size / elapsed
            results.append((batch_size, elapsed, throughput))
            
            print(f"Batch {batch_size}: {elapsed:.4f}s, throughput: {throughput:.1f} samples/s")
        
        # All batch sizes should complete
        assert len(results) == len(batch_sizes)


class TestMemoryEfficiency:
    """Test memory efficiency during training"""
    
    @pytest.mark.slow
    def test_training_memory_usage(self):
        """Monitor memory usage during training"""
        import tracemalloc
        
        tracemalloc.start()
        
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 100)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        # Train for 50 epochs
        for _ in range(50):
            params, _ = train_step(X, y, params, learning_rate=0.01)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"Peak memory during training: {peak_mb:.2f} MB")
        
        # Should use < 100MB for this small dataset
        assert peak_mb < 100
    
    def test_gradient_memory_cleanup(self):
        """Test that gradients are properly cleaned up"""
        import tracemalloc
        
        X = np.random.rand(100, TestConfig.FEATURE_DIM)
        y = np.random.randint(0, TestConfig.NUM_CLASSES, 100)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        tracemalloc.start()
        
        # Multiple training steps
        for _ in range(10):
            params, _ = train_step(X, y, params, learning_rate=0.01)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should stabilize (not grow linearly with steps)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 50


class TestTrainingConvergence:
    """Test training convergence speed"""
    
    def test_convergence_speed(self):
        """Measure time to convergence"""
        # Create simple separable dataset
        num_samples = 50
        X = []
        y = []
        
        for i in range(num_samples):
            label = i % TestConfig.NUM_CLASSES
            feature = np.random.rand(TestConfig.FEATURE_DIM) + label * 2
            X.append(feature)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        params = initialize_parameters(
            TestConfig.FEATURE_DIM,
            TestConfig.HIDDEN_DIM,
            TestConfig.NUM_CLASSES
        )
        
        start = time.time()
        epochs_to_converge = 0
        prev_loss = float('inf')
        
        for epoch in range(100):
            params, loss = train_step(X, y, params, learning_rate=0.01)
            
            # Check for convergence (loss change < 0.001)
            if abs(loss - prev_loss) < 0.001:
                epochs_to_converge = epoch
                break
            
            prev_loss = loss
        
        elapsed = time.time() - start
        
        print(f"Converged in {epochs_to_converge} epochs, time: {elapsed:.2f}s")
        
        # Should converge in reasonable time
        assert elapsed < 10.0  # < 10 seconds


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])