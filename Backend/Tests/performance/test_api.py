"""
API Performance Tests
Load testing and performance benchmarks for API endpoints
"""
import pytest
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestAPIResponseTime:
    """Test API response times"""
    
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time"""
        times = []
        
        for _ in range(100):
            start = time.time()
            response = client.get('/api/health')
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        print(f"Health endpoint - Avg: {avg_time*1000:.2f}ms, P95: {p95_time*1000:.2f}ms")
        
        # Should be very fast (< 100ms average)
        assert avg_time < 0.1
    
    def test_camera_status_response_time(self, client):
        """Test camera status endpoint response time"""
        times = []
        
        for _ in range(50):
            start = time.time()
            response = client.get('/api/cameras/status')
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        print(f"Camera status - Avg: {avg_time*1000:.2f}ms")
        
        # Should be fast (< 200ms average)
        assert avg_time < 0.2
    
    def test_results_endpoint_response_time(self, client):
        """Test results endpoint response time"""
        times = []
        
        for _ in range(50):
            start = time.time()
            response = client.get('/api/results')
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        print(f"Results endpoint - Avg: {avg_time*1000:.2f}ms")
        
        # Should be reasonably fast (< 500ms average)
        assert avg_time < 0.5


class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    @pytest.mark.slow
    def test_concurrent_health_checks(self, client):
        """Test handling 100 concurrent health checks"""
        num_requests = 100
        
        def make_request():
            start = time.time()
            response = client.get('/api/health')
            elapsed = time.time() - start
            return elapsed, response.status_code
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Check all succeeded
        success_count = sum(1 for _, status in results if status == 200)
        
        print(f"Concurrent requests: {num_requests} in {total_time:.2f}s")
        print(f"Success rate: {success_count}/{num_requests}")
        
        # Should handle all requests successfully
        assert success_count == num_requests
        
        # Should complete in reasonable time (< 10s for 100 requests)
        assert total_time < 10.0
    
    @pytest.mark.slow
    def test_concurrent_camera_status_requests(self, client):
        """Test concurrent camera status requests"""
        num_requests = 50
        
        def make_request():
            response = client.get('/api/cameras/status')
            return response.status_code
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            status_codes = [future.result() for future in as_completed(futures)]
        
        success_count = sum(1 for status in status_codes if status == 200)
        
        print(f"Camera status concurrent: {success_count}/{num_requests} successful")
        
        # Most should succeed
        assert success_count >= num_requests * 0.9  # 90% success rate


class TestAPIThroughput:
    """Test API throughput"""
    
    @pytest.mark.slow
    def test_sustained_load(self, client):
        """Test sustained load for 30 seconds"""
        duration = 30  # seconds
        request_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            response = client.get('/api/health')
            if response.status_code == 200:
                request_count += 1
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed
        
        print(f"Sustained load: {request_count} requests in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        
        # Should handle > 10 req/s
        assert throughput > 10.0
    
    def test_burst_traffic(self, client):
        """Test handling burst of traffic"""
        burst_size = 100
        
        start = time.time()
        
        for _ in range(burst_size):
            response = client.get('/api/health')
        
        elapsed = time.time() - start
        throughput = burst_size / elapsed
        
        print(f"Burst traffic: {burst_size} requests in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        
        # Should handle burst without major slowdown
        assert elapsed < 10.0  # < 10s for 100 requests


class TestDatabaseQueryPerformance:
    """Test database query performance under load"""
    
    def test_results_query_with_large_dataset(self, client, test_collection):
        """Test results query with large dataset"""
        # Insert many results
        results = []
        for i in range(1000):
            results.append({
                "object_id": f"obj{i:04d}",
                "fruit_type": ["market", "standard", "premium"][i % 3],
                "category": "A",
                "timestamp": f"2025-01-01T00:{i//60:02d}:{i%60:02d}"
            })
        
        test_collection.insert_many(results)
        
        # Query results
        start = time.time()
        response = client.get('/api/results?limit=100')
        elapsed = time.time() - start
        
        print(f"Query 100 results from 1000: {elapsed*1000:.2f}ms")
        
        # Should be fast even with large dataset
        assert elapsed < 0.5
    
    def test_filtered_query_performance(self, client, test_collection):
        """Test performance of filtered queries"""
        # Insert test data
        results = []
        for i in range(500):
            results.append({
                "object_id": f"obj{i:04d}",
                "fruit_type": "market" if i % 2 == 0 else "standard",
                "category": "A",
                "timestamp": f"2025-01-01T00:00:00"
            })
        
        test_collection.insert_many(results)
        
        # Filtered query
        start = time.time()
        response = client.get('/api/results?fruit_type=market')
        elapsed = time.time() - start
        
        print(f"Filtered query: {elapsed*1000:.2f}ms")
        
        # Should be fast with indexed field
        assert elapsed < 0.2


class TestAPIErrorHandling:
    """Test API error handling under load"""
    
    def test_invalid_requests_dont_crash(self, client):
        """Test that invalid requests don't crash server"""
        # Send many invalid requests
        for _ in range(100):
            # Invalid endpoint
            response = client.get('/api/nonexistent')
            assert response.status_code == 404
            
            # Invalid method
            response = client.post('/api/health')
            assert response.status_code in [405, 404]
    
    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON"""
        # Send malformed JSON
        response = client.post(
            '/api/fruit/add',
            data='{"invalid": json}',
            content_type='application/json'
        )
        
        # Should return 400 bad request
        assert response.status_code in [400, 415, 422]


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])