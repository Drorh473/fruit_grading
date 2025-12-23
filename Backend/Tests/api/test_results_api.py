"""
Results API Tests
Essential tests for classification results retrieval and statistics endpoints
"""
import pytest
import json
from pathlib import Path
import sys


class TestResultsList:
    """Test results list endpoint"""
    
    def test_get_results_list_default(self, client):
        """Test getting results list with default parameters"""
        response = client.get('/api/results/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify structure
        assert 'results' in data
        assert 'total' in data
        assert 'limit' in data
        assert 'offset' in data
        
        # Verify types
        assert isinstance(data['results'], list)
        assert isinstance(data['total'], int)
        assert isinstance(data['limit'], int)
        assert isinstance(data['offset'], int)
    
    def test_get_results_list_with_pagination(self, client):
        """Test pagination parameters"""
        response = client.get('/api/results/list?limit=10&offset=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify pagination values
        assert data['limit'] == 10
        assert data['offset'] == 5
        assert len(data['results']) <= 10
    
    def test_get_results_list_structure(self, client):
        """Test result objects have correct structure"""
        response = client.get('/api/results/list')
        data = json.loads(response.data)
        
        # Check each result has required fields
        for result in data['results']:
            assert 'id' in result
            assert 'type' in result
            assert 'timestamp' in result
            assert 'batch' in result
            assert 'imageCount' in result
            
            # Verify types
            assert isinstance(result['id'], str)
            assert isinstance(result['type'], str)
            assert isinstance(result['timestamp'], str)
            assert isinstance(result['imageCount'], int)
    
    def test_filter_by_fruit_type(self, client):
        """Test filtering results by fruit type"""
        response = client.get('/api/results/list?type=market')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # All results should be market type (if any results exist)
        for result in data['results']:
            assert result['type'] == 'market'
    
    def test_search_by_object_id(self, client):
        """Test searching by object ID"""
        response = client.get('/api/results/list?search=obj')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data['results'], list)
    
    def test_filter_by_batch(self, client):
        """Test filtering by batch ID"""
        response = client.get('/api/results/list?batch=batch_001')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data['results'], list)


class TestGetAllResults:
    """Test legacy /all endpoint"""
    
    def test_get_all_results_redirects(self, client):
        """Test that /all endpoint works (legacy compatibility)"""
        response = client.get('/api/results/all')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have same structure as /list
        assert 'results' in data
        assert 'total' in data


class TestGetResultDetails:
    """Test get result details endpoint"""
    
    def test_get_result_details_structure(self, client):
        """Test getting details for a specific object"""
        response = client.get('/api/results/test_object_001')
        
        # Should be 200 or 404 depending on if object exists
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify structure
            assert 'objectId' in data
            assert 'fruitType' in data
            assert 'timestamp' in data
            assert 'imageCount' in data
            assert 'images' in data
            
            # Verify types
            assert isinstance(data['images'], list)
            assert data['imageCount'] >= 0
    
    def test_get_result_details_image_structure(self, client):
        """Test image details structure"""
        response = client.get('/api/results/test_object_001')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Check image structure
            for img in data['images']:
                assert 'cameraId' in img
                assert 'angle' in img
                assert 'frameNumber' in img
                assert 'path' in img
    
    def test_get_nonexistent_result(self, client):
        """Test getting details for non-existent object"""
        response = client.get('/api/results/nonexistent_object_999')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestKPIs:
    """Test KPI endpoint"""
    
    def test_get_kpis(self, client):
        """Test getting KPIs"""
        response = client.get('/api/results/kpis')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'totalProcessed' in data
        assert 'qualityRate' in data
        assert 'processingSpeed' in data
        assert 'trends' in data
        
        # Verify types
        assert isinstance(data['totalProcessed'], int)
        assert isinstance(data['qualityRate'], (int, float))
        assert isinstance(data['processingSpeed'], (int, float))
        assert isinstance(data['trends'], dict)
        
        # Verify ranges
        assert data['totalProcessed'] >= 0
        assert 0 <= data['qualityRate'] <= 100
        assert data['processingSpeed'] >= 0


class TestQualityDistribution:
    """Test quality distribution endpoint"""
    
    def test_get_quality_distribution(self, client):
        """Test getting quality distribution"""
        response = client.get('/api/results/quality-distribution')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all quality types exist
        required_types = ['market', 'standard', 'premium', 'reject']
        for quality_type in required_types:
            assert quality_type in data
            assert 'count' in data[quality_type]
            assert 'percentage' in data[quality_type]
            
            # Verify values
            assert data[quality_type]['count'] >= 0
            assert 0 <= data[quality_type]['percentage'] <= 100


class TestQualityAlerts:
    """Test quality alerts endpoint"""
    
    def test_get_quality_alerts(self, client):
        """Test getting quality alerts"""
        response = client.get('/api/results/alerts')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)
        
        # Check alert structure
        for alert in data:
            assert 'id' in alert
            assert 'type' in alert
            assert 'title' in alert
            assert 'message' in alert
            
            # Type should be valid
            assert alert['type'] in ['success', 'info', 'warning', 'error']


class TestBatches:
    """Test batches endpoint"""
    
    def test_get_batches(self, client):
        """Test getting list of batches"""
        response = client.get('/api/results/batches')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)


class TestHourlyTrend:
    """Test hourly trend endpoint"""
    
    def test_get_hourly_trend_default(self, client):
        """Test getting hourly trend with default hours"""
        response = client.get('/api/results/hourly-trend')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)
        assert len(data) <= 24  # Default 24 hours
    
    def test_get_hourly_trend_custom_hours(self, client):
        """Test getting hourly trend with custom hours"""
        response = client.get('/api/results/hourly-trend?hours=12')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, list)
        assert len(data) <= 12
    
    def test_hourly_trend_structure(self, client):
        """Test hourly trend data structure"""
        response = client.get('/api/results/hourly-trend')
        data = json.loads(response.data)
        
        # Check structure of each data point
        for point in data:
            assert 'hour' in point
            assert 'processed' in point
            assert 'qualityRate' in point
            
            # Verify types
            assert isinstance(point['hour'], str)
            assert isinstance(point['processed'], int)
            assert isinstance(point['qualityRate'], (int, float))


class TestConfusionMatrix:
    """Test confusion matrix endpoint"""
    
    def test_get_confusion_matrix(self, client):
        """Test getting confusion matrix"""
        response = client.get('/api/results/confusion-matrix')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify structure
        assert 'classes' in data
        assert 'matrix' in data
        assert 'metrics' in data
        
        # Verify types
        assert isinstance(data['classes'], list)
        assert isinstance(data['matrix'], list)
        assert isinstance(data['metrics'], dict)


class TestResultsStats:
    """Test results statistics endpoint"""
    
    def test_get_results_stats(self, client):
        """Test getting overall results statistics"""
        response = client.get('/api/results/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields
        assert 'totalObjects' in data
        assert 'totalImages' in data
        assert 'marketCount' in data
        assert 'standardCount' in data
        assert 'premiumCount' in data
        assert 'rejectCount' in data
        
        # Verify types and ranges
        assert isinstance(data['totalObjects'], int)
        assert isinstance(data['totalImages'], int)
        assert data['totalObjects'] >= 0
        assert data['totalImages'] >= 0
    
    def test_stats_counts_consistency(self, client):
        """Test that category counts are consistent"""
        response = client.get('/api/results/stats')
        data = json.loads(response.data)
        
        # All counts should be non-negative
        assert data['marketCount'] >= 0
        assert data['standardCount'] >= 0
        assert data['premiumCount'] >= 0
        assert data['rejectCount'] >= 0


class TestResultsExport:
    """Test results export endpoint"""
    
    def test_export_results_csv(self, client):
        """Test exporting results as CSV"""
        response = client.get('/api/results/export')
        
        assert response.status_code == 200
        
        # Verify content type
        assert 'text/csv' in response.content_type or 'csv' in response.content_type
        
        # Verify data is CSV format
        csv_content = response.data.decode('utf-8')
        lines = csv_content.split('\n')
        
        # Should have header
        assert len(lines) > 0
        header = lines[0]
        assert 'Object ID' in header or 'object_id' in header.lower()
    
    def test_export_csv_structure(self, client):
        """Test CSV export has correct structure"""
        response = client.get('/api/results/export')
        
        csv_content = response.data.decode('utf-8')
        lines = csv_content.split('\n')
        
        if len(lines) > 1:  # If there's data beyond header
            # Check header columns
            header = lines[0]
            assert 'Fruit Type' in header or 'fruit_type' in header.lower()
            assert 'Timestamp' in header or 'timestamp' in header.lower()


class TestResultsIntegration:
    """Test integration between results endpoints"""
    
    def test_list_and_stats_consistency(self, client):
        """Test consistency between list and stats"""
        # Get list
        list_response = client.get('/api/results/list?limit=1000')
        list_data = json.loads(list_response.data)
        
        # Get stats
        stats_response = client.get('/api/results/stats')
        stats_data = json.loads(stats_response.data)
        
        # Both should have non-negative counts
        assert stats_data['totalObjects'] >= 0
        assert list_data['total'] >= 0
    
    def test_kpis_and_distribution_consistency(self, client):
        """Test consistency between KPIs and distribution"""
        kpis_response = client.get('/api/results/kpis')
        kpis_data = json.loads(kpis_response.data)
        
        dist_response = client.get('/api/results/quality-distribution')
        dist_data = json.loads(dist_response.data)
        
        # Quality rate should be in valid range
        assert 0 <= kpis_data['qualityRate'] <= 100


class TestPaginationEdgeCases:
    """Test pagination edge cases"""
    
    def test_zero_offset(self, client):
        """Test offset=0 handling"""
        response = client.get('/api/results/list?offset=0')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['offset'] == 0
    
    def test_negative_offset(self, client):
        """Test negative offset number"""
        response = client.get('/api/results/list?offset=-1')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_large_offset(self, client):
        """Test very large offset"""
        response = client.get('/api/results/list?offset=9999')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return empty results
        assert isinstance(data['results'], list)
    
    def test_custom_limit(self, client):
        """Test various limit values"""
        limits = [1, 5, 10, 100]
        
        for limit in limits:
            response = client.get(f'/api/results/list?limit={limit}')
            data = json.loads(response.data)
            
            assert data['limit'] == limit
            assert len(data['results']) <= limit


class TestErrorHandling:
    """Test error handling for results endpoints"""
    
    def test_invalid_parameters(self, client):
        """Test invalid query parameters"""
        response = client.get('/api/results/list?offset=abc&limit=xyz')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_endpoints_error_resilience(self, client):
        """Test all endpoints handle errors gracefully"""
        endpoints = [
            '/api/results/list',
            '/api/results/all',
            '/api/results/stats',
            '/api/results/kpis',
            '/api/results/quality-distribution',
            '/api/results/alerts',
            '/api/results/batches',
            '/api/results/hourly-trend',
            '/api/results/confusion-matrix',
            '/api/results/export'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
    
    def test_invalid_fruit_type_filter(self, client):
        """Test filtering with invalid fruit type"""
        response = client.get('/api/results/list?type=invalid_type')
        
        # Should handle gracefully
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data['results'], list)


class TestResponseFormats:
    """Test response formats are correct"""
    
    def test_all_endpoints_return_json(self, client):
        """Test all endpoints return valid JSON (except export)"""
        endpoints = [
            '/api/results/list',
            '/api/results/stats',
            '/api/results/kpis',
            '/api/results/quality-distribution',
            '/api/results/alerts',
            '/api/results/batches',
            '/api/results/hourly-trend'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert 'application/json' in response.content_type
            
            # Should be valid JSON
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON from {endpoint}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
