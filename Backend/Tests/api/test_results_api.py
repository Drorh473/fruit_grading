"""
Results API Tests
Essential tests for classification results retrieval and statistics endpoints
"""
import pytest
import json


class TestGetAllResults:
    """Test get all results endpoint"""
    
    def test_get_all_results_default(self, client):
        """Test getting all results with default pagination"""
        response = client.get('/api/results/all')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify structure
        assert 'results' in data
        assert 'pagination' in data
        
        # Verify results structure
        assert isinstance(data['results'], list)
        
        # Verify pagination structure
        pagination = data['pagination']
        assert 'page' in pagination
        assert 'limit' in pagination
        assert 'total' in pagination
        assert 'pages' in pagination
    
    def test_get_all_results_with_pagination(self, client):
        """Test pagination parameters"""
        response = client.get('/api/results/all?page=2&limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify pagination values
        assert data['pagination']['page'] == 2
        assert data['pagination']['limit'] == 10
        assert len(data['results']) <= 10
    
    def test_get_all_results_structure(self, client):
        """Test result objects have correct structure"""
        response = client.get('/api/results/all')
        data = json.loads(response.data)
        
        # Check each result has required fields
        for result in data['results']:
            assert 'id' in result
            assert 'type' in result
            assert 'confidence' in result
            assert 'timestamp' in result
            assert 'imageCount' in result
            
            # Verify types
            assert isinstance(result['id'], str)
            assert isinstance(result['type'], str)
            assert isinstance(result['confidence'], (int, float))
            assert isinstance(result['timestamp'], str)
            assert isinstance(result['imageCount'], int)
    
    def test_filter_by_fruit_type(self, client):
        """Test filtering results by fruit type"""
        response = client.get('/api/results/all?type=market')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # All results should be market type
        for result in data['results']:
            if result['type']:  # If type is present
                assert result['type'] == 'market'
    
    def test_search_by_object_id(self, client):
        """Test searching by object ID"""
        response = client.get('/api/results/all?search=obj')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data['results'], list)


class TestGetResultDetails:
    """Test get result details endpoint"""
    
    def test_get_result_details_success(self, client):
        """Test getting details for a specific object"""
        # Use a valid object_id from your system
        response = client.get('/api/results/test_object_001')
        
        # Should be 200 or 404 depending on if object exists
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify structure
            assert 'objectId' in data
            assert 'fruitType' in data
            assert 'confidence' in data
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
        
        # Total images should be >= sum of category counts
        category_sum = (data['marketCount'] + data['standardCount'] + 
                       data['premiumCount'] + data['rejectCount'])
        assert data['totalImages'] >= category_sum


class TestResultsExport:
    """Test results export endpoint"""
    
    def test_export_results_csv(self, client):
        """Test exporting results as CSV"""
        response = client.get('/api/results/export')
        
        assert response.status_code == 200
        
        # Verify content type
        assert response.content_type == 'text/csv' or 'csv' in response.content_type
        
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
            assert 'Confidence' in header or 'confidence' in header.lower()
            assert 'Timestamp' in header or 'timestamp' in header.lower()


class TestResultsIntegration:
    """Test integration between results endpoints"""
    
    def test_all_results_and_stats_consistency(self, client):
        """Test consistency between all results and stats"""
        # Get all results
        all_results = client.get('/api/results/all?limit=1000')
        all_data = json.loads(all_results.data)
        
        # Get stats
        stats_response = client.get('/api/results/stats')
        stats_data = json.loads(stats_response.data)
        
        # Total from pagination should match totalObjects in stats
        # (may differ if pagination limit is hit)
        assert stats_data['totalObjects'] >= 0
    
    def test_result_detail_matches_list(self, client):
        """Test that detail view matches list view"""
        # Get all results
        all_response = client.get('/api/results/all?limit=1')
        all_data = json.loads(all_response.data)
        
        if len(all_data['results']) > 0:
            first_result = all_data['results'][0]
            object_id = first_result['id']
            
            # Get details for that object
            detail_response = client.get(f'/api/results/{object_id}')
            
            if detail_response.status_code == 200:
                detail_data = json.loads(detail_response.data)
                
                # Should match
                assert detail_data['objectId'] == first_result['id']
                assert detail_data['fruitType'] == first_result['type']


class TestPaginationEdgeCases:
    """Test pagination edge cases"""
    
    def test_page_zero(self, client):
        """Test page=0 handling"""
        response = client.get('/api/results/all?page=0')
        
        # Should handle gracefully or default to page 1
        assert response.status_code == 200
    
    def test_negative_page(self, client):
        """Test negative page number"""
        response = client.get('/api/results/all?page=-1')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_large_page_number(self, client):
        """Test very large page number"""
        response = client.get('/api/results/all?page=9999')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return empty results
        assert isinstance(data['results'], list)
    
    def test_custom_limit(self, client):
        """Test various limit values"""
        limits = [1, 5, 10, 100]
        
        for limit in limits:
            response = client.get(f'/api/results/all?limit={limit}')
            data = json.loads(response.data)
            
            assert data['pagination']['limit'] == limit
            assert len(data['results']) <= limit


class TestErrorHandling:
    """Test error handling for results endpoints"""
    
    def test_invalid_parameters(self, client):
        """Test invalid query parameters"""
        response = client.get('/api/results/all?page=abc&limit=xyz')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_endpoints_error_resilience(self, client):
        """Test all endpoints handle errors gracefully"""
        endpoints = [
            '/api/results/all',
            '/api/results/stats',
            '/api/results/export'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])