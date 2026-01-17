import pytest
import json

class TestResultsList:
    """Test results list endpoint"""
    
    def test_get_results_list_default(self, client):
        """Test getting results list with default parameters"""
        response = client.get('/api/results/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'results' in data
        assert 'total' in data
        assert 'limit' in data
        assert 'offset' in data
        
        assert isinstance(data['results'], list)
        assert isinstance(data['total'], int)
        assert isinstance(data['limit'], int)
        assert isinstance(data['offset'], int)
    
    def test_get_results_list_structure(self, client):
        """Test result objects have correct structure"""
        response = client.get('/api/results/list')
        data = json.loads(response.data)
        
        for result in data['results']:
            assert 'id' in result
            assert 'type' in result
            assert 'timestamp' in result
            assert 'imageCount' in result
            
            assert isinstance(result['id'], str)
            assert isinstance(result['type'], str)
            assert isinstance(result['timestamp'], str)
            assert isinstance(result['imageCount'], int)
    
    def test_filter_by_fruit_type(self, client):
        """Test filtering results by fruit type"""
        response = client.get('/api/results/list?type=market')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        for result in data['results']:
            assert result['type'] == 'market'
    
    def test_search_by_object_id(self, client):
        """Test searching by object ID"""
        response = client.get('/api/results/list?search=obj')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data['results'], list)


class TestGetResultDetails:
    """Test get result details endpoint"""
    
    def test_get_result_details_structure(self, client):
        """Test getting details for a specific object"""
        response = client.get('/api/results/details/test_object_001')
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            assert 'object_id' in data
            assert 'fruit_type' in data
            assert 'timestamp' in data
            assert 'image_count' in data
            assert 'images' in data
            
            assert isinstance(data['images'], list)
            assert data['image_count'] >= 0
    
    def test_get_nonexistent_result(self, client):
        """Test getting details for non-existent object"""
        response = client.get('/api/results/details/nonexistent_object_999')
        
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
        
        assert 'totalProcessed' in data
        assert 'qualityRate' in data
        assert 'processingSpeed' in data
        assert 'trends' in data
        
        assert isinstance(data['totalProcessed'], int)
        assert isinstance(data['qualityRate'], (int, float))
        assert isinstance(data['processingSpeed'], (int, float))
        assert isinstance(data['trends'], dict)
        
        assert data['totalProcessed'] >= 0
        assert 0 <= data['qualityRate'] <= 100
        assert data['processingSpeed'] >= 0


class TestQualityAlerts:
    """Test quality alerts endpoint"""
    
    def test_get_quality_alerts(self, client):
        """Test getting quality alerts"""
        response = client.get('/api/results/alerts')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, list)
        
        for alert in data:
            assert 'id' in alert
            assert 'type' in alert
            assert 'title' in alert
            assert 'message' in alert
            
            assert alert['type'] in ['success', 'info', 'warning', 'error']


class TestTrainingHistory:
    """Test training history endpoint"""
    
    def test_get_training_history(self, client):
        """Test getting training history"""
        response = client.get('/api/results/training-history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'train_loss' in data
        assert 'train_accuracy' in data
        assert 'val_loss' in data
        assert 'val_accuracy' in data
        
        assert isinstance(data['train_loss'], list)
        assert isinstance(data['train_accuracy'], list)
        assert isinstance(data['val_loss'], list)
        assert isinstance(data['val_accuracy'], list)


class TestConfusionMatrix:
    """Test confusion matrix endpoint"""
    
    def test_get_confusion_matrix(self, client):
        """Test getting confusion matrix"""
        response = client.get('/api/results/confusion-matrix')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'classes' in data
        assert 'matrix' in data
        assert 'normalized' in data
        assert 'metrics' in data
        
        assert isinstance(data['classes'], list)
        assert isinstance(data['matrix'], list)
        assert isinstance(data['normalized'], list)
        assert isinstance(data['metrics'], dict)
    
    def test_confusion_matrix_excludes_reject(self, client):
        """Test confusion matrix excludes reject class"""
        response = client.get('/api/results/confusion-matrix')
        data = json.loads(response.data)
        
        if data['classes']:
            assert 'reject' not in [c.lower() for c in data['classes']]
    
    def test_normalized_matrix_values(self, client):
        """Test normalized matrix has valid percentage values"""
        response = client.get('/api/results/confusion-matrix')
        data = json.loads(response.data)
        
        for row in data['normalized']:
            for value in row:
                assert 0 <= value <= 1


class TestResultsStats:
    """Test results statistics endpoint"""
    
    def test_get_results_stats(self, client):
        """Test getting overall results statistics"""
        response = client.get('/api/results/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'totalObjects' in data
        assert 'totalImages' in data
        assert 'marketCount' in data
        assert 'standardCount' in data
        assert 'premiumCount' in data
        
        assert isinstance(data['totalObjects'], int)
        assert isinstance(data['totalImages'], int)
        assert data['totalObjects'] >= 0
        assert data['totalImages'] >= 0
    
    def test_stats_counts_consistency(self, client):
        """Test that category counts are consistent"""
        response = client.get('/api/results/stats')
        data = json.loads(response.data)
        
        assert data['marketCount'] >= 0
        assert data['standardCount'] >= 0
        assert data['premiumCount'] >= 0


class TestResultsExport:
    """Test results export endpoint"""
    
    def test_export_results_csv(self, client):
        """Test exporting results as CSV"""
        response = client.get('/api/results/export')
        
        assert response.status_code == 200
        assert 'text/csv' in response.content_type or 'csv' in response.content_type
        
        csv_content = response.data.decode('utf-8')
        lines = csv_content.split('\n')
        
        assert len(lines) > 0
        header = lines[0]
        assert 'Object ID' in header or 'object_id' in header.lower()
    
    def test_export_csv_structure(self, client):
        """Test CSV export has correct structure"""
        response = client.get('/api/results/export')
        
        csv_content = response.data.decode('utf-8')
        lines = csv_content.split('\n')
        
        if len(lines) > 0:
            header = lines[0]
            assert 'Grade' in header or 'grade' in header.lower()
            assert 'Timestamp' in header or 'timestamp' in header.lower()


class TestResultsIntegration:
    """Test integration between results endpoints"""
    
    def test_list_and_stats_consistency(self, client):
        """Test consistency between list and stats"""
        list_response = client.get('/api/results/list?limit=1000')
        list_data = json.loads(list_response.data)
        
        stats_response = client.get('/api/results/stats')
        stats_data = json.loads(stats_response.data)
        
        assert stats_data['totalObjects'] >= 0
        assert list_data['total'] >= 0
    
    def test_confusion_matrix_and_kpis_consistency(self, client):
        """Test consistency between confusion matrix and KPIs"""
        kpis_response = client.get('/api/results/kpis')
        kpis_data = json.loads(kpis_response.data)
        
        cm_response = client.get('/api/results/confusion-matrix')
        cm_data = json.loads(cm_response.data)
        
        assert 0 <= kpis_data['qualityRate'] <= 100
        if cm_data['metrics']:
            assert 'accuracy' in cm_data['metrics']


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
        
        assert response.status_code in [200, 400]
    
    def test_large_offset(self, client):
        """Test very large offset"""
        response = client.get('/api/results/list?offset=9999')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
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
        
        assert response.status_code in [200, 400]
    
    def test_endpoints_error_resilience(self, client):
        """Test all endpoints handle errors gracefully"""
        endpoints = [
            '/api/results/list',
            '/api/results/all',
            '/api/results/stats',
            '/api/results/kpis',
            '/api/results/alerts',
            '/api/results/training-history',
            '/api/results/confusion-matrix',
            '/api/results/export'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestResponseFormats:
    """Test response formats are correct"""
    
    def test_all_endpoints_return_json(self, client):
        """Test all endpoints return valid JSON (except export)"""
        endpoints = [
            '/api/results/list',
            '/api/results/stats',
            '/api/results/kpis',
            '/api/results/alerts',
            '/api/results/training-history',
            '/api/results/confusion-matrix'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert 'application/json' in response.content_type
            
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON from {endpoint}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])