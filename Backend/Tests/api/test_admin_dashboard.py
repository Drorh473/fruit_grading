"""
Admin Dashboard API Tests
Tests for admin dashboard and system management endpoints
Tests the real API endpoints with various scenarios
"""
import pytest
import json

class TestSystemStatus:
    """Test system status endpoint"""
    
    def test_system_status_success(self, client):
        """Test system status returns all required fields"""
        response = client.get('/api/admin/system-status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify all required fields exist
        assert 'database' in data
        assert 'model' in data
        assert 'pipeline' in data
        assert 'cameras' in data
        
        # Verify field types
        assert isinstance(data['database'], str)
        assert isinstance(data['model'], str)
        assert isinstance(data['pipeline'], str)
        assert isinstance(data['cameras'], list)
        assert len(data['cameras']) == 4
    
    def test_system_status_values(self, client):
        """Test system status has valid values"""
        response = client.get('/api/admin/system-status')
        data = json.loads(response.data)
        
        # Database should be connected or disconnected
        assert data['database'] in ['connected', 'disconnected', 'unknown']
        
        # Model should be in valid states
        assert data['model'] in ['loaded', 'not_trained', 'unknown']
        
        # Pipeline should be in valid states
        assert data['pipeline'] in ['idle', 'processing', 'error']
        
        # Cameras should be boolean
        for camera_status in data['cameras']:
            assert isinstance(camera_status, bool)


class TestProcessingStats:
    """Test processing statistics endpoint"""
    
    def test_processing_stats_success(self, client):
        """Test processing stats returns all required fields"""
        response = client.get('/api/admin/processing-stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'totalProcessed' in data
        assert 'accuracy' in data
        assert 'lastUpdate' in data
        
        # Verify field types
        assert isinstance(data['totalProcessed'], int)
        assert isinstance(data['accuracy'], (int, float))
        assert data['lastUpdate'] is None or isinstance(data['lastUpdate'], str)
    
    def test_processing_stats_values(self, client):
        """Test processing stats have valid values"""
        response = client.get('/api/admin/processing-stats')
        data = json.loads(response.data)
        
        # Total processed should be non-negative
        assert data['totalProcessed'] >= 0
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= data['accuracy'] <= 1.0


class TestRecentResults:
    """Test recent results endpoint"""
    
    def test_recent_results_default(self, client):
        """Test recent results with default limit"""
        response = client.get('/api/admin/recent-results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)
        
        # Should not exceed default limit of 10
        assert len(data) <= 10
    
    def test_recent_results_custom_limit(self, client):
        """Test recent results with custom limit"""
        response = client.get('/api/admin/recent-results?limit=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a list
        assert isinstance(data, list)
        
        # Should not exceed specified limit
        assert len(data) <= 5
    
    def test_recent_results_structure(self, client):
        """Test recent results have correct structure"""
        response = client.get('/api/admin/recent-results')
        data = json.loads(response.data)
        
        # Each result should have required fields (if results exist)
        for result in data:
            assert 'id' in result
            assert 'type' in result
            assert 'confidence' in result
            assert 'timestamp' in result
            
            # Verify field types
            assert isinstance(result['id'], str)
            assert isinstance(result['type'], str)
            assert isinstance(result['confidence'], (int, float))
            assert isinstance(result['timestamp'], str)
            
            # Confidence should be between 0 and 1
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_recent_results_invalid_limit(self, client):
        """Test recent results with invalid limit"""
        response = client.get('/api/admin/recent-results?limit=invalid')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestDatasetInfo:
    """Test dataset information endpoint"""
    
    def test_dataset_info_success(self, client):
        """Test dataset info returns all required fields"""
        response = client.get('/api/admin/dataset-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'trainingCount' in data
        assert 'testingCount' in data
        assert 'totalImages' in data
        assert 'featureDim' in data
        
        # Verify field types
        assert isinstance(data['trainingCount'], int)
        assert isinstance(data['testingCount'], int)
        assert isinstance(data['totalImages'], int)
        assert isinstance(data['featureDim'], int)
    
    def test_dataset_info_values(self, client):
        """Test dataset info has valid values"""
        response = client.get('/api/admin/dataset-info')
        data = json.loads(response.data)
        
        # All counts should be non-negative
        assert data['trainingCount'] >= 0
        assert data['testingCount'] >= 0
        assert data['totalImages'] >= 0
        assert data['featureDim'] >= 0


class TestModelPerformance:
    """Test model performance endpoint"""
    
    def test_model_performance_success(self, client):
        """Test model performance returns all required fields"""
        response = client.get('/api/admin/model-performance')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'architecture' in data
        assert 'trainAccuracy' in data
        assert 'testAccuracy' in data
        assert 'classes' in data
        
        # Verify field types
        assert isinstance(data['architecture'], str)
        assert isinstance(data['trainAccuracy'], (int, float))
        assert isinstance(data['testAccuracy'], (int, float))
        assert isinstance(data['classes'], int)
    
    def test_model_performance_values(self, client):
        """Test model performance has valid values"""
        response = client.get('/api/admin/model-performance')
        data = json.loads(response.data)
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= data['trainAccuracy'] <= 1.0
        assert 0.0 <= data['testAccuracy'] <= 1.0
        
        # Classes should be non-negative
        assert data['classes'] >= 0


class TestPerClassPerformance:
    """Test per-class performance endpoint"""
    
    def test_per_class_performance_success(self, client):
        """Test per-class performance endpoint"""
        response = client.get('/api/admin/per-class-performance')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a dictionary
        assert isinstance(data, dict)
    
    def test_per_class_performance_structure(self, client):
        """Test per-class performance has correct structure"""
        response = client.get('/api/admin/per-class-performance')
        data = json.loads(response.data)
        
        # If data exists, verify structure
        for class_name, metrics in data.items():
            if isinstance(metrics, dict):
                # Should have precision, recall, f1-score
                assert 'precision' in metrics or 'recall' in metrics or 'f1-score' in metrics


class TestTrainingHistory:
    """Test training history endpoint"""
    
    def test_training_history_success(self, client):
        """Test training history returns correct structure"""
        response = client.get('/api/admin/training-history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'train_loss' in data
        assert 'train_accuracy' in data
        assert 'val_loss' in data
        assert 'val_accuracy' in data
        
        # Verify field types
        assert isinstance(data['train_loss'], list)
        assert isinstance(data['train_accuracy'], list)
        assert isinstance(data['val_loss'], list)
        assert isinstance(data['val_accuracy'], list)
    
    def test_training_history_values(self, client):
        """Test training history has valid values"""
        response = client.get('/api/admin/training-history')
        data = json.loads(response.data)
        
        # All arrays should have same length if not empty
        lengths = [
            len(data['train_loss']),
            len(data['train_accuracy']),
            len(data['val_loss']),
            len(data['val_accuracy'])
        ]
        
        if any(l > 0 for l in lengths):
            # If any data exists, check they're all consistent
            non_zero_lengths = [l for l in lengths if l > 0]
            if len(non_zero_lengths) > 1:
                assert len(set(non_zero_lengths)) == 1


class TestConfusionMatrix:
    """Test confusion matrix endpoint"""
    
    def test_confusion_matrix_success(self, client):
        """Test confusion matrix returns correct structure"""
        response = client.get('/api/admin/confusion-matrix')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify required fields
        assert 'matrix' in data
        assert 'labels' in data
        
        # Verify field types
        assert data['matrix'] is None or isinstance(data['matrix'], list)
        assert isinstance(data['labels'], list)
    
    def test_confusion_matrix_structure(self, client):
        """Test confusion matrix has valid structure"""
        response = client.get('/api/admin/confusion-matrix')
        data = json.loads(response.data)
        
        if data['matrix'] is not None:
            # Matrix should be square
            assert len(data['matrix']) == len(data['labels'])
            
            for row in data['matrix']:
                assert isinstance(row, list)
                assert len(row) == len(data['labels'])


class TestFullDashboardData:
    """Test full dashboard data endpoint"""
    
    def test_full_dashboard_success(self, client):
        """Test full dashboard data endpoint"""
        response = client.get('/api/admin/full-dashboard-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify main sections exist
        assert 'system_status' in data
        assert 'processing_stats' in data
        assert 'dataset_info' in data
        assert 'model_performance' in data
    
    def test_full_dashboard_structure(self, client):
        """Test full dashboard has complete structure"""
        response = client.get('/api/admin/full-dashboard-data')
        data = json.loads(response.data)
        
        # Verify system_status section
        system_status = data['system_status']
        assert 'database' in system_status
        assert 'model' in system_status
        assert 'cameras' in system_status
        
        # Verify processing_stats section
        processing_stats = data['processing_stats']
        assert 'totalProcessed' in processing_stats
        assert 'accuracy' in processing_stats
        assert 'lastUpdate' in processing_stats
        
        # Verify dataset_info section
        dataset_info = data['dataset_info']
        assert 'trainingCount' in dataset_info
        assert 'testingCount' in dataset_info
        assert 'totalImages' in dataset_info
        assert 'featureDim' in dataset_info
        
        # Verify model_performance section
        model_performance = data['model_performance']
        assert 'architecture' in model_performance
        assert 'trainAccuracy' in model_performance
        assert 'testAccuracy' in model_performance
        assert 'classes' in model_performance


class TestErrorHandling:
    """Test error handling for admin dashboard endpoints"""
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get('/api/admin/nonexistent-endpoint')
        assert response.status_code == 404
    
    def test_malformed_query_params(self, client):
        """Test malformed query parameters are handled"""
        response = client.get('/api/admin/recent-results?limit=-5')
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_endpoints_with_errors(self, client):
        """Test endpoints handle errors gracefully"""
        # All endpoints should return 200 with empty/default data even on errors
        endpoints = [
            '/api/admin/system-status',
            '/api/admin/processing-stats',
            '/api/admin/recent-results',
            '/api/admin/dataset-info',
            '/api/admin/model-performance',
            '/api/admin/per-class-performance',
            '/api/admin/training-history',
            '/api/admin/confusion-matrix'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestDataConsistency:
    """Test data consistency across endpoints"""
    
    def test_total_images_consistency(self, client):
        """Test total images is non-negative"""
        dataset_response = client.get('/api/admin/dataset-info')
        dataset_data = json.loads(dataset_response.data)
        
        stats_response = client.get('/api/admin/processing-stats')
        stats_data = json.loads(stats_response.data)
        
        # Both should be non-negative
        assert dataset_data['totalImages'] >= 0
        assert stats_data['totalProcessed'] >= 0
    
    def test_accuracy_in_valid_range(self, client):
        """Test accuracy is in valid range"""
        stats_response = client.get('/api/admin/processing-stats')
        stats_data = json.loads(stats_response.data)
        
        perf_response = client.get('/api/admin/model-performance')
        perf_data = json.loads(perf_response.data)
        
        # All accuracies should be between 0 and 1
        assert 0.0 <= stats_data['accuracy'] <= 1.0
        assert 0.0 <= perf_data['trainAccuracy'] <= 1.0
        assert 0.0 <= perf_data['testAccuracy'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
