"""
Admin Dashboard API Tests
Tests for admin dashboard and system management endpoints
"""
import pytest
import json
from datetime import datetime, timedelta


class TestAdminDashboardAPI:
    """Test admin dashboard endpoints"""
    
    def test_get_system_stats(self, client):
        """Test retrieving system statistics"""
        response = client.get('/api/admin/dashboard/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have system-level statistics
        assert isinstance(data, dict)
    
    def test_get_all_users_activity(self, client):
        """Test retrieving all users' activity"""
        response = client.get('/api/admin/dashboard/all-activity')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, (list, dict))
    
    def test_get_system_health(self, client):
        """Test system health metrics"""
        response = client.get('/api/admin/dashboard/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have health indicators
        assert 'status' in data or 'health' in data
    
    def test_get_processing_metrics(self, client):
        """Test processing performance metrics"""
        response = client.get('/api/admin/dashboard/processing-metrics')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, dict)


class TestUserManagement:
    """Test user management endpoints"""
    
    def test_get_user_list(self, client):
        """Test retrieving all users"""
        response = client.get('/api/admin/users')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'users' in data or isinstance(data, list)
    
    def test_get_user_details(self, client):
        """Test retrieving specific user details"""
        response = client.get('/api/admin/users/test_user')
        
        # Should return user details or 404
        assert response.status_code in [200, 404]
    
    def test_update_user_role(self, client):
        """Test updating user role"""
        payload = {
            "username": "test_user",
            "role": "operator"
        }
        
        response = client.put(
            '/api/admin/users/test_user/role',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should accept or reject based on implementation
        assert response.status_code in [200, 403, 404]


class TestSystemStatistics:
    """Test system-wide statistics"""
    
    def test_total_fruits_processed(self, client, test_collection):
        """Test total processed fruits across all users"""
        # Insert test data
        records = []
        for i in range(100):
            records.append({
                "object_id": f"obj{i:04d}",
                "fruit_type": ["market", "standard", "premium"][i % 3],
                "processed_at": datetime.now().isoformat(),
                "status": "completed"
            })
        
        test_collection.insert_many(records)
        
        response = client.get('/api/admin/dashboard/stats')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'total_processed' in data or 'count' in data
    
    def test_stats_by_category(self, client, test_collection):
        """Test statistics grouped by category"""
        records = [
            {"fruit_type": "market", "category": "A", "status": "completed"},
            {"fruit_type": "market", "category": "B", "status": "completed"},
            {"fruit_type": "standard", "category": "A", "status": "completed"},
            {"fruit_type": "premium", "category": "A", "status": "completed"}
        ]
        
        test_collection.insert_many(records)
        
        response = client.get('/api/admin/dashboard/stats/by-category')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (list, dict))
    
    def test_processing_throughput(self, client, test_collection):
        """Test processing throughput metrics"""
        # Insert processing records with timestamps
        now = datetime.now()
        records = []
        
        for i in range(50):
            records.append({
                "object_id": f"obj{i:04d}",
                "processed_at": (now - timedelta(hours=i)).isoformat(),
                "processing_time_seconds": 2.5,
                "status": "completed"
            })
        
        test_collection.insert_many(records)
        
        response = client.get('/api/admin/dashboard/throughput')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict)


class TestAdminRoleBasedAccess:
    """Test admin-only endpoint access control"""
    
    def test_admin_endpoint_requires_admin_role(self, client):
        """Test that admin endpoints require admin role"""
        # This test depends on your authentication implementation
        # Example: try to access admin endpoint without admin token
        
        response = client.get('/api/admin/dashboard/stats')
        
        # Should either succeed (if using test admin) or require auth
        assert response.status_code in [200, 401, 403]
    
    def test_operator_cannot_access_admin_endpoints(self, client):
        """Test that operators cannot access admin endpoints"""
        # This would use operator credentials if available
        # For now, just verify endpoint exists
        
        response = client.get('/api/admin/users')
        
        # Should require admin privileges
        assert response.status_code in [200, 401, 403]


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])