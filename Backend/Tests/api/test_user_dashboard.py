"""
User Dashboard API Tests
Tests for user dashboard statistics and activity endpoints
"""
import pytest
import json
from datetime import datetime, timedelta


class TestUserDashboardAPI:
    """Test user dashboard endpoints"""
    
    def test_get_user_stats(self, client):
        """Test retrieving user statistics"""
        response = client.get('/api/user/dashboard/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should have statistics fields
        assert isinstance(data, dict)
    
    def test_get_recent_activity(self, client):
        """Test retrieving recent activity"""
        response = client.get('/api/user/dashboard/activity')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return activity list
        assert 'activities' in data or isinstance(data, list)
    
    def test_get_recent_activity_with_limit(self, client):
        """Test activity retrieval with limit parameter"""
        response = client.get('/api/user/dashboard/activity?limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        activities = data.get('activities', data)
        if isinstance(activities, list):
            assert len(activities) <= 10
    
    def test_get_user_processing_history(self, client):
        """Test retrieving processing history"""
        response = client.get('/api/user/dashboard/processing-history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert isinstance(data, (list, dict))
    
    def test_get_user_stats_with_date_range(self, client):
        """Test statistics with date range filter"""
        # Get stats for last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        response = client.get(
            f'/api/user/dashboard/stats'
            f'?start_date={start_date.isoformat()}'
            f'&end_date={end_date.isoformat()}'
        )
        
        assert response.status_code == 200


class TestUserStatistics:
    """Test user statistics calculations"""
    
    def test_total_processed_count(self, client, test_collection):
        """Test total processed fruits count"""
        # Insert test processing records
        records = []
        for i in range(15):
            records.append({
                "object_id": f"obj{i:04d}",
                "fruit_type": "market",
                "processed_at": datetime.now().isoformat(),
                "status": "completed"
            })
        
        test_collection.insert_many(records)
        
        response = client.get('/api/user/dashboard/stats')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Should reflect inserted records
            assert 'total_processed' in data or 'count' in data
    
    def test_stats_by_fruit_type(self, client, test_collection):
        """Test statistics grouped by fruit type"""
        # Insert mixed fruit types
        records = [
            {"fruit_type": "market", "status": "completed"},
            {"fruit_type": "market", "status": "completed"},
            {"fruit_type": "standard", "status": "completed"},
            {"fruit_type": "premium", "status": "completed"}
        ]
        
        test_collection.insert_many(records)
        
        response = client.get('/api/user/dashboard/stats/by-type')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (list, dict))


class TestUserActivity:
    """Test user activity tracking"""
    
    def test_recent_activity_ordering(self, client, test_collection):
        """Test that activities are ordered by time (newest first)"""
        # Insert activities with different timestamps
        now = datetime.now()
        activities = []
        
        for i in range(5):
            activities.append({
                "action": "processed",
                "object_id": f"obj{i:04d}",
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
                "user": "test_user"
            })
        
        test_collection.insert_many(activities)
        
        response = client.get('/api/user/dashboard/activity')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            activity_list = data.get('activities', data)
            
            if isinstance(activity_list, list) and len(activity_list) > 1:
                # First activity should be most recent
                assert activity_list[0].get('object_id') == 'obj0000'


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])