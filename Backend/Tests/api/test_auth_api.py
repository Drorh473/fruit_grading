import pytest
import json


class TestAuthenticationAPI:
    """Test authentication endpoints"""
    
    def test_login_success(self, client):
        """Test successful login with valid credentials"""
        payload = {
            "username": "admin",
            "password": "admin123"
        }
        
        response = client.post('/api/auth/login', 
                             data=json.dumps(payload),
                             content_type='application/json')
        
        # Should return success or token
        assert response.status_code in [200, 201]
    
    def test_login_failure_invalid_credentials(self, client):
        """Test rejection of invalid credentials"""
        payload = {
            "username": "invalid",
            "password": "wrong"
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        # Should return 401 unauthorized
        assert response.status_code in [401, 403]
    
    def test_logout(self, client):
        """Test session termination"""
        response = client.post('/api/auth/logout')
        
        assert response.status_code in [200, 204]
    
    def test_role_based_access(self, client):
        """Test admin vs operator permissions"""
        # This would test that admin endpoints require admin role
        # Implementation depends on your auth system
        pass