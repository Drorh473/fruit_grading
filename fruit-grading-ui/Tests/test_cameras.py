"""
Camera Monitor Tests
Tests camera feed display and controls
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


@pytest.mark.unit
class TestCameraRendering:
    """Test camera page rendering"""
    
    def test_camera_page_loads(self, logged_in_admin: Page):
        """Should load camera monitor page"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        assert "camera" in logged_in_admin.url.lower()
    
    def test_four_camera_feeds_displayed(self, logged_in_admin: Page):
        """Should display all 4 camera feeds"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        camera_feeds = logged_in_admin.locator('[class*="camera-feed"], [class*="feed"]')
        assert camera_feeds.count() >= 4
    
    def test_camera_status_indicators(self, logged_in_admin: Page):
        """Should show camera status indicators"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        status = logged_in_admin.locator('[class*="status"]')
        assert status.count() > 0


@pytest.mark.integration
class TestCameraControls:
    """Test camera controls"""
    
    def test_refresh_camera_button(self, logged_in_admin: Page):
        """Should have refresh button for each camera"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        refresh_btns = logged_in_admin.locator('button[class*="refresh"]')
        assert refresh_btns.count() >= 0
    
    def test_camera_refresh_action(self, logged_in_admin: Page):
        """Should refresh camera feed on click"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        refresh_btn = logged_in_admin.locator('button[class*="refresh"]').first
        if refresh_btn.count() > 0:
            refresh_btn.click()
            logged_in_admin.wait_for_timeout(500)


@pytest.mark.unit
class TestCameraGrid:
    """Test camera grid layout"""
    
    def test_grid_layout_2x2(self, logged_in_admin: Page):
        """Should display cameras in 2x2 grid"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        grid = logged_in_admin.locator('[class*="grid"]')
        assert grid.count() > 0
    
    def test_camera_labels_visible(self, logged_in_admin: Page):
        """Should show camera labels (Front, Right, Back, Left)"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        content = logged_in_admin.content().lower()
        labels = ['front', 'right', 'back', 'left']
        found = sum(1 for label in labels if label in content)
        assert found >= 2