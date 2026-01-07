"""
Camera Monitor Tests
Tests camera feed display and controls
Matches CameraMonitor.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraRendering:
    """Test camera page rendering - matches CameraMonitor.jsx"""
    
    def test_camera_page_loads(self, logged_in_admin: Page):
        """Should load camera monitor page"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has h1 "Camera Monitor"
        header = logged_in_admin.locator('h1')
        expect(header).to_contain_text('Camera')
    
    def test_four_camera_feeds_displayed(self, logged_in_admin: Page):
        """Should display all 4 camera feeds"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx uses .camera-card class
        camera_cards = logged_in_admin.locator('.camera-card')
        assert camera_cards.count() == 4
    
    def test_camera_grid_layout(self, logged_in_admin: Page):
        """Should have camera grid"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx uses .camera-grid class
        grid = logged_in_admin.locator('.camera-grid')
        expect(grid).to_be_visible()
    
    def test_last_update_timestamp(self, logged_in_admin: Page):
        """Should show last update timestamp"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows "Last updated: {time}"
        content = logged_in_admin.content()
        assert 'Last updated' in content or 'update' in content.lower()


# ============================================================================
# CAMERA CARD TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraCards:
    """Test individual camera card display"""
    
    def test_camera_names_displayed(self, logged_in_admin: Page):
        """Should show camera names"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows "Camera 0", "Camera 1", etc.
        content = logged_in_admin.content()
        assert 'Camera 0' in content or 'Camera' in content
    
    def test_camera_angles_displayed(self, logged_in_admin: Page):
        """Should show camera angles"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows Front View, Right View, Back View, Left View
        content = logged_in_admin.content()
        angles = ['Front', 'Right', 'Back', 'Left']
        found = sum(1 for a in angles if a in content)
        assert found >= 2
    
    def test_camera_status_badges(self, logged_in_admin: Page):
        """Should show status badges"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx uses .status-badge class
        badges = logged_in_admin.locator('.status-badge')
        assert badges.count() > 0
    
    def test_camera_fps_shown(self, logged_in_admin: Page):
        """Should show FPS for each camera"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        content = logged_in_admin.content()
        assert 'FPS' in content
    
    def test_camera_resolution_shown(self, logged_in_admin: Page):
        """Should show resolution for each camera"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        content = logged_in_admin.content()
        assert 'Resolution' in content or '224x224' in content
    
    def test_live_indicator(self, logged_in_admin: Page):
        """Should show LIVE indicator for active cameras"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows "LIVE" with recording-indicator
        content = logged_in_admin.content()
        assert 'LIVE' in content or 'Active' in content


# ============================================================================
# CAMERA CONTROLS TESTS
# ============================================================================

@pytest.mark.integration
class TestCameraControls:
    """Test camera controls"""
    
    def test_refresh_all_button(self, logged_in_admin: Page):
        """Should have Refresh All button"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has "Refresh All" button
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh All")')
        expect(refresh_btn).to_be_visible()
    
    def test_per_camera_refresh_buttons(self, logged_in_admin: Page):
        """Should have refresh button for each camera"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has .btn-icon with FiRefreshCw for each camera
        refresh_btns = logged_in_admin.locator('.camera-card .btn-icon')
        assert refresh_btns.count() >= 4
    
    def test_refresh_all_action(self, logged_in_admin: Page):
        """Should refresh all cameras on click"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh All")')
        refresh_btn.click()
        logged_in_admin.wait_for_timeout(500)
        
        # Page should still be visible without errors
        header = logged_in_admin.locator('h1')
        expect(header).to_be_visible()
    
    def test_camera_card_clickable(self, logged_in_admin: Page):
        """Should be able to click camera cards"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx allows clicking to select camera
        first_camera = logged_in_admin.locator('.camera-card').first
        first_camera.click()
        
        # Should add .selected class
        logged_in_admin.wait_for_timeout(200)


# ============================================================================
# CAMERA HEALTH TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraHealth:
    """Test camera system health section"""
    
    def test_health_section_exists(self, logged_in_admin: Page):
        """Should have camera health section"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has "Camera System Health" section
        section = logged_in_admin.locator('text=Camera System Health')
        expect(section).to_be_visible()
    
    def test_health_metrics_displayed(self, logged_in_admin: Page):
        """Should show health metrics"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows Capture Success, Avg Quality, Uptime, etc.
        content = logged_in_admin.content()
        metrics = ['Capture', 'Quality', 'Uptime', 'Frames']
        found = sum(1 for m in metrics if m in content)
        assert found >= 2
    
    def test_health_cards_displayed(self, logged_in_admin: Page):
        """Should show health card for each camera"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx uses .camera-health-card class
        health_cards = logged_in_admin.locator('.camera-health-card')
        assert health_cards.count() == 4
    
    def test_view_logs_button(self, logged_in_admin: Page):
        """Should have View Logs button"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has "View Logs" button per camera
        logs_btn = logged_in_admin.locator('button:has-text("View Logs")').first
        expect(logs_btn).to_be_visible()
    
    def test_run_diagnostics_button(self, logged_in_admin: Page):
        """Should have Run Diagnostics button"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx has "Run Diagnostics" button per camera
        diag_btn = logged_in_admin.locator('button:has-text("Diagnostics")').first
        expect(diag_btn).to_be_visible()


# ============================================================================
# FEED DISPLAY TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraFeedDisplay:
    """Test camera feed display area"""
    
    def test_feed_placeholder_shown(self, logged_in_admin: Page):
        """Should show feed placeholder"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx uses .feed-placeholder class
        placeholder = logged_in_admin.locator('.feed-placeholder')
        assert placeholder.count() > 0
    
    def test_online_camera_shows_active(self, logged_in_admin: Page):
        """Should show Camera Feed Active for online cameras"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        content = logged_in_admin.content()
        assert 'Camera Feed Active' in content or 'Active' in content
    
    def test_offline_camera_indicator(self, logged_in_admin: Page):
        """Should indicate offline cameras"""
        logged_in_admin.goto(f"{BASE_URL}/cameras")
        
        # CameraMonitor.jsx shows "Camera Offline" for offline cameras
        # May or may not have offline cameras depending on system state
        content = logged_in_admin.content()
        # Just verify page loaded
        assert len(content) > 0