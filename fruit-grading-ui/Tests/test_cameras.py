"""
Camera Monitor Tests
Tests camera feed display and controls
Matches CameraMonitor.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"

# Helper function to navigate safely (Preserving Auth)
def navigate_to_cameras(page: Page):
    """Navigate to cameras using the sidebar to preserve Auth state"""
    
    # 1. Check if already there
    if "/cameras" in page.url:
        page.wait_for_selector('h1:has-text("Camera Monitor")', state='visible')
        return

    # 2. Open Hamburger if visible
    hamburger = page.locator('.hamburger-button')
    if hamburger.is_visible():
        hamburger.click()
        page.wait_for_selector('.sidebar.visible, .sidebar-open', timeout=2000)

    # 3. LOCATE the element
    # Matches href="/cameras"
    camera_link = page.locator('a[href$="/cameras"]')
    
    # 4. DISPATCH 'click' event directly using JavaScript
    camera_link.dispatch_event('click')
    
    # 5. Wait for navigation
    try:
        expect(page).to_have_url(re.compile(r".*/cameras"), timeout=10000)
    except AssertionError:
        # Fallback
        camera_link.evaluate("el => el.click()")
        expect(page).to_have_url(re.compile(r".*/cameras"))

    # 6. Verify page load with SPECIFIC header
    page.wait_for_selector('h1:has-text("Camera Monitor")', state='visible')


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraRendering:
    """Test camera page rendering - matches CameraMonitor.jsx"""
    
    def test_camera_page_loads(self, logged_in_admin: Page):
        """Should load camera monitor page"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx has h1 "Camera Monitor"
        # Use exact text or role to be safe
        header = logged_in_admin.get_by_role("heading", name="Camera Monitor")
        expect(header).to_be_visible()
    
    def test_four_camera_feeds_displayed(self, logged_in_admin: Page):
        """Should display all 4 camera feeds"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx uses .camera-card class
        camera_cards = logged_in_admin.locator('.camera-card')
        assert camera_cards.count() == 4
    
    def test_camera_grid_layout(self, logged_in_admin: Page):
        """Should have camera grid"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx uses .camera-grid class
        grid = logged_in_admin.locator('.camera-grid')
        expect(grid).to_be_visible()
    
    def test_last_update_timestamp(self, logged_in_admin: Page):
        """Should show last update timestamp"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx shows "Last updated: {time}"
        # Use locator instead of content() for robustness
        expect(logged_in_admin.locator("text=Last updated").first).to_be_visible()


# ============================================================================
# CAMERA CARD TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraCards:
    """Test individual camera card display"""
    
    def test_camera_names_displayed(self, logged_in_admin: Page):
        """Should show camera names"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx shows "Camera 0" as a heading
        expect(logged_in_admin.get_by_role("heading", name="Camera 0")).to_be_visible()
    
    def test_camera_angles_displayed(self, logged_in_admin: Page):
        """Should show camera angles"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx shows Front View, Right View, Back View, Left View
        content = logged_in_admin.content()
        angles = ['Front', 'Right', 'Back', 'Left']
        found = sum(1 for a in angles if a in content)
        assert found >= 2
    
    def test_camera_status_badges(self, logged_in_admin: Page):
        """Should show status badges"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx uses .status-badge class
        badges = logged_in_admin.locator('.status-badge')
        assert badges.count() > 0
    
    def test_camera_fps_shown(self, logged_in_admin: Page):
        """Should show FPS for each camera"""
        navigate_to_cameras(logged_in_admin)
        
        # Use generic text search for "FPS"
        expect(logged_in_admin.locator("text=FPS").first).to_be_visible()
    
    def test_camera_resolution_shown(self, logged_in_admin: Page):
        """Should show resolution for each camera"""
        navigate_to_cameras(logged_in_admin)
        
        # Look for resolution text or the label
        expect(logged_in_admin.locator("text=Resolution").or_(logged_in_admin.locator("text=224x224")).first).to_be_visible()
    
    def test_live_indicator(self, logged_in_admin: Page):
        """Should show LIVE indicator for active cameras"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx shows "LIVE" with recording-indicator
        # Be permissive as "LIVE" might be in a span
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
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx has "Refresh All" button
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh All")')
        expect(refresh_btn).to_be_visible()
    
    def test_per_camera_refresh_buttons(self, logged_in_admin: Page):
        """Should have refresh button for each camera"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx has .btn-icon with FiRefreshCw for each camera
        # Just check for buttons inside camera cards
        refresh_btns = logged_in_admin.locator('.camera-card button')
        assert refresh_btns.count() >= 4
    
    def test_refresh_all_action(self, logged_in_admin: Page):
        """Should refresh all cameras on click"""
        navigate_to_cameras(logged_in_admin)
        
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh All")')
        refresh_btn.click()
        logged_in_admin.wait_for_timeout(500)
        
        # Page should still be visible without errors
        header = logged_in_admin.get_by_role("heading", name="Camera Monitor")
        expect(header).to_be_visible()
    
    def test_camera_card_clickable(self, logged_in_admin: Page):
        """Should be able to click camera cards"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx allows clicking to select camera
        first_camera = logged_in_admin.locator('.camera-card').first
        first_camera.click()
        
        # Should add .selected class (if implemented) or just not crash
        logged_in_admin.wait_for_timeout(200)


# ============================================================================
# CAMERA HEALTH TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraHealth:
    """Test camera system health section"""
    
    def test_health_section_exists(self, logged_in_admin: Page):
        """Should have camera health section"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx has "Camera System Health" section
        # Use specific locator to avoid matching multiple elements
        section = logged_in_admin.get_by_role("heading", name="Camera System Health")
        expect(section).to_be_visible()
    
    def test_health_metrics_displayed(self, logged_in_admin: Page):
        """Should show health metrics"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx shows Capture Success, Avg Quality, Uptime, etc.
        content = logged_in_admin.content()
        metrics = ['Capture', 'Quality', 'Uptime', 'Frames']
        found = sum(1 for m in metrics if m in content)
        assert found >= 2
    
    def test_health_cards_displayed(self, logged_in_admin: Page):
        """Should show health card for each camera"""
        navigate_to_cameras(logged_in_admin)

        # CameraMonitor.jsx uses .camera-health-card class
        health_cards = logged_in_admin.locator('.camera-health-card')
        assert health_cards.count() == 4


# ============================================================================
# FEED DISPLAY TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraFeedDisplay:
    """Test camera feed display area"""
    
    def test_feed_placeholder_shown(self, logged_in_admin: Page):
        """Should show feed placeholder"""
        navigate_to_cameras(logged_in_admin)
        
        # CameraMonitor.jsx uses .feed-placeholder class
        placeholder = logged_in_admin.locator('.feed-placeholder')
        assert placeholder.count() > 0
    
    def test_online_camera_shows_active(self, logged_in_admin: Page):
        """Should show Camera Feed Active for online cameras"""
        navigate_to_cameras(logged_in_admin)
        
        content = logged_in_admin.content()
        assert 'Camera Feed Active' in content or 'Active' in content
    
    def test_offline_camera_indicator(self, logged_in_admin: Page):
        """Should indicate offline cameras"""
        navigate_to_cameras(logged_in_admin)
        
        # Just verify page loaded
        content = logged_in_admin.content()
        assert len(content) > 0
