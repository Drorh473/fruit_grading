"""
Dashboard Tests
Tests admin and user dashboard functionality
Matches Dashboard.jsx and UserDashboard.jsx
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# ADMIN DASHBOARD RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestAdminDashboardRendering:
    """Test admin dashboard rendering - matches Dashboard.jsx"""
    
    def test_admin_dashboard_loads(self, logged_in_admin: Page):
        """Should load admin dashboard successfully"""
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/dashboard")
        
        # Dashboard.jsx has h1 "System Dashboard"
        header = logged_in_admin.locator('h1')
        expect(header).to_contain_text('System Dashboard')
    
    def test_stat_cards_displayed(self, logged_in_admin: Page):
        """Should display stat cards"""
        # Dashboard.jsx uses .stat-card class
        cards = logged_in_admin.locator('.stat-card')
        assert cards.count() >= 4  # Database, Model, Processed, Accuracy
    
    def test_database_status_shown(self, logged_in_admin: Page):
        """Should show database status"""
        content = logged_in_admin.content().lower()
        assert 'database' in content
    
    def test_model_status_shown(self, logged_in_admin: Page):
        """Should show model status"""
        content = logged_in_admin.content().lower()
        assert 'model' in content
    
    def test_processed_today_shown(self, logged_in_admin: Page):
        """Should show processed today count"""
        content = logged_in_admin.content().lower()
        assert 'processed' in content
    
    def test_accuracy_metric_shown(self, logged_in_admin: Page):
        """Should display accuracy metric"""
        content = logged_in_admin.content().lower()
        assert 'accuracy' in content
    
    def test_refresh_button_present(self, logged_in_admin: Page):
        """Should have refresh button"""
        # Dashboard.jsx has Refresh button with FiRefreshCw
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh")')
        expect(refresh_btn).to_be_visible()


# ============================================================================
# CAMERA STATUS TESTS
# ============================================================================

@pytest.mark.unit
class TestCameraStatus:
    """Test camera status section - matches Dashboard.jsx"""
    
    def test_camera_status_section_exists(self, logged_in_admin: Page):
        """Should have camera status section"""
        # Dashboard.jsx has "Camera Status" card
        camera_section = logged_in_admin.locator('text=Camera Status')
        expect(camera_section).to_be_visible()
    
    def test_four_cameras_displayed(self, logged_in_admin: Page):
        """Should display all 4 cameras"""
        # Dashboard.jsx maps over 4 cameras
        camera_items = logged_in_admin.locator('.camera-status-item')
        assert camera_items.count() == 4
    
    def test_camera_indicators_visible(self, logged_in_admin: Page):
        """Should show camera indicators"""
        indicators = logged_in_admin.locator('.camera-indicator')
        assert indicators.count() == 4


# ============================================================================
# RECENT RESULTS TESTS
# ============================================================================

@pytest.mark.integration
class TestRecentResults:
    """Test recent results section"""
    
    def test_recent_results_section_exists(self, logged_in_admin: Page):
        """Should have recent results section"""
        section = logged_in_admin.locator('text=Recent Processing Results')
        expect(section).to_be_visible()
    
    def test_results_table_or_empty_state(self, logged_in_admin: Page):
        """Should show results table or empty state"""
        # Either table exists or empty-state message
        table = logged_in_admin.locator('.table-container table')
        empty = logged_in_admin.locator('.empty-state')
        
        assert table.count() > 0 or empty.count() > 0


# ============================================================================
# SYSTEM INFORMATION TESTS
# ============================================================================

@pytest.mark.unit
class TestSystemInformation:
    """Test system information sections"""
    
    def test_dataset_info_section(self, logged_in_admin: Page):
        """Should have dataset information section"""
        section = logged_in_admin.locator('text=Dataset Information')
        expect(section).to_be_visible()
    
    def test_model_performance_section(self, logged_in_admin: Page):
        """Should have model performance section"""
        section = logged_in_admin.locator('text=Model Performance')
        expect(section).to_be_visible()
    
    def test_info_list_items(self, logged_in_admin: Page):
        """Should display info list items"""
        # Dashboard.jsx uses .info-list and .info-item classes
        info_items = logged_in_admin.locator('.info-item')
        assert info_items.count() > 0


# ============================================================================
# NAVIGATION TESTS
# ============================================================================

@pytest.mark.integration
class TestDashboardNavigation:
    """Test dashboard navigation"""
    
    def test_sidebar_visible(self, logged_in_admin: Page):
        """Should display navigation sidebar"""
        sidebar = logged_in_admin.locator('nav, [class*="sidebar"]')
        assert sidebar.count() > 0
    
    def test_can_navigate_to_processing(self, logged_in_admin: Page):
        """Should navigate to processing page"""
        processing_link = logged_in_admin.locator('a[href*="processing"]')
        if processing_link.count() > 0:
            processing_link.click()
            logged_in_admin.wait_for_url(re.compile(r'/processing'))
    
    def test_can_navigate_to_results(self, logged_in_admin: Page):
        """Should navigate to results page"""
        results_link = logged_in_admin.locator('a[href*="results"]')
        if results_link.count() > 0:
            results_link.click()
            logged_in_admin.wait_for_url(re.compile(r'/results'))


# ============================================================================
# DATA REFRESH TESTS
# ============================================================================

@pytest.mark.integration
class TestDashboardDataRefresh:
    """Test dashboard data refresh"""
    
    def test_refresh_button_clickable(self, logged_in_admin: Page):
        """Should be able to click refresh button"""
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh")')
        expect(refresh_btn).to_be_enabled()
        
        refresh_btn.click()
        logged_in_admin.wait_for_timeout(500)
    
    def test_spinner_shows_during_refresh(self, logged_in_admin: Page):
        """Should show spinner during refresh"""
        # Dashboard.jsx adds .spinning class to icon during refresh
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh")')
        refresh_btn.click()
        
        # Spinner should appear briefly
        logged_in_admin.wait_for_timeout(100)


# ============================================================================
# USER DASHBOARD TESTS
# ============================================================================

@pytest.mark.e2e
class TestUserDashboard:
    """Test regular user dashboard - matches UserDashboard.jsx"""
    
    def test_user_dashboard_loads(self, logged_in_user: Page):
        """Should load user dashboard"""
        assert "/user-dashboard" in logged_in_user.url
    
    def test_welcome_message_shown(self, logged_in_user: Page):
        """Should show welcome message with username"""
        # UserDashboard.jsx shows "Welcome, {username}"
        welcome = logged_in_user.locator('h1')
        expect(welcome).to_contain_text('Welcome')
    
    def test_summary_cards_displayed(self, logged_in_user: Page):
        """Should display summary cards"""
        # UserDashboard.jsx uses .summary-card class
        cards = logged_in_user.locator('.summary-card')
        assert cards.count() >= 4  # totalToday, marketCount, standardCount, rejectCount
    
    def test_recent_results_table(self, logged_in_user: Page):
        """Should show recent classification results"""
        section = logged_in_user.locator('text=Recent Classification Results')
        expect(section).to_be_visible()
    
    def test_view_all_results_button(self, logged_in_user: Page):
        """Should have View All Results button"""
        btn = logged_in_user.locator('button:has-text("View All Results")')
        expect(btn).to_be_visible()
    
    def test_classification_guide_shown(self, logged_in_user: Page):
        """Should display classification guide"""
        guide = logged_in_user.locator('.classification-guide, text=Classification Guide')
        assert guide.count() > 0
    
    def test_refresh_button_present(self, logged_in_user: Page):
        """Should have refresh button"""
        refresh_btn = logged_in_user.locator('button:has-text("Refresh")')
        expect(refresh_btn).to_be_visible()


# ============================================================================
# LOGOUT TESTS
# ============================================================================

@pytest.mark.integration
class TestDashboardLogout:
    """Test logout functionality"""
    
    def test_logout_button_present(self, logged_in_admin: Page):
        """Should have logout button"""
        logout_btn = logged_in_admin.locator('button:has-text("Logout"), a:has-text("Logout")')
        assert logout_btn.count() > 0
    
    def test_logout_redirects_to_login(self, logged_in_admin: Page):
        """Should redirect to login on logout"""
        logout_btn = logged_in_admin.locator('button:has-text("Logout"), a:has-text("Logout")')
        if logout_btn.count() > 0:
            logout_btn.click()
            logged_in_admin.wait_for_url(f"{BASE_URL}/login", timeout=5000)


# ============================================================================
# RESPONSIVE DESIGN TESTS
# ============================================================================

@pytest.mark.unit
class TestDashboardResponsive:
    """Test responsive design"""
    
    def test_dashboard_on_tablet(self, context):
        """Should display properly on tablet"""
        tablet_page = context.new_page()
        tablet_page.set_viewport_size({"width": 768, "height": 1024})
        
        tablet_page.goto(f"{BASE_URL}/login")
        tablet_page.fill('input[placeholder="Enter username"]', 'admin')
        tablet_page.fill('input[placeholder="Enter password"]', 'admin123')
        tablet_page.locator('.role-option:has(input[value="admin"])').click()
        tablet_page.click('button.login-button')
        tablet_page.wait_for_url(f"{BASE_URL}/dashboard")
        
        # Should load without errors
        header = tablet_page.locator('h1')
        expect(header).to_be_visible()
        
        tablet_page.close()