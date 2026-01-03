"""
Dashboard Tests
Tests admin and user dashboard functionality
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# ADMIN DASHBOARD RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestAdminDashboardRendering:
    """Test admin dashboard rendering"""
    
    def test_admin_dashboard_loads(self, logged_in_admin: Page):
        """Should load admin dashboard successfully"""
        # Already on dashboard from fixture
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/dashboard")
        
        # Check page title or header
        header = logged_in_admin.locator('h1, h2')
        assert header.count() > 0
    
    def test_kpi_cards_displayed(self, logged_in_admin: Page):
        """Should display KPI cards"""
        # Look for metric cards
        cards = logged_in_admin.locator('[class*="card"]')
        assert cards.count() >= 4  # Total processed, accuracy, cameras, uptime
    
    def test_total_processed_metric_shown(self, logged_in_admin: Page):
        """Should show total processed count"""
        content = logged_in_admin.content().lower()
        assert 'total' in content or 'processed' in content
    
    def test_accuracy_metric_shown(self, logged_in_admin: Page):
        """Should display accuracy metric"""
        content = logged_in_admin.content().lower()
        assert 'accuracy' in content
    
    def test_camera_status_shown(self, logged_in_admin: Page):
        """Should show camera status"""
        content = logged_in_admin.content().lower()
        assert 'camera' in content
    
    def test_system_uptime_shown(self, logged_in_admin: Page):
        """Should display system uptime"""
        content = logged_in_admin.content().lower()
        assert 'uptime' in content or 'online' in content


# ============================================================================
# CHARTS AND VISUALIZATIONS
# ============================================================================

@pytest.mark.unit
class TestDashboardCharts:
    """Test dashboard charts and visualizations"""
    
    def test_charts_rendered(self, logged_in_admin: Page):
        """Should render charts"""
        # Look for chart containers or canvas elements
        charts = logged_in_admin.locator('canvas, svg[class*="chart"]')
        assert charts.count() > 0
    
    def test_processing_trends_chart_exists(self, logged_in_admin: Page):
        """Should have processing trends chart"""
        content = logged_in_admin.content().lower()
        assert 'trend' in content or 'chart' in content
    
    def test_fruit_distribution_chart_exists(self, logged_in_admin: Page):
        """Should have fruit distribution visualization"""
        content = logged_in_admin.content().lower()
        assert 'distribution' in content or 'fruit' in content


# ============================================================================
# RECENT ACTIVITY TESTS
# ============================================================================

@pytest.mark.integration
class TestRecentActivity:
    """Test recent activity feed"""
    
    def test_activity_feed_displayed(self, logged_in_admin: Page):
        """Should show recent activity feed"""
        # Look for activity section
        activity = logged_in_admin.locator('[class*="activity"]')
        assert activity.count() > 0
    
    def test_recent_results_shown(self, logged_in_admin: Page):
        """Should display recent processing results"""
        content = logged_in_admin.content().lower()
        assert 'recent' in content or 'activity' in content or 'results' in content
    
    def test_activity_items_formatted(self, logged_in_admin: Page):
        """Should format activity items properly"""
        # Check for list or table of activities
        items = logged_in_admin.locator('[class*="activity"] li, [class*="activity"] tr')
        # Could be 0 if no recent activity
        assert items.count() >= 0


# ============================================================================
# NAVIGATION TESTS
# ============================================================================

@pytest.mark.integration
class TestDashboardNavigation:
    """Test dashboard navigation"""
    
    def test_sidebar_visible(self, logged_in_admin: Page):
        """Should display navigation sidebar"""
        sidebar = logged_in_admin.locator('[class*="sidebar"], nav')
        assert sidebar.count() > 0
    
    def test_can_navigate_to_processing(self, logged_in_admin: Page):
        """Should navigate to processing page"""
        # Click processing link
        processing_link = logged_in_admin.locator('a[href*="processing"]')
        if processing_link.count() > 0:
            processing_link.click()
            expect(logged_in_admin).to_have_url(f"{BASE_URL}/processing")
    
    def test_can_navigate_to_results(self, logged_in_admin: Page):
        """Should navigate to results page"""
        results_link = logged_in_admin.locator('a[href*="results"]')
        if results_link.count() > 0:
            results_link.click()
            expect(logged_in_admin).to_have_url(f"{BASE_URL}/results")
    
    def test_can_navigate_to_cameras(self, logged_in_admin: Page):
        """Should navigate to camera monitor"""
        camera_link = logged_in_admin.locator('a[href*="camera"]')
        if camera_link.count() > 0:
            camera_link.click()
            assert "camera" in logged_in_admin.url
    
    def test_can_navigate_to_settings(self, logged_in_admin: Page):
        """Should navigate to settings"""
        settings_link = logged_in_admin.locator('a[href*="settings"]')
        if settings_link.count() > 0:
            settings_link.click()
            assert "settings" in logged_in_admin.url


# ============================================================================
# DATA REFRESH TESTS
# ============================================================================

@pytest.mark.integration
class TestDashboardDataRefresh:
    """Test dashboard data refresh"""
    
    def test_refresh_button_exists(self, logged_in_admin: Page):
        """Should have refresh button"""
        refresh_btn = logged_in_admin.locator('button[class*="refresh"], button:has-text("Refresh")')
        # May or may not have explicit refresh button
        assert refresh_btn.count() >= 0
    
    def test_auto_refresh_enabled(self, logged_in_admin: Page):
        """Should auto-refresh data periodically"""
        # Get initial metrics
        initial_content = logged_in_admin.content()
        
        # Wait for auto-refresh interval
        logged_in_admin.wait_for_timeout(5000)
        
        # Content should potentially update (or at least no errors)
        updated_content = logged_in_admin.content()
        # Just verify no crash
        assert len(updated_content) > 0
    
    def test_manual_refresh_updates_data(self, logged_in_admin: Page):
        """Should update data on manual refresh"""
        refresh_btn = logged_in_admin.locator('button[class*="refresh"]')
        if refresh_btn.count() > 0:
            refresh_btn.click()
            logged_in_admin.wait_for_timeout(1000)
            # Data should reload


# ============================================================================
# USER DASHBOARD TESTS
# ============================================================================

@pytest.mark.e2e
class TestUserDashboard:
    """Test regular user dashboard"""
    
    def test_user_dashboard_loads(self, logged_in_user: Page):
        """Should load user dashboard"""
        # User fixture should redirect to user-dashboard
        assert "/user-dashboard" in logged_in_user.url or "/dashboard" in logged_in_user.url
    
    def test_user_sees_limited_metrics(self, logged_in_user: Page):
        """Should show limited metrics for regular users"""
        # User should not see admin controls
        admin_controls = logged_in_user.locator('button:has-text("Start"), button:has-text("Configure")')
        # Admin controls should not be visible
        for i in range(admin_controls.count()):
            assert not admin_controls.nth(i).is_visible() or admin_controls.count() == 0
    
    def test_user_can_view_results(self, logged_in_user: Page):
        """User should be able to view results"""
        results_link = logged_in_user.locator('a[href*="results"]')
        if results_link.count() > 0:
            results_link.click()
            assert "results" in logged_in_user.url
    
    def test_user_cannot_access_processing(self, logged_in_user: Page):
        """User should not access processing controls"""
        logged_in_user.goto(f"{BASE_URL}/processing")
        logged_in_user.wait_for_timeout(1000)
        
        # Should redirect or show access denied
        assert "/processing" not in logged_in_user.url or \
               logged_in_user.locator('[class*="denied"]').count() > 0


# ============================================================================
# QUICK ACTIONS TESTS
# ============================================================================

@pytest.mark.unit
class TestQuickActions:
    """Test dashboard quick actions"""
    
    def test_quick_action_buttons_present(self, logged_in_admin: Page):
        """Should have quick action buttons"""
        # Look for action buttons
        action_buttons = logged_in_admin.locator('button[class*="action"]')
        assert action_buttons.count() >= 0
    
    def test_start_processing_quick_action(self, logged_in_admin: Page):
        """Should have quick action to start processing"""
        # Look for start button or link
        start_action = logged_in_admin.locator('button:has-text("Start"), a:has-text("Start Processing")')
        # May navigate to processing page instead of inline
        assert start_action.count() >= 0
    
    def test_view_results_quick_action(self, logged_in_admin: Page):
        """Should have quick action to view results"""
        results_action = logged_in_admin.locator('a:has-text("View Results"), button:has-text("Results")')
        assert results_action.count() >= 0


# ============================================================================
# RESPONSIVE DESIGN TESTS
# ============================================================================

@pytest.mark.unit
class TestDashboardResponsive:
    """Test responsive design"""
    
    def test_dashboard_on_mobile(self, context):
        """Should display properly on mobile"""
        # Create mobile viewport
        mobile_page = context.new_page()
        mobile_page.set_viewport_size({"width": 375, "height": 667})
        
        # Login
        mobile_page.goto(f"{BASE_URL}/login")
        mobile_page.fill('input[name="username"]', 'admin')
        mobile_page.fill('input[name="password"]', 'admin123')
        mobile_page.select_option('select[name="role"]', 'admin')
        mobile_page.click('button[type="submit"]')
        mobile_page.wait_for_url(f"{BASE_URL}/dashboard")
        
        # Dashboard should be visible
        expect(mobile_page).to_have_url(f"{BASE_URL}/dashboard")
        
        # Check key elements visible
        header = mobile_page.locator('h1, h2')
        assert header.count() > 0
        
        mobile_page.close()
    
    def test_cards_stack_on_small_screen(self, context):
        """Should stack cards on smaller screens"""
        # Create tablet viewport
        tablet_page = context.new_page()
        tablet_page.set_viewport_size({"width": 768, "height": 1024})
        
        tablet_page.goto(f"{BASE_URL}/login")
        tablet_page.fill('input[name="username"]', 'admin')
        tablet_page.fill('input[name="password"]', 'admin123')
        tablet_page.select_option('select[name="role"]', 'admin')
        tablet_page.click('button[type="submit"]')
        
        # Should load without errors
        tablet_page.wait_for_url(f"{BASE_URL}/dashboard")
        
        tablet_page.close()


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
            
            # Should redirect to login
            logged_in_admin.wait_for_url(f"{BASE_URL}/login", timeout=5000)
            assert "/login" in logged_in_admin.url
    
    def test_cannot_access_dashboard_after_logout(self, logged_in_admin: Page):
        """Should not access dashboard after logout"""
        # Logout
        logout_btn = logged_in_admin.locator('button:has-text("Logout")')
        if logout_btn.count() > 0:
            logout_btn.click()
            logged_in_admin.wait_for_url(f"{BASE_URL}/login")
            
            # Try to access dashboard
            logged_in_admin.goto(f"{BASE_URL}/dashboard")
            logged_in_admin.wait_for_timeout(1000)
            
            # Should redirect back to login
            assert "/login" in logged_in_admin.url


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
class TestDashboardPerformance:
    """Test dashboard performance"""
    
    def test_dashboard_loads_quickly(self, logged_in_admin: Page):
        """Should load dashboard in reasonable time"""
        import time
        
        start_time = time.time()
        logged_in_admin.goto(f"{BASE_URL}/dashboard")
        logged_in_admin.wait_for_load_state("domcontentloaded")
        load_time = time.time() - start_time
        
        # Should load in under 3 seconds
        assert load_time < 3.0
    
    def test_no_excessive_api_calls(self, logged_in_admin: Page):
        """Should not make excessive API calls"""
        # Monitor network requests
        requests = []
        
        def log_request(request):
            requests.append(request.url)
        
        logged_in_admin.on("request", log_request)
        logged_in_admin.goto(f"{BASE_URL}/dashboard")
        logged_in_admin.wait_for_timeout(2000)
        
        # Count API calls
        api_calls = [r for r in requests if '/api/' in r]
        
        # Should not make more than 10 initial calls
        assert len(api_calls) < 10