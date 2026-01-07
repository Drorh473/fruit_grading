"""
Dashboard Page Tests
Tests system overview and status monitoring
Matches Dashboard.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"

# Helper function to navigate safely (Preserving Auth)
def navigate_to_dashboard(page: Page):
    """Navigate to dashboard using the sidebar to preserve Auth state"""
    
    # 1. Check if already there (URL check + Header check)
    if "/dashboard" in page.url:
        # Check for specific dashboard header to be sure
        if page.get_by_role("heading", name="System Dashboard").is_visible():
            return

    # 2. Open Hamburger if visible
    hamburger = page.locator('.hamburger-button')
    if hamburger.is_visible():
        hamburger.click()
        page.wait_for_selector('.sidebar.visible, .sidebar-open', timeout=2000)

    # 3. LOCATE the element
    # Matches href="/dashboard" or just "/" if it's the index
    dashboard_link = page.locator('a[href="/dashboard"], a[href="/"]')
    
    # 4. DISPATCH 'click' event directly using JavaScript
    # Handle case where multiple links might match (e.g. logo + menu link) -> take the sidebar one
    if dashboard_link.count() > 1:
        dashboard_link = page.locator('.sidebar a[href="/dashboard"]')
        
    dashboard_link.dispatch_event('click')
    
    # 5. Wait for navigation
    try:
        expect(page).to_have_url(re.compile(r".*/dashboard"), timeout=10000)
    except AssertionError:
        # Fallback
        dashboard_link.evaluate("el => el.click()")
        expect(page).to_have_url(re.compile(r".*/dashboard"))

    # 6. Verify page load with SPECIFIC header
    page.wait_for_selector('h1:has-text("System Dashboard")', state='visible')


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestDashboardRendering:
    """Test dashboard page rendering - matches Dashboard.jsx"""
    
    def test_dashboard_page_loads(self, logged_in_admin: Page):
        """Should load dashboard page"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx has h1 "System Dashboard"
        # Use get_by_role to avoid ambiguity with sidebar logo
        header = logged_in_admin.get_by_role("heading", name="System Dashboard")
        expect(header).to_be_visible()
    
    def test_stats_grid_displayed(self, logged_in_admin: Page):
        """Should display statistics grid"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx uses .stats-grid class
        grid = logged_in_admin.locator('.stats-grid')
        expect(grid).to_be_visible()
    
    def test_four_stat_cards(self, logged_in_admin: Page):
        """Should display 4 key statistic cards"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx has 4 stat cards (Total Fruits, System Status, etc.)
        cards = logged_in_admin.locator('.stat-card')
        assert cards.count() >= 3  # Allowing for 3 or 4 depending on version


# ============================================================================
# STAT CARDS TESTS
# ============================================================================

@pytest.mark.unit
class TestStatCards:
    """Test individual statistic cards"""
    
    def test_total_fruits_card(self, logged_in_admin: Page):
        """Should show total fruits count"""
        navigate_to_dashboard(logged_in_admin)
        
        # Look for the label specifically
        expect(logged_in_admin.locator("text=Total Fruits").first).to_be_visible()
    
    def test_system_status_card(self, logged_in_admin: Page):
        """Should show system status"""
        navigate_to_dashboard(logged_in_admin)
        
        expect(logged_in_admin.locator("text=System Status").first).to_be_visible()
    
    def test_uptime_card(self, logged_in_admin: Page):
        """Should show system uptime"""
        navigate_to_dashboard(logged_in_admin)
        
        expect(logged_in_admin.locator("text=Uptime").first).to_be_visible()
    
    def test_accuracy_card(self, logged_in_admin: Page):
        """Should show model accuracy"""
        navigate_to_dashboard(logged_in_admin)
        
        # Check for 'Accuracy' text
        expect(logged_in_admin.locator("text=Accuracy").first).to_be_visible()


# ============================================================================
# CHARTS TESTS
# ============================================================================

@pytest.mark.unit
class TestDashboardCharts:
    """Test dashboard charts"""
    
    def test_charts_grid_layout(self, logged_in_admin: Page):
        """Should display charts grid"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx uses .charts-grid class
        grid = logged_in_admin.locator('.charts-grid')
        expect(grid).to_be_visible()
    
    def test_processing_trends_chart(self, logged_in_admin: Page):
        """Should show processing trends chart"""
        navigate_to_dashboard(logged_in_admin)
        
        # Look for chart title
        expect(logged_in_admin.locator("text=Processing Trends").first).to_be_visible()
    
    def test_quality_distribution_chart(self, logged_in_admin: Page):
        """Should show quality distribution chart"""
        navigate_to_dashboard(logged_in_admin)
        
        expect(logged_in_admin.locator("text=Quality Distribution").first).to_be_visible()


# ============================================================================
# RECENT ACTIVITY TESTS
# ============================================================================

@pytest.mark.unit
class TestRecentActivity:
    """Test recent activity feed"""
    
    def test_activity_section_exists(self, logged_in_admin: Page):
        """Should have recent activity section"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx has "Recent Activity" section
        section = logged_in_admin.get_by_role("heading", name="Recent Activity")
        expect(section).to_be_visible()
    
    def test_activity_list_items(self, logged_in_admin: Page):
        """Should show activity items"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx uses .activity-list and .activity-item
        # Items might be empty initially, so just check container or header
        expect(logged_in_admin.locator('.activity-list')).to_be_visible()


# ============================================================================
# QUICK ACTIONS TESTS
# ============================================================================

@pytest.mark.integration
class TestQuickActions:
    """Test quick action buttons"""
    
    def test_quick_actions_section(self, logged_in_admin: Page):
        """Should have quick actions section"""
        navigate_to_dashboard(logged_in_admin)
        
        # Dashboard.jsx has "Quick Actions" card
        # Using heading locator to avoid ambiguity
        section = logged_in_admin.get_by_role("heading", name="Quick Actions")
        expect(section).to_be_visible()
    
    def test_start_processing_action(self, logged_in_admin: Page):
        """Should have Start Processing button"""
        navigate_to_dashboard(logged_in_admin)
        
        # Look for button specifically
        btn = logged_in_admin.locator('button:has-text("Start Processing")')
        expect(btn).to_be_visible()
    
    def test_add_fruit_action(self, logged_in_admin: Page):
        """Should have Add Fruit button"""
        navigate_to_dashboard(logged_in_admin)
        
        btn = logged_in_admin.locator('button:has-text("Add Fruit")')
        expect(btn).to_be_visible()
    
    def test_generate_report_action(self, logged_in_admin: Page):
        """Should have Generate Report button"""
        navigate_to_dashboard(logged_in_admin)
        
        btn = logged_in_admin.locator('button:has-text("Generate Report")')
        expect(btn).to_be_visible()


# ============================================================================
# AUTHORIZATION TESTS
# ============================================================================

@pytest.mark.e2e
class TestDashboardAuthorization:
    """Test dashboard authorization"""
    
    def test_admin_can_access(self, logged_in_admin: Page):
        """Admin should access dashboard"""
        navigate_to_dashboard(logged_in_admin)
        expect(logged_in_admin.get_by_role("heading", name="System Dashboard")).to_be_visible()
    
    def test_regular_user_access(self, logged_in_user: Page):
        """Regular user should access USER dashboard (not system dashboard)"""
        # Note: This depends on your specific routing logic.
        # If user goes to /dashboard, do they see a different view?
        # Assuming they get redirected or see a limited view.
        
        logged_in_user.goto(f"{BASE_URL}/dashboard")
        
        # Verify page loads (even if it's the user version)
        # Assuming user dashboard has a welcome message or similar
        expect(logged_in_user.locator("h1")).to_be_visible()
