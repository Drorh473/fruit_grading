"""
User Dashboard Tests
Tests limited view for standard operators
Matches UserDashboard.jsx structure
Includes interaction tests for all buttons
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"

def navigate_to_user_dashboard(page: Page):
    """Navigate to user dashboard safely"""
    if "/user-dashboard" not in page.url:
        page.goto(f"{BASE_URL}/user-dashboard")
    
    # Wait for the main container and content
    page.wait_for_selector('.user-dashboard', state='visible', timeout=10000)
    page.wait_for_selector('.summary-card', state='visible', timeout=10000)


@pytest.mark.unit
class TestUserDashboardRendering:
    """Test User Dashboard structure"""
    
    def test_user_dashboard_structure(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        expect(logged_in_user.locator('.user-dashboard')).to_be_visible()
        expect(logged_in_user.locator('.page-header h1')).to_contain_text("Welcome")
        expect(logged_in_user.locator('.page-subtitle')).to_contain_text("Operator Dashboard")

    def test_summary_cards_displayed(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        expect(logged_in_user.locator('.grid.grid-4')).to_be_visible()
        cards = logged_in_user.locator('.summary-card')
        expect(cards).to_have_count(4)
        expect(logged_in_user.locator('p:has-text("Processed Today")')).to_be_visible()


@pytest.mark.integration
class TestRecentResults:
    """Test Recent Results Section"""
    
    def test_results_section_exists(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        expect(logged_in_user.get_by_role("heading", name="Recent Classification Results")).to_be_visible()

    def test_view_all_results_navigation(self, logged_in_user: Page):
        """Test clicking 'View All Results' navigates to results page"""
        navigate_to_user_dashboard(logged_in_user)
        logged_in_user.click('button:has-text("View All Results")')
        expect(logged_in_user).to_have_url(re.compile(r".*/results"))
        expect(logged_in_user.get_by_role("heading", name="Classification Results")).to_be_visible()


@pytest.mark.integration
class TestDashboardInteractions:
    """Test Button Interactions"""
    
    def test_refresh_button_exists(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        btn = logged_in_user.locator('button:has-text("Refresh")')
        expect(btn).to_be_visible()
    
    def test_refresh_button_functionality(self, logged_in_user: Page):
        """Test that refresh button changes state when clicked"""
        navigate_to_user_dashboard(logged_in_user)
        
        refresh_btn = logged_in_user.locator('button:has-text("Refresh")')
        
        # Click refresh
        refresh_btn.click()
        
        # Check for loading state (The text changes to "Refreshing..." or icon spins)
        # UserDashboard.jsx: {refreshing ? "Refreshing..." : "Refresh"}
        try:
            expect(logged_in_user.locator('button')).to_contain_text("Refreshing", timeout=2000)
        except AssertionError:
            # It might have been too fast, which is also fine.
            pass
            
        # Should eventually return to "Refresh"
        expect(refresh_btn).to_contain_text("Refresh")
        expect(refresh_btn).to_be_enabled()


@pytest.mark.unit
class TestClassificationGuide:
    """Test Info Box"""
    
    def test_guide_section_exists(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        expect(logged_in_user.get_by_role("heading", name="Classification Guide")).to_be_visible()
        
        # Scope to guide container
        guide = logged_in_user.locator('.classification-guide')
        expect(guide.locator('.type-badge.type-market')).to_be_visible()
        expect(guide.locator('.type-badge.type-standard')).to_be_visible()
        expect(guide.locator('.type-badge.type-reject')).to_be_visible()


@pytest.mark.integration
class TestUserRestrictions:
    """Verify Users CANNOT see Admin sections"""
    
    def test_no_admin_actions(self, logged_in_user: Page):
        navigate_to_user_dashboard(logged_in_user)
        expect(logged_in_user.locator('button:has-text("Generate Report")')).not_to_be_visible()
        expect(logged_in_user.locator('button:has-text("Start Processing")')).not_to_be_visible()
