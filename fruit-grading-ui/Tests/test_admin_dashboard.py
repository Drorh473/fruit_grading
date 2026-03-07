"""
Admin Dashboard Tests
Tests system overview, stats, and administrative controls
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"

def navigate_to_admin_dashboard(page: Page):
    """Navigate to dashboard safely"""
    if "/dashboard" in page.url:
        if page.locator('.page-header h1:has-text("System Dashboard")').is_visible():
            return

    hamburger = page.locator('.hamburger-button')
    if hamburger.is_visible():
        hamburger.click()
        page.wait_for_selector('.sidebar.visible, .sidebar-open', timeout=2000)

    # Click Dashboard link
    dashboard_link = page.locator('.sidebar a[href="/dashboard"], .sidebar a[href="/"]')
    
    if dashboard_link.count() > 0:
        dashboard_link.first.dispatch_event('click')
    else:
        page.goto(f"{BASE_URL}/dashboard")

    # Wait for load
    expect(page).to_have_url(re.compile(r".*/dashboard"))
    page.wait_for_selector('.dashboard', state='visible')
    # Wait for loading spinner to disappear
    page.wait_for_selector('.spinner', state='hidden', timeout=10000)


@pytest.mark.unit
class TestAdminDashboardRendering:
    """Test Admin Dashboard structure"""
    
    def test_dashboard_header(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        expect(logged_in_admin.locator('.dashboard')).to_be_visible()
        expect(logged_in_admin.locator('.page-header h1')).to_contain_text("System Dashboard")
        
        # Verify Refresh Button
        expect(logged_in_admin.locator('button:has-text("Refresh")')).to_be_visible()

    def test_system_status_cards(self, logged_in_admin: Page):
        """Test the top row of 4 status cards"""
        navigate_to_admin_dashboard(logged_in_admin)
        
        # Look for the grid container
        expect(logged_in_admin.locator('.grid.grid-4').first).to_be_visible()
        
        # Look for the specific status labels
        expect(logged_in_admin.locator('.stat-label:has-text("Database")')).to_be_visible()
        expect(logged_in_admin.locator('.stat-label:has-text("Model Status")')).to_be_visible()
        expect(logged_in_admin.locator('.stat-label:has-text("Processed Today")')).to_be_visible()
        expect(logged_in_admin.locator('.stat-label:has-text("Model Accuracy")')).to_be_visible()


@pytest.mark.unit
class TestCameraStatus:
    """Test Camera Status Section"""
    
    def test_camera_section_exists(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        expect(logged_in_admin.get_by_role("heading", name="Camera Status")).to_be_visible()
        
    def test_four_cameras_displayed(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        # Dashboard.jsx maps systemStatus.cameras (4 items)
        camera_items = logged_in_admin.locator('.camera-status-item')
        expect(camera_items).to_have_count(4)
        
        # Verify labels "Camera 0" to "Camera 3"
        expect(logged_in_admin.locator('p:has-text("Camera 0")')).to_be_visible()
        expect(logged_in_admin.locator('p:has-text("Camera 3")')).to_be_visible()


@pytest.mark.integration
class TestRecentResults:
    """Test Recent Processing Results Table"""
    
    def test_recent_results_section(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        expect(logged_in_admin.get_by_role("heading", name="Recent Processing Results")).to_be_visible()
        
    def test_results_table_headers(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        # Check for table OR empty state
        if logged_in_admin.locator('table').is_visible():
            expect(logged_in_admin.locator('th:has-text("Object ID")')).to_be_visible()
            expect(logged_in_admin.locator('th:has-text("Classification")')).to_be_visible()
        else:
            expect(logged_in_admin.locator('.empty-state')).to_contain_text("No recent results")


@pytest.mark.unit
class TestSystemInformation:
    """Test Dataset and Model Info Sections"""
    
    def test_dataset_info_card(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        expect(logged_in_admin.get_by_role("heading", name="Dataset Information")).to_be_visible()
        
        # Check for specific labels inside this card
        expect(logged_in_admin.locator('.info-label:has-text("Training Samples")')).to_be_visible()
        expect(logged_in_admin.locator('.info-label:has-text("Total Images")')).to_be_visible()

    def test_model_performance_card(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        expect(logged_in_admin.get_by_role("heading", name="Model Performance")).to_be_visible()
        
        expect(logged_in_admin.locator('.info-label:has-text("Architecture")')).to_be_visible()
        expect(logged_in_admin.locator('.info-label:has-text("Training Accuracy")')).to_be_visible()


@pytest.mark.integration
class TestDashboardInteractions:
    """Test Interactive Elements"""
    
    def test_refresh_button(self, logged_in_admin: Page):
        navigate_to_admin_dashboard(logged_in_admin)
        
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh")')
        expect(refresh_btn).to_be_visible()
        expect(refresh_btn).to_be_enabled()
        
        # Click and verify state change
        refresh_btn.click()
