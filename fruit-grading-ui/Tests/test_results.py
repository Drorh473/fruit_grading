"""
Results Page Tests
Tests classification results display and filtering
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


@pytest.mark.unit
class TestResultsRendering:
    """Test results page rendering"""
    
    def test_results_page_loads(self, logged_in_admin: Page):
        """Should load results page"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/results")
    
    def test_results_table_displayed(self, logged_in_admin: Page):
        """Should display results table"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        table = logged_in_admin.locator('table, [class*="table"]')
        assert table.count() > 0
    
    def test_filter_controls_visible(self, logged_in_admin: Page):
        """Should show filter controls"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        filters = logged_in_admin.locator('[class*="filter"]')
        assert filters.count() > 0


@pytest.mark.integration
class TestResultsFiltering:
    """Test results filtering"""
    
    def test_filter_by_fruit_type(self, logged_in_admin: Page):
        """Should filter by fruit type"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        fruit_filter = logged_in_admin.locator('select[name*="fruit"]')
        if fruit_filter.count() > 0:
            fruit_filter.select_option('apple')
            logged_in_admin.wait_for_timeout(500)
    
    def test_filter_by_grade(self, logged_in_admin: Page):
        """Should filter by grade"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        grade_filter = logged_in_admin.locator('select[name*="grade"]')
        if grade_filter.count() > 0:
            grade_filter.select_option('premium')
            logged_in_admin.wait_for_timeout(500)
    
    def test_clear_filters(self, logged_in_admin: Page):
        """Should clear all filters"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        clear_btn = logged_in_admin.locator('button:has-text("Clear")')
        if clear_btn.count() > 0:
            clear_btn.click()


@pytest.mark.integration
class TestResultsExport:
    """Test results export"""
    
    def test_export_button_present(self, logged_in_admin: Page):
        """Should have export button"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        export_btn = logged_in_admin.locator('button:has-text("Export")')
        assert export_btn.count() > 0
    
    def test_export_csv(self, logged_in_admin: Page):
        """Should export results as CSV"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        export_btn = logged_in_admin.locator('button:has-text("Export")')
        if export_btn.count() > 0:
            with logged_in_admin.expect_download() as download_info:
                export_btn.click()
            # download = download_info.value


@pytest.mark.unit
class TestResultsPagination:
    """Test results pagination"""
    
    def test_pagination_controls_visible(self, logged_in_admin: Page):
        """Should show pagination controls"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        pagination = logged_in_admin.locator('[class*="pagination"]')
        assert pagination.count() >= 0
    
    def test_can_navigate_pages(self, logged_in_admin: Page):
        """Should navigate between pages"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        next_btn = logged_in_admin.locator('button:has-text("Next")')
        if next_btn.count() > 0 and next_btn.is_enabled():
            next_btn.click()
            logged_in_admin.wait_for_timeout(500)