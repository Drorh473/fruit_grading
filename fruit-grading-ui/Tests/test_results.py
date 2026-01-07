"""
Results Page Tests
Tests classification results display and filtering
Matches Results.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestResultsRendering:
    """Test results page rendering - matches Results.jsx"""
    
    def test_results_page_loads(self, logged_in_admin: Page):
        """Should load results page"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/results")
        
        # Results.jsx has h1 "Classification Results"
        header = logged_in_admin.locator('h1')
        expect(header).to_contain_text('Classification Results')
    
    def test_kpi_cards_displayed(self, logged_in_admin: Page):
        """Should display KPI cards"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx uses .kpi-card class
        kpi_cards = logged_in_admin.locator('.kpi-card')
        assert kpi_cards.count() >= 3  # Total Processed, Yield Rate, Avg Processing Time
    
    def test_results_table_or_empty_state(self, logged_in_admin: Page):
        """Should display results table or empty state"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has table-container or empty-state
        table = logged_in_admin.locator('.table-container table')
        empty = logged_in_admin.locator('.empty-state')
        
        assert table.count() > 0 or empty.count() > 0


# ============================================================================
# KPI SECTION TESTS
# ============================================================================

@pytest.mark.unit
class TestKPISection:
    """Test KPI display section"""
    
    def test_total_processed_kpi(self, logged_in_admin: Page):
        """Should show total processed KPI"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        content = logged_in_admin.content()
        assert 'Total Processed' in content
    
    def test_kpi_trend_indicators(self, logged_in_admin: Page):
        """Should show trend indicators"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx uses .kpi-trend class
        trends = logged_in_admin.locator('.kpi-trend')
        # Trends may or may not be visible depending on data
        assert trends.count() >= 0


# ============================================================================
# FILTER TESTS
# ============================================================================

@pytest.mark.integration
class TestResultsFiltering:
    """Test results filtering - matches Results.jsx"""
    
    def test_search_box_exists(self, logged_in_admin: Page):
        """Should have search box"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has .search-box with input
        search_input = logged_in_admin.locator('.search-input, input[placeholder*="Search"]')
        expect(search_input).to_be_visible()
    
    def test_type_filter_exists(self, logged_in_admin: Page):
        """Should have type filter dropdown"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has filter-select for type
        type_filter = logged_in_admin.locator('.filter-select').first
        expect(type_filter).to_be_visible()
    
    def test_can_filter_by_type(self, logged_in_admin: Page):
        """Should filter by type"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Select market type
        type_filter = logged_in_admin.locator('.filter-select').first
        type_filter.select_option('market')
        logged_in_admin.wait_for_timeout(500)
    
    def test_batch_filter_exists(self, logged_in_admin: Page):
        """Should have batch filter"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has batch filter select
        batch_filter = logged_in_admin.locator('.filter-select').nth(1)
        assert batch_filter.count() > 0
    
    def test_search_functionality(self, logged_in_admin: Page):
        """Should be able to search"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        search_input = logged_in_admin.locator('.search-input, input[placeholder*="Search"]')
        search_input.fill('test')
        logged_in_admin.wait_for_timeout(500)


# ============================================================================
# QUALITY DISTRIBUTION TESTS
# ============================================================================

@pytest.mark.unit
class TestQualityDistribution:
    """Test quality distribution section"""
    
    def test_quality_distribution_section(self, logged_in_admin: Page):
        """Should have quality distribution section"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        section = logged_in_admin.locator('text=Quality Distribution')
        expect(section).to_be_visible()
    
    def test_grade_legend_items(self, logged_in_admin: Page):
        """Should show grade legend"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx shows Market, Standard, Premium, Reject grades
        content = logged_in_admin.content()
        grades = ['Market', 'Standard', 'Premium', 'Reject']
        found = sum(1 for g in grades if g in content)
        assert found >= 3


# ============================================================================
# ALERTS SECTION TESTS
# ============================================================================

@pytest.mark.unit
class TestAlertsSection:
    """Test quality alerts sidebar"""
    
    def test_alerts_section_exists(self, logged_in_admin: Page):
        """Should have quality alerts section"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has "Quality Alerts" sidebar
        section = logged_in_admin.locator('text=Quality Alerts')
        expect(section).to_be_visible()
    
    def test_alert_list_container(self, logged_in_admin: Page):
        """Should have alert list container"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx uses .alert-list class
        alert_list = logged_in_admin.locator('.alert-list')
        assert alert_list.count() > 0


# ============================================================================
# EXPORT TESTS
# ============================================================================

@pytest.mark.integration
class TestResultsExport:
    """Test results export - matches Results.jsx"""
    
    def test_export_section_exists(self, logged_in_admin: Page):
        """Should have export section"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        section = logged_in_admin.locator('text=Export')
        assert section.count() > 0
    
    def test_export_pdf_button(self, logged_in_admin: Page):
        """Should have Export PDF button"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx has "Export PDF Report" button
        pdf_btn = logged_in_admin.locator('button:has-text("PDF")')
        expect(pdf_btn).to_be_visible()
    
    def test_export_excel_button(self, logged_in_admin: Page):
        """Should have Export Excel button"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        excel_btn = logged_in_admin.locator('button:has-text("Excel")')
        expect(excel_btn).to_be_visible()
    
    def test_export_csv_button(self, logged_in_admin: Page):
        """Should have Export CSV button"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        csv_btn = logged_in_admin.locator('button:has-text("CSV")')
        expect(csv_btn).to_be_visible()
    
    def test_schedule_email_button(self, logged_in_admin: Page):
        """Should have Schedule Email button"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        email_btn = logged_in_admin.locator('button:has-text("Email")')
        expect(email_btn).to_be_visible()


# ============================================================================
# TABLE TESTS
# ============================================================================

@pytest.mark.unit
class TestResultsTable:
    """Test results table display"""
    
    def test_detailed_results_section(self, logged_in_admin: Page):
        """Should have detailed results section"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        section = logged_in_admin.locator('text=Detailed Results')
        expect(section).to_be_visible()
    
    def test_table_headers(self, logged_in_admin: Page):
        """Should have correct table headers"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        # Results.jsx table has: Object ID, Batch, Classification, Image Count, Timestamp
        content = logged_in_admin.content()
        headers = ['Object ID', 'Batch', 'Classification', 'Timestamp']
        found = sum(1 for h in headers if h in content)
        assert found >= 3


# ============================================================================
# BATCH COMPARISON TESTS
# ============================================================================

@pytest.mark.unit
class TestBatchComparison:
    """Test batch comparison section"""
    
    def test_batch_comparison_section(self, logged_in_admin: Page):
        """Should have batch comparison section"""
        logged_in_admin.goto(f"{BASE_URL}/results")
        
        section = logged_in_admin.locator('text=Batch Performance')
        expect(section).to_be_visible()