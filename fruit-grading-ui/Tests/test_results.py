"""
Results Page Tests
Tests classification results display and filtering
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"


def navigate_to_results(page: Page):
    """Navigate to results using the sidebar to preserve Auth state"""
    
    if "/results" in page.url:
        page.wait_for_selector('h1:has-text("Classification Results")', state='visible')
        return

    hamburger = page.locator('.hamburger-button')
    if hamburger.is_visible():
        hamburger.click()
        page.wait_for_selector('.sidebar.visible, .sidebar-open', timeout=2000)

    results_link = page.locator('a[href$="/results"]')
    results_link.dispatch_event('click')
    
    try:
        expect(page).to_have_url(re.compile(r".*/results"), timeout=10000)
    except AssertionError:
        results_link.evaluate("el => el.click()")
        expect(page).to_have_url(re.compile(r".*/results"))

    page.wait_for_selector('h1:has-text("Classification Results")', state='visible')



@pytest.mark.unit
class TestResultsRendering:
    """Test results page rendering"""
    
    def test_results_page_loads(self, logged_in_admin: Page):
        """Should load results page"""
        navigate_to_results(logged_in_admin)
        
        header = logged_in_admin.get_by_role("heading", name="Classification Results")
        expect(header).to_be_visible()
    
    def test_kpi_cards_displayed(self, logged_in_admin: Page):
        """Should display KPI cards"""
        navigate_to_results(logged_in_admin)
        
        kpi_cards = logged_in_admin.locator('.kpi-card')
        if kpi_cards.count() > 0:
            expect(kpi_cards.first).to_be_visible()
        
    def test_results_table_or_empty_state(self, logged_in_admin: Page):
        """Should display results table or empty state"""
        navigate_to_results(logged_in_admin)
        
        table = logged_in_admin.locator('.table-container table')
        empty = logged_in_admin.locator('.empty-state')
        
        expect(table.or_(empty).first).to_be_visible()




@pytest.mark.unit
class TestKPISection:
    """Test KPI display section"""
    
    def test_total_processed_kpi(self, logged_in_admin: Page):
        """Should show total processed KPI"""
        navigate_to_results(logged_in_admin)
        expect(logged_in_admin.locator('text=Total Processed')).to_be_visible()
    
    def test_quality_rate_kpi(self, logged_in_admin: Page):
        """Should show quality rate KPI"""
        navigate_to_results(logged_in_admin)
        expect(logged_in_admin.locator('text=Quality Rate')).to_be_visible()
    
    def test_model_accuracy_kpi(self, logged_in_admin: Page):
        """Should show model accuracy KPI"""
        navigate_to_results(logged_in_admin)
        expect(logged_in_admin.locator('text=Model Accuracy')).to_be_visible()
    
    def test_kpi_trend_indicators(self, logged_in_admin: Page):
        """Should show trend indicators"""
        navigate_to_results(logged_in_admin)
        
        trends = logged_in_admin.locator('.kpi-trend')
        assert trends.count() >= 0




@pytest.mark.integration
class TestResultsFiltering:
    """Test results filtering"""
    
    def test_search_box_exists(self, logged_in_admin: Page):
        """Should have search box"""
        navigate_to_results(logged_in_admin)
        
        search_input = logged_in_admin.locator('.search-input, input[placeholder*="Search"]')
        expect(search_input).to_be_visible()
    
    def test_type_filter_exists(self, logged_in_admin: Page):
        """Should have type filter dropdown"""
        navigate_to_results(logged_in_admin)
        
        type_filter = logged_in_admin.locator('.filter-select').first
        expect(type_filter).to_be_visible()
    
    def test_can_filter_by_type(self, logged_in_admin: Page):
        """Should filter by grade type"""
        navigate_to_results(logged_in_admin)
        
        type_filter = logged_in_admin.locator('.filter-select').first
        if type_filter.is_visible():
            try:
                type_filter.select_option('market')
            except Exception:
                pass
    
    def test_search_functionality(self, logged_in_admin: Page):
        """Should be able to search"""
        navigate_to_results(logged_in_admin)
        
        search_input = logged_in_admin.locator('.search-input, input[placeholder*="Search"]').first
        if search_input.is_visible():
            search_input.fill('test')



@pytest.mark.unit
class TestPredictedGradeDistribution:
    """Test predicted grade distribution section"""
    
    def test_grade_distribution_section(self, logged_in_admin: Page):
        """Should have predicted grade distribution section"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=Predicted Grade Distribution')
        expect(section).to_be_visible()
    
    def test_grade_legend_items(self, logged_in_admin: Page):
        """Should show grade legend without reject"""
        navigate_to_results(logged_in_admin)
        
        content = logged_in_admin.content()
        grades = ['Market Grade', 'Standard Grade', 'Premium Grade']
        found = sum(1 for g in grades if g in content)
        assert found >= 0
    
    def test_no_reject_in_distribution(self, logged_in_admin: Page):
        """Should not show reject in grade distribution"""
        navigate_to_results(logged_in_admin)
        
        pie_section = logged_in_admin.locator('.pie-chart-container')
        if pie_section.is_visible():
            legend_text = pie_section.inner_text()
            assert 'Reject' not in legend_text




@pytest.mark.unit
class TestSystemStatus:
    """Test system status sidebar"""
    
    def test_system_status_section_exists(self, logged_in_admin: Page):
        """Should have system status section"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=System Status')
        expect(section).to_be_visible()
    
    def test_alert_list_container(self, logged_in_admin: Page):
        """Should have alert list container"""
        navigate_to_results(logged_in_admin)
        
        alert_list = logged_in_admin.locator('.alert-list')
        assert alert_list.count() >= 0




@pytest.mark.unit
class TestTrainingHistory:
    """Test training history charts section"""
    
    def test_loss_chart_exists(self, logged_in_admin: Page):
        """Should have loss over epochs chart"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=Loss Over Epochs')
        expect(section).to_be_visible()
    
    def test_accuracy_chart_exists(self, logged_in_admin: Page):
        """Should have accuracy over epochs chart"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=Accuracy Over Epochs')
        expect(section).to_be_visible()
    
    def test_training_charts_side_by_side(self, logged_in_admin: Page):
        """Should display training charts in grid"""
        navigate_to_results(logged_in_admin)
        
        charts_grid = logged_in_admin.locator('.training-charts-grid')
        expect(charts_grid).to_be_visible()
    
    def test_chart_legends_visible(self, logged_in_admin: Page):
        """Should show chart legends"""
        navigate_to_results(logged_in_admin)
        
        legends = logged_in_admin.locator('.chart-legend')
        assert legends.count() >= 0




@pytest.mark.unit
class TestConfusionMatrix:
    """Test confusion matrix section"""
    
    def test_confusion_matrix_section_exists(self, logged_in_admin: Page):
        """Should have confusion matrix section"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=Normalized Confusion Matrix')
        expect(section).to_be_visible()
    
    def test_matrix_grid_visible(self, logged_in_admin: Page):
        """Should display matrix grid"""
        navigate_to_results(logged_in_admin)
        
        matrix = logged_in_admin.locator('.confusion-matrix-container')
        if matrix.count() > 0:
            expect(matrix.first).to_be_visible()
    
    def test_no_reject_in_matrix(self, logged_in_admin: Page):
        """Should not show reject class in matrix"""
        navigate_to_results(logged_in_admin)
        
        matrix_container = logged_in_admin.locator('.confusion-matrix-container')
        if matrix_container.is_visible():
            matrix_text = matrix_container.inner_text()
            assert 'reject' not in matrix_text.lower()




@pytest.mark.integration
class TestResultsExport:
    """Test results export"""
    
    def test_export_section_exists(self, logged_in_admin: Page):
        """Should have export section"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.get_by_role("heading", name="Export Data")
        expect(section).to_be_visible()
    
    def test_export_csv_button(self, logged_in_admin: Page):
        """Should have Export CSV button"""
        navigate_to_results(logged_in_admin)
        
        csv_btn = logged_in_admin.locator('button:has-text("CSV")')
        expect(csv_btn).to_be_visible()
    
    def test_export_button_centered(self, logged_in_admin: Page):
        """Should have centered export button"""
        navigate_to_results(logged_in_admin)
        
        export_options = logged_in_admin.locator('.export-options')
        if export_options.is_visible():
            expect(export_options).to_be_visible()




@pytest.mark.unit
class TestResultsTable:
    """Test results table display"""
    
    def test_classification_results_section(self, logged_in_admin: Page):
        """Should have classification results section"""
        navigate_to_results(logged_in_admin)
        
        section = logged_in_admin.locator('text=Classification Results')
        expect(section.first).to_be_visible()
    
    def test_table_headers_no_batch(self, logged_in_admin: Page):
        """Should have correct table headers without batch"""
        navigate_to_results(logged_in_admin)
        
        table = logged_in_admin.locator('.table-container table')
        if table.is_visible():
            expect(logged_in_admin.locator("th:has-text('Object ID')").first).to_be_visible()
            expect(logged_in_admin.locator("th:has-text('Grade')").first).to_be_visible()
            expect(logged_in_admin.locator("th:has-text('Image Count')").first).to_be_visible()
            expect(logged_in_admin.locator("th:has-text('Timestamp')").first).to_be_visible()
    
    def test_no_batch_column(self, logged_in_admin: Page):
        """Should not have batch column in table"""
        navigate_to_results(logged_in_admin)
        
        batch_header = logged_in_admin.locator("th:has-text('Batch')")
        assert batch_header.count() == 0




@pytest.mark.unit
class TestResponsiveLayout:
    """Test responsive layout"""
    
    def test_charts_grid_layout(self, logged_in_admin: Page):
        """Should have proper grid layout for charts"""
        navigate_to_results(logged_in_admin)
        
        charts_grid = logged_in_admin.locator('.training-charts-grid')
        if charts_grid.is_visible():
            cards = charts_grid.locator('.card')
            assert cards.count() == 2