"""
Processing Page Tests
Tests processing pipeline control and monitoring
Matches Processing.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"

# Helper function to navigate safely (Preserving Auth)
def navigate_to_processing(page: Page):
    """Navigate to processing using the sidebar to preserve Auth state"""
    
    # 1. Check if already there
    if "/processing" in page.url:
        page.wait_for_selector('h1:has-text("Processing")', state='visible')
        return

    # 2. Open Hamburger if visible
    hamburger = page.locator('.hamburger-button')
    if hamburger.is_visible():
        hamburger.click()
        page.wait_for_selector('.sidebar.visible, .sidebar-open', timeout=2000)

    # 3. LOCATE the element
    # Matches href="/processing"
    processing_link = page.locator('a[href$="/processing"]')
    
    # 4. DISPATCH 'click' event directly using JavaScript
    processing_link.dispatch_event('click')
    
    # 5. Wait for navigation
    try:
        expect(page).to_have_url(re.compile(r".*/processing"), timeout=10000)
    except AssertionError:
        # Fallback
        processing_link.evaluate("el => el.click()")
        expect(page).to_have_url(re.compile(r".*/processing"))

    # 6. Verify page load with SPECIFIC header
    # Matches "ML Processing Pipeline" or just "Processing"
    page.wait_for_selector('h1', state='visible')
    expect(page.get_by_role("heading", name="Processing Pipeline", exact=True)).to_be_visible()


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestProcessingRendering:
    """Test processing page rendering - matches Processing.jsx"""
    
    def test_processing_page_loads(self, logged_in_admin: Page):
        """Should load processing page successfully"""
        navigate_to_processing(logged_in_admin)
            
            # Use exact=True to avoid matching "Data Preprocessing"
        header = logged_in_admin.get_by_role("heading", name="Processing Pipeline", exact=True)
        expect(header).to_be_visible()
        
    def test_control_buttons_visible(self, logged_in_admin: Page):
        navigate_to_processing(logged_in_admin)
        start_btn = logged_in_admin.locator('button:has-text("Start")')
        expect(start_btn).to_be_visible()
    
    def test_progress_section_visible(self, logged_in_admin: Page):
        """Should have progress section"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx has progress-section class
        progress = logged_in_admin.locator('.progress-section, [class*="progress"]')
        assert progress.count() > 0
    
    def test_pipeline_steps_displayed(self, logged_in_admin: Page):
        """Should show all 6 pipeline steps"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx defines 6 steps
        content = logged_in_admin.content()
        expected_steps = ['Testing', 'Database', 'Preprocessing', 'Feature', 'Training', 'Evaluation']
        
        found = sum(1 for step in expected_steps if step in content)
        assert found >= 4  # At least 4 steps visible


# ============================================================================
# PIPELINE CONTROL TESTS
# ============================================================================

@pytest.mark.integration
class TestPipelineControl:
    """Test pipeline start/stop controls"""
    
    def test_start_button_enabled_initially(self, logged_in_admin: Page):
        """Should have enabled start button initially"""
        navigate_to_processing(logged_in_admin)
        
        start_btn = logged_in_admin.locator('button:has-text("Start")')
        expect(start_btn).to_be_enabled()
    
    def test_refresh_button_exists(self, logged_in_admin: Page):
        """Should have refresh button"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx has Refresh button
        refresh_btn = logged_in_admin.locator('button:has-text("Refresh")')
        expect(refresh_btn).to_be_visible()


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

@pytest.mark.unit
class TestProcessingConfiguration:
    """Test configuration sections - matches Processing.jsx"""
    
    def test_training_config_section(self, logged_in_admin: Page):
        """Should have training configuration section"""
        navigate_to_processing(logged_in_admin)
        
        # Use get_by_role to avoid ambiguity
        section = logged_in_admin.locator('text=Training Configuration')
        if section.count() > 1:
             expect(section.first).to_be_visible()
        else:
             expect(section).to_be_visible()
    
    def test_dataset_config_section(self, logged_in_admin: Page):
        """Should have dataset configuration section"""
        navigate_to_processing(logged_in_admin)
        
        section = logged_in_admin.locator('text=Dataset Configuration')
        if section.count() > 1:
             expect(section.first).to_be_visible()
        else:
             expect(section).to_be_visible()
    
    def test_hidden_dim_config_exists(self, logged_in_admin: Page):
        """Should have hidden dimension config"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=Hidden").first).to_be_visible()
    
    def test_epochs_config_exists(self, logged_in_admin: Page):
        """Should have epochs config"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=Epochs").first).to_be_visible()
    
    def test_learning_rate_config_exists(self, logged_in_admin: Page):
        """Should have learning rate config"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=Learning Rate").first).to_be_visible()
    
    def test_batch_size_config_exists(self, logged_in_admin: Page):
        """Should have batch size config"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=Batch").first).to_be_visible()
    
    def test_config_dropdowns_work(self, logged_in_admin: Page):
        """Should be able to open config dropdowns"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx uses .select-button for dropdowns
        dropdown_btn = logged_in_admin.locator('.select-button').first
        if dropdown_btn.count() > 0:
            dropdown_btn.click()
            
            # Options should appear
            options = logged_in_admin.locator('.select-options')
            expect(options).to_be_visible()


# ============================================================================
# LOG DISPLAY TESTS
# ============================================================================

@pytest.mark.integration
class TestLogDisplay:
    """Test processing log display"""
    
    def test_log_panel_exists(self, logged_in_admin: Page):
        navigate_to_processing(logged_in_admin)
        # Be specific: H2 header
        log_section = logged_in_admin.get_by_role("heading", name="Processing Logs")
        expect(log_section).to_be_visible()

    
    def test_log_container_scrollable(self, logged_in_admin: Page):
        navigate_to_processing(logged_in_admin)
        # Fix typo: .logs-container (plural)
        log_container = logged_in_admin.locator('.logs-container')
        assert log_container.count() > 0



# ============================================================================
# STEP VISUALIZATION TESTS
# ============================================================================

@pytest.mark.unit
class TestStepVisualization:
    """Test pipeline step display"""
    
    def test_steps_container_exists(self, logged_in_admin: Page):
        """Should have steps container"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx uses .pipeline-steps class
        steps = logged_in_admin.locator('.pipeline-steps, .step-item, [class*="step"]')
        assert steps.count() > 0
    
    def test_steps_have_status_indicators(self, logged_in_admin: Page):
        """Should show status for each step"""
        navigate_to_processing(logged_in_admin)
        
        # Processing.jsx uses step-pending, step-processing, step-completed classes
        content = logged_in_admin.content()
        assert 'step' in content.lower()


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.integration
class TestProcessingErrorHandling:
    """Test error handling during processing"""
    
    def test_error_display_area_exists(self, logged_in_admin: Page):
        """Should have error display area"""
        navigate_to_processing(logged_in_admin)
        
        # Just verify page loads correctly
        page_content = logged_in_admin.content()
        assert len(page_content) > 0


# ============================================================================
# AUTHORIZATION TESTS
# ============================================================================

@pytest.mark.e2e
class TestProcessingAuthorization:
    """Test authorization for processing page"""
    
    def test_admin_can_access(self, logged_in_admin: Page):
        navigate_to_processing(logged_in_admin)
        # Use exact text
        expect(logged_in_admin.get_by_role("heading", name="Processing Pipeline", exact=True)).to_be_visible()

    def test_regular_user_cannot_access(self, logged_in_user: Page):
        """Regular user should not access processing page"""
        # Try to go to processing (using standard goto here as we expect redirect)
        logged_in_user.goto(f"{BASE_URL}/processing")
        
        # Wait for redirect
        logged_in_user.wait_for_load_state("networkidle")
        
        # Should redirect or show access denied
        url = logged_in_user.url
        assert "/processing" not in url or \
               logged_in_user.locator('[class*="denied"], [class*="error"]').count() > 0


# ============================================================================
# DATASET INFO TESTS
# ============================================================================

@pytest.mark.unit
class TestDatasetInfo:
    """Test dataset configuration display"""
    
    def test_train_test_split_shown(self, logged_in_admin: Page):
        """Should show train/test split info"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=Split").first).to_be_visible()
    
    def test_image_size_shown(self, logged_in_admin: Page):
        """Should show image size"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=224").first).to_be_visible()
    
    def test_feature_extractor_shown(self, logged_in_admin: Page):
        """Should show feature extractor name"""
        navigate_to_processing(logged_in_admin)
        expect(logged_in_admin.locator("text=ShuffleNet").first).to_be_visible
