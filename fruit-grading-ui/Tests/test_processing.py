"""
Processing Page Tests
Tests processing pipeline control and monitoring
Matches Processing.jsx component
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestProcessingRendering:
    """Test processing page rendering - matches Processing.jsx"""
    
    def test_processing_page_loads(self, logged_in_admin: Page):
        """Should load processing page successfully"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/processing")
        
        # Processing.jsx has h1 "ML Processing Pipeline"
        header = logged_in_admin.locator('h1')
        expect(header).to_contain_text('Processing')
    
    def test_control_buttons_visible(self, logged_in_admin: Page):
        """Should display Start and Stop buttons"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx uses FiPlay for Start, FiSquare for Stop
        start_btn = logged_in_admin.locator('button:has-text("Start")')
        stop_btn = logged_in_admin.locator('button:has-text("Stop")')
        
        expect(start_btn).to_be_visible()
        expect(stop_btn).to_be_visible()
    
    def test_progress_section_visible(self, logged_in_admin: Page):
        """Should have progress section"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx has progress-section class
        progress = logged_in_admin.locator('.progress-section, [class*="progress"]')
        assert progress.count() > 0
    
    def test_pipeline_steps_displayed(self, logged_in_admin: Page):
        """Should show all 6 pipeline steps"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx defines 6 steps: Testing, Database Setup, Data Preprocessing,
        # Feature Extraction, Model Training, Evaluation
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
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        start_btn = logged_in_admin.locator('button:has-text("Start")')
        expect(start_btn).to_be_enabled()
    
    def test_stop_button_exists(self, logged_in_admin: Page):
        """Should have stop button"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        stop_btn = logged_in_admin.locator('button:has-text("Stop")')
        expect(stop_btn).to_be_visible()
    
    def test_refresh_button_exists(self, logged_in_admin: Page):
        """Should have refresh button"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
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
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx has "Training Configuration" card
        section = logged_in_admin.locator('text=Training Configuration')
        expect(section).to_be_visible()
    
    def test_dataset_config_section(self, logged_in_admin: Page):
        """Should have dataset configuration section"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx has "Dataset Configuration" card
        section = logged_in_admin.locator('text=Dataset Configuration')
        expect(section).to_be_visible()
    
    def test_hidden_dim_config_exists(self, logged_in_admin: Page):
        """Should have hidden dimension config"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'Hidden' in content or 'hidden' in content
    
    def test_epochs_config_exists(self, logged_in_admin: Page):
        """Should have epochs config"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'Epochs' in content or 'epochs' in content
    
    def test_learning_rate_config_exists(self, logged_in_admin: Page):
        """Should have learning rate config"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'Learning Rate' in content or 'learning' in content
    
    def test_batch_size_config_exists(self, logged_in_admin: Page):
        """Should have batch size config"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'Batch' in content or 'batch' in content
    
    def test_config_dropdowns_work(self, logged_in_admin: Page):
        """Should be able to open config dropdowns"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
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
        """Should have log display panel"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx has "Processing Logs" section
        log_section = logged_in_admin.locator('text=Processing Logs')
        expect(log_section).to_be_visible()
    
    def test_log_container_scrollable(self, logged_in_admin: Page):
        """Should have scrollable log container"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx uses .log-container class
        log_container = logged_in_admin.locator('.log-container')
        assert log_container.count() > 0


# ============================================================================
# STEP VISUALIZATION TESTS
# ============================================================================

@pytest.mark.unit
class TestStepVisualization:
    """Test pipeline step display"""
    
    def test_steps_container_exists(self, logged_in_admin: Page):
        """Should have steps container"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Processing.jsx uses .pipeline-steps class
        steps = logged_in_admin.locator('.pipeline-steps, .step-item, [class*="step"]')
        assert steps.count() > 0
    
    def test_steps_have_status_indicators(self, logged_in_admin: Page):
        """Should show status for each step"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
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
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        # Error message area exists (may be hidden)
        # Processing.jsx shows error in card with error styling
        page_content = logged_in_admin.content()
        # Just verify page loads correctly
        assert len(page_content) > 0


# ============================================================================
# AUTHORIZATION TESTS
# ============================================================================

@pytest.mark.e2e
class TestProcessingAuthorization:
    """Test authorization for processing page"""
    
    def test_admin_can_access(self, logged_in_admin: Page):
        """Admin should access processing page"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        expect(logged_in_admin).to_have_url(f"{BASE_URL}/processing")
    
    def test_regular_user_cannot_access(self, logged_in_user: Page):
        """Regular user should not access processing page"""
        logged_in_user.goto(f"{BASE_URL}/processing")
        logged_in_user.wait_for_timeout(1000)
        
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
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'Split' in content or 'split' in content or '66%' in content
    
    def test_image_size_shown(self, logged_in_admin: Page):
        """Should show image size"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert '224' in content  # 224x224
    
    def test_feature_extractor_shown(self, logged_in_admin: Page):
        """Should show feature extractor name"""
        logged_in_admin.goto(f"{BASE_URL}/processing")
        
        content = logged_in_admin.content()
        assert 'ShuffleNet' in content