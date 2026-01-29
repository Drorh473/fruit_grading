"""
Tests for Add Fruit Page
Tests the fruit upload and classification workflow
"""

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:3000"


class TestAddFruitPageLoad:
    """Test Add Fruit page loading and initial state"""

    def test_add_fruit_page_loads(self, logged_in_admin):
        """Test that Add Fruit page loads correctly"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Check page title
        expect(page.locator('h1')).to_contain_text('Add New Fruit')

    def test_add_fruit_shows_instructions(self, logged_in_admin):
        """Test that instructions are displayed"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Check instruction card exists
        instruction_card = page.locator('.instruction-card')
        expect(instruction_card).to_be_visible()

        # Check step numbers are visible
        expect(page.locator('.step-number').first).to_be_visible()

    def test_add_fruit_shows_folder_selection(self, logged_in_admin):
        """Test that folder selection area is displayed"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Check folder selection section
        expect(page.locator('text=Folder Selection')).to_be_visible()

        # Check browse button exists
        browse_button = page.locator('button:has-text("Browse")')
        expect(browse_button).to_be_visible()


class TestAddFruitInitialState:
    """Test initial state of Add Fruit page"""

    def test_add_fruit_button_disabled_initially(self, logged_in_admin):
        """Test that Add Fruit button is disabled when no folder selected"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Find the add fruit button
        add_button = page.locator('button:has-text("Add Fruit")')
        expect(add_button).to_be_disabled()

    def test_no_folder_selected_initially(self, logged_in_admin):
        """Test that no folder is selected initially"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Check the folder input shows placeholder
        folder_input = page.locator('input[placeholder="No folder selected"]')
        expect(folder_input).to_have_value('')

    def test_no_validation_message_initially(self, logged_in_admin):
        """Test that no validation message is shown initially"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Validation result should not be visible
        validation = page.locator('.validation-result')
        expect(validation).not_to_be_visible()


class TestAddFruitUIElements:
    """Test UI elements and interactions"""

    def test_browse_button_clickable(self, logged_in_admin):
        """Test that browse button is clickable"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        browse_button = page.locator('button:has-text("Browse")')
        expect(browse_button).to_be_enabled()

    def test_folder_input_is_readonly(self, logged_in_admin):
        """Test that folder input is read-only"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        folder_input = page.locator('input[placeholder="No folder selected"]')
        expect(folder_input).to_have_attribute('readonly', '')

    def test_page_has_card_layout(self, logged_in_admin):
        """Test that page uses card layout"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Should have multiple cards
        cards = page.locator('.card')
        expect(cards.first).to_be_visible()


class TestAddFruitErrorDisplay:
    """Test error display functionality"""

    def test_error_card_hidden_initially(self, logged_in_admin):
        """Test that error card is hidden when no errors"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Error card should not be visible initially
        error_card = page.locator('[style*="var(--error)"]')
        expect(error_card).not_to_be_visible()


class TestAddFruitNavigation:
    """Test navigation from Add Fruit page"""

    def test_can_navigate_to_add_fruit_from_sidebar(self, logged_in_admin):
        """Test navigation to Add Fruit via sidebar"""
        page = logged_in_admin

        # Click on Add Fruit in sidebar
        add_fruit_link = page.locator('a[href="/add-fruit"]')
        if add_fruit_link.is_visible():
            add_fruit_link.click()
            page.wait_for_url(f"{BASE_URL}/add-fruit")
            expect(page.locator('h1')).to_contain_text('Add New Fruit')


class TestAddFruitInstructions:
    """Test instruction content"""

    def test_instruction_step_1_content(self, logged_in_admin):
        """Test step 1 instruction content"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Check step 1 mentions image preparation
        step1 = page.locator('.instruction-step').first
        expect(step1).to_contain_text('Prepare')

    def test_instruction_mentions_angle_folders(self, logged_in_admin):
        """Test instructions mention angle folders"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Should mention angle folders (an_0, an_1, etc.)
        expect(page.locator('.instructions')).to_contain_text('an_0')

    def test_three_instruction_steps_exist(self, logged_in_admin):
        """Test that all 3 instruction steps are shown"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        step_numbers = page.locator('.step-number')
        expect(step_numbers).to_have_count(3)


class TestAddFruitDragAndDrop:
    """Test drag and drop functionality"""

    def test_drag_overlay_not_visible_initially(self, logged_in_admin):
        """Test drag overlay is hidden initially"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # The drag overlay text should not be visible
        drop_text = page.locator('text=Drop Folder Here')
        expect(drop_text).not_to_be_visible()


class TestAddFruitResultDisplay:
    """Test result display after processing (mocked)"""

    def test_result_card_hidden_initially(self, logged_in_admin):
        """Test result card is not shown before processing"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Result card should not exist
        result_card = page.locator('.result-card-simple')
        expect(result_card).not_to_be_visible()

    def test_processing_spinner_hidden_initially(self, logged_in_admin):
        """Test processing spinner is not shown initially"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Processing section should not be visible
        processing = page.locator('text=Processing Pipeline')
        expect(processing).not_to_be_visible()


class TestAddFruitAccessibility:
    """Test accessibility features"""

    def test_browse_button_has_icon(self, logged_in_admin):
        """Test browse button has folder icon"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Button should have icon (svg element inside)
        browse_button = page.locator('button:has-text("Browse")')
        expect(browse_button.locator('svg')).to_be_visible()

    def test_input_has_icon(self, logged_in_admin):
        """Test folder input has associated icon"""
        page = logged_in_admin
        page.goto(f"{BASE_URL}/add-fruit")
        page.wait_for_load_state('networkidle')

        # Input group should have icon
        input_group = page.locator('.input-group-inline')
        expect(input_group.locator('.input-icon')).to_be_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
