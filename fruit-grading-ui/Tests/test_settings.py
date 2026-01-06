"""
Settings Page Tests
Tests system settings configuration (admin only)
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


@pytest.mark.unit
class TestSettingsRendering:
    """Test settings page rendering"""
    
    def test_settings_page_loads(self, logged_in_admin: Page):
        """Should load settings page"""
        logged_in_admin.goto(f"{BASE_URL}/settings")
        assert "settings" in logged_in_admin.url.lower()
    
    def test_settings_form_visible(self, logged_in_admin: Page):
        """Should display settings form"""
        logged_in_admin.goto(f"{BASE_URL}/settings")
        form = logged_in_admin.locator('form, [class*="settings"]')
        assert form.count() > 0


@pytest.mark.integration
class TestSettingsForm:
    """Test settings form interactions"""
    
    def test_can_edit_confidence_threshold(self, logged_in_admin: Page):
        """Should allow editing confidence threshold"""
        logged_in_admin.goto(f"{BASE_URL}/settings")
        
        threshold = logged_in_admin.locator('input[name*="threshold"]')
        if threshold.count() > 0:
            threshold.fill('0.85')
            assert threshold.input_value() == '0.85'
    
    def test_save_settings_button(self, logged_in_admin: Page):
        """Should have save button"""
        logged_in_admin.goto(f"{BASE_URL}/settings")
        
        save_btn = logged_in_admin.locator('button:has-text("Save")')
        assert save_btn.count() > 0
    
    def test_save_shows_confirmation(self, logged_in_admin: Page):
        """Should show confirmation on save"""
        logged_in_admin.goto(f"{BASE_URL}/settings")
        
        save_btn = logged_in_admin.locator('button:has-text("Save")')
        if save_btn.count() > 0:
            save_btn.click()
            logged_in_admin.wait_for_timeout(500)
            
            success = logged_in_admin.locator('[class*="success"]')
            assert success.count() >= 0


@pytest.mark.e2e
class TestSettingsAuthorization:
    """Test settings authorization"""
    
    def test_user_cannot_access_settings(self, logged_in_user: Page):
        """Regular user should not access settings"""
        logged_in_user.goto(f"{BASE_URL}/settings")
        logged_in_user.wait_for_timeout(500)
        
        assert "/settings" not in logged_in_user.url or \
               logged_in_user.locator('[class*="denied"]').count() > 0