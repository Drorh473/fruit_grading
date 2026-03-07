"""
Login Page Tests
Tests authentication flow and login UI
"""

import re
import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"



@pytest.mark.unit
class TestLoginRendering:
    """Test login page rendering"""
    
    def test_login_page_loads(self, page: Page):
        """Should load login page successfully"""
        page.goto(f"{BASE_URL}/login")
        
        # Check page title contains Fruit Grading
        expect(page).to_have_title(re.compile(r"Fruit Grading"))
        
        # Check form elements exist
        expect(page.locator('input.login-input').first).to_be_visible()
        expect(page.locator('button.login-button')).to_be_visible()
    
    def test_role_selector_present(self, page: Page):
        """Should have role selector with admin and user options (radio buttons)"""
        page.goto(f"{BASE_URL}/login")
        
        # Login.jsx uses radio buttons for role, not select
        role_options = page.locator('.role-selector .role-option')
        expect(role_options).to_have_count(2)
        
        # Check radio inputs exist
        user_radio = page.locator('input[type="radio"][value="user"]')
        admin_radio = page.locator('input[type="radio"][value="admin"]')
        expect(user_radio).to_be_attached()
        expect(admin_radio).to_be_attached()
    
    def test_password_field_is_masked(self, page: Page):
        """Should render password as masked input"""
        page.goto(f"{BASE_URL}/login")
        
        password_input = page.locator('input[type="password"]')
        expect(password_input).to_be_visible()
    
    def test_logo_displayed(self, page: Page):
        """Should display logo"""
        page.goto(f"{BASE_URL}/login")
        
        logo = page.locator('.logo-container-login, .logo-icon-login')
        assert logo.count() > 0
    
    def test_demo_credentials_shown(self, page: Page):
        """Should display demo credentials in footer"""
        page.goto(f"{BASE_URL}/login")
        
        demo_section = page.locator('.demo-credentials')
        expect(demo_section).to_be_visible()



@pytest.mark.unit
class TestLoginInteractions:
    """Test form interactions"""
    
    def test_can_type_username(self, page: Page):
        """Should allow typing in username field"""
        page.goto(f"{BASE_URL}/login")
        
        username_input = page.locator('input[placeholder="Enter username"]')
        username_input.fill('testuser')
        assert username_input.input_value() == 'testuser'
    
    def test_can_type_password(self, page: Page):
        """Should allow typing in password field"""
        page.goto(f"{BASE_URL}/login")
        
        password_input = page.locator('input[placeholder="Enter password"]')
        password_input.fill('testpass123')
        assert password_input.input_value() == 'testpass123'
    
    def test_can_select_role(self, page: Page):
        """Should allow role selection via radio buttons"""
        page.goto(f"{BASE_URL}/login")
        
        admin_option = page.locator('.role-option:has(input[value="admin"])')
        admin_option.click()
        
        admin_radio = page.locator('input[type="radio"][value="admin"]')
        expect(admin_radio).to_be_checked()
    
    def test_submit_button_enabled(self, page: Page):
        """Should have enabled submit button"""
        page.goto(f"{BASE_URL}/login")
        
        submit_btn = page.locator('button.login-button')
        expect(submit_btn).to_be_enabled()
    
    def test_user_role_selected_by_default(self, page: Page):
        """Should have user role selected by default"""
        page.goto(f"{BASE_URL}/login")
        
        user_radio = page.locator('input[type="radio"][value="user"]')
        expect(user_radio).to_be_checked()



@pytest.mark.integration
class TestLoginValidation:
    """Test form validation"""
    
    def test_empty_fields_shows_error(self, page: Page):
        """Should show error when fields are empty"""
        page.goto(f"{BASE_URL}/login")
        
        page.click('button.login-button')
        
        error = page.locator('.login-error')
        expect(error).to_be_visible(timeout=2000)
    
    def test_invalid_credentials_show_error(self, page: Page):
        """Should show error for invalid credentials"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[placeholder="Enter username"]', 'wronguser')
        page.fill('input[placeholder="Enter password"]', 'wrongpass')
        page.click('button.login-button')
        
        error = page.locator('.login-error')
        expect(error).to_be_visible(timeout=2000)



@pytest.mark.e2e
@pytest.mark.auth
class TestLoginFlow:
    """Test complete authentication flow"""
    
    def test_admin_login_success(self, page: Page):
        """Should successfully login as admin"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[placeholder="Enter username"]', 'admin')
        page.fill('input[placeholder="Enter password"]', 'admin123')
        page.locator('.role-option:has(input[value="admin"])').click()
        page.click('button.login-button')
        
        page.wait_for_url(f"{BASE_URL}/dashboard", timeout=5000)
        assert "/dashboard" in page.url
    
    def test_user_login_success(self, page: Page):
        """Should successfully login as user"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[placeholder="Enter username"]', 'user')
        page.fill('input[placeholder="Enter password"]', 'user123')
        page.click('button.login-button')
        
        page.wait_for_url(f"{BASE_URL}/user-dashboard", timeout=5000)
        assert "/user-dashboard" in page.url
    
    def test_wrong_role_prevents_login(self, page: Page):
        """Should reject login with wrong role"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[placeholder="Enter username"]', 'admin')
        page.fill('input[placeholder="Enter password"]', 'admin123')
        # Leave user role selected
        page.click('button.login-button')
        
        error = page.locator('.login-error')
        expect(error).to_be_visible(timeout=2000)
        assert "/login" in page.url



@pytest.mark.unit
class TestLoginAccessibility:
    """Test accessibility features"""
    
    def test_labels_present(self, page: Page):
        """Should have proper labels for inputs"""
        page.goto(f"{BASE_URL}/login")
        
        labels = page.locator('.form-group label')
        assert labels.count() >= 2
    
    def test_submit_with_enter_key(self, page: Page):
        """Should submit form with Enter key"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[placeholder="Enter username"]', 'admin')
        page.fill('input[placeholder="Enter password"]', 'admin123')
        page.locator('.role-option:has(input[value="admin"])').click()
        page.keyboard.press('Enter')
        
        page.wait_for_url(f"{BASE_URL}/dashboard", timeout=5000)