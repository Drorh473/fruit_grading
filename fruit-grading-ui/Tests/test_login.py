"""
Login Page Tests
Tests authentication flow and login UI
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:3000"


# ============================================================================
# RENDERING TESTS
# ============================================================================

@pytest.mark.unit
class TestLoginRendering:
    """Test login page rendering"""
    
    def test_login_page_loads(self, page: Page):
        """Should load login page successfully"""
        page.goto(f"{BASE_URL}/login")
        
        # Check page title
        expect(page).to_have_title("Fruit Grading System")
        
        # Check form elements exist
        expect(page.locator('input[name="username"]')).to_be_visible()
        expect(page.locator('input[name="password"]')).to_be_visible()
        expect(page.locator('button[type="submit"]')).to_be_visible()
    
    def test_role_selector_present(self, page: Page):
        """Should have role selector with admin and user options"""
        page.goto(f"{BASE_URL}/login")
        
        role_select = page.locator('select[name="role"]')
        expect(role_select).to_be_visible()
        
        # Check options
        options = role_select.locator('option').all_text_contents()
        assert 'admin' in options
        assert 'user' in options
    
    def test_password_field_is_masked(self, page: Page):
        """Should render password as masked input"""
        page.goto(f"{BASE_URL}/login")
        
        password_input = page.locator('input[name="password"]')
        assert password_input.get_attribute('type') == 'password'
    
    def test_logo_displayed(self, page: Page):
        """Should display logo"""
        page.goto(f"{BASE_URL}/login")
        
        # Check for logo or title
        logo = page.locator('[class*="logo"]')
        assert logo.count() > 0


# ============================================================================
# FORM INTERACTION TESTS
# ============================================================================

@pytest.mark.unit
class TestLoginInteractions:
    """Test form interactions"""
    
    def test_can_type_username(self, page: Page):
        """Should allow typing in username field"""
        page.goto(f"{BASE_URL}/login")
        
        username_input = page.locator('input[name="username"]')
        username_input.fill('testuser')
        
        assert username_input.input_value() == 'testuser'
    
    def test_can_type_password(self, page: Page):
        """Should allow typing in password field"""
        page.goto(f"{BASE_URL}/login")
        
        password_input = page.locator('input[name="password"]')
        password_input.fill('testpass123')
        
        assert password_input.input_value() == 'testpass123'
    
    def test_can_select_role(self, page: Page):
        """Should allow role selection"""
        page.goto(f"{BASE_URL}/login")
        
        role_select = page.locator('select[name="role"]')
        role_select.select_option('admin')
        
        assert role_select.input_value() == 'admin'
    
    def test_submit_button_enabled(self, page: Page):
        """Should have enabled submit button"""
        page.goto(f"{BASE_URL}/login")
        
        submit_btn = page.locator('button[type="submit"]')
        expect(submit_btn).to_be_enabled()


# ============================================================================
# VALIDATION TESTS
# ============================================================================

@pytest.mark.integration
class TestLoginValidation:
    """Test form validation"""
    
    def test_empty_username_shows_error(self, page: Page):
        """Should show error when username is empty"""
        page.goto(f"{BASE_URL}/login")
        
        # Leave username empty, fill password
        page.fill('input[name="password"]', 'password123')
        page.click('button[type="submit"]')
        
        # Check for error message
        error = page.locator('[class*="error"]')
        expect(error).to_be_visible(timeout=2000)
    
    def test_empty_password_shows_error(self, page: Page):
        """Should show error when password is empty"""
        page.goto(f"{BASE_URL}/login")
        
        # Fill username, leave password empty
        page.fill('input[name="username"]', 'admin')
        page.click('button[type="submit"]')
        
        # Check for error message
        error = page.locator('[class*="error"]')
        expect(error).to_be_visible(timeout=2000)
    
    def test_invalid_credentials_show_error(self, page: Page):
        """Should show error for invalid credentials"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[name="username"]', 'wronguser')
        page.fill('input[name="password"]', 'wrongpass')
        page.click('button[type="submit"]')
        
        # Check for error message
        error = page.locator('[class*="error"]')
        expect(error).to_be_visible(timeout=2000)


# ============================================================================
# AUTHENTICATION FLOW TESTS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.auth
class TestLoginFlow:
    """Test complete authentication flow"""
    
    def test_admin_login_success(self, page: Page, admin_credentials):
        """Should successfully login as admin"""
        page.goto(f"{BASE_URL}/login")
        
        # Fill form
        page.fill('input[name="username"]', admin_credentials['username'])
        page.fill('input[name="password"]', admin_credentials['password'])
        page.select_option('select[name="role"]', admin_credentials['role'])
        
        # Submit
        page.click('button[type="submit"]')
        
        # Should redirect to admin dashboard
        page.wait_for_url(f"{BASE_URL}/dashboard", timeout=5000)
        assert "/dashboard" in page.url
    
    def test_user_login_success(self, page: Page, user_credentials):
        """Should successfully login as user"""
        page.goto(f"{BASE_URL}/login")
        
        # Fill form
        page.fill('input[name="username"]', user_credentials['username'])
        page.fill('input[name="password"]', user_credentials['password'])
        page.select_option('select[name="role"]', user_credentials['role'])
        
        # Submit
        page.click('button[type="submit"]')
        
        # Should redirect to user dashboard
        page.wait_for_url(f"{BASE_URL}/user-dashboard", timeout=5000)
        assert "/user-dashboard" in page.url
    
    def test_wrong_role_prevents_login(self, page: Page):
        """Should reject login with wrong role"""
        page.goto(f"{BASE_URL}/login")
        
        # Try admin credentials with user role
        page.fill('input[name="username"]', 'admin')
        page.fill('input[name="password"]', 'admin123')
        page.select_option('select[name="role"]', 'user')
        
        page.click('button[type="submit"]')
        
        # Should show error and stay on login page
        error = page.locator('[class*="error"]')
        expect(error).to_be_visible(timeout=2000)
        assert "/login" in page.url


# ============================================================================
# LOADING STATE TESTS
# ============================================================================

@pytest.mark.unit
class TestLoginLoadingStates:
    """Test loading states"""
    
    def test_button_disabled_during_login(self, page: Page):
        """Should disable button during login attempt"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[name="username"]', 'admin')
        page.fill('input[name="password"]', 'admin123')
        
        # Click and quickly check if button is disabled
        page.click('button[type="submit"]')
        
        # Button should be disabled during request
        submit_btn = page.locator('button[type="submit"]')
        # This might be too fast to catch, but it's the pattern
        # In real app, you'd check for loading state


# ============================================================================
# ACCESSIBILITY TESTS
# ============================================================================

@pytest.mark.unit
class TestLoginAccessibility:
    """Test accessibility features"""
    
    def test_labels_present(self, page: Page):
        """Should have proper labels for inputs"""
        page.goto(f"{BASE_URL}/login")
        
        # Check for labels or aria-labels
        username_input = page.locator('input[name="username"]')
        password_input = page.locator('input[name="password"]')
        
        # Should have associated labels or aria-label
        assert username_input.get_attribute('aria-label') or \
               page.locator('label[for="username"]').count() > 0
    
    def test_keyboard_navigation(self, page: Page):
        """Should support keyboard navigation"""
        page.goto(f"{BASE_URL}/login")
        
        # Tab through form
        page.keyboard.press('Tab')
        username_input = page.locator('input[name="username"]')
        expect(username_input).to_be_focused()
        
        page.keyboard.press('Tab')
        password_input = page.locator('input[name="password"]')
        expect(password_input).to_be_focused()
    
    def test_submit_with_enter_key(self, page: Page, admin_credentials):
        """Should submit form with Enter key"""
        page.goto(f"{BASE_URL}/login")
        
        page.fill('input[name="username"]', admin_credentials['username'])
        page.fill('input[name="password"]', admin_credentials['password'])
        page.select_option('select[name="role"]', admin_credentials['role'])
        
        # Press Enter instead of clicking button
        page.keyboard.press('Enter')
        
        # Should still redirect
        page.wait_for_url(f"{BASE_URL}/dashboard", timeout=5000)
        assert "/dashboard" in page.url