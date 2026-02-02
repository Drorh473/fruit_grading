"""
Pytest Configuration and Fixtures
Shared test setup and utilities
Updated to match actual JSX components
"""

import pytest
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page ,expect

BASE_URL = "http://localhost:3000"
API_URL = "http://localhost:5000/api"

@pytest.fixture(scope="session")
def browser():
    """Session-scoped browser instance"""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,  # Set to False to see browser
            slow_mo=0,
        )
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def context(browser: Browser):
    """Function-scoped context - fresh for each test"""
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        locale='en-US',
        timezone_id='America/New_York',
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext):
    """Function-scoped page - fresh for each test"""
    page = context.new_page()
    yield page
    page.close()

@pytest.fixture
def admin_credentials():
    """Admin user credentials"""
    return {
        'username': 'admin',
        'password': 'admin123',
        'role': 'admin'
    }


@pytest.fixture
def user_credentials():
    """Regular user credentials"""
    return {
        'username': 'user',
        'password': 'user123',
        'role': 'user'
    }


@pytest.fixture(scope="session")
def base_url():
    """Base URL for the frontend application"""
    return BASE_URL


@pytest.fixture
def logged_in_admin(page: Page, admin_credentials):
    """
    Pre-authenticated admin user
    Returns page already logged in as admin
    Updated to match Login.jsx (uses radio buttons, not select)
    """
    page.goto(f"{BASE_URL}/login")
    
    # Fill login form - Login.jsx uses placeholders
    page.fill('input[placeholder="Enter username"]', admin_credentials['username'])
    page.fill('input[placeholder="Enter password"]', admin_credentials['password'])
    
    # Select admin role - Login.jsx uses radio buttons
    page.locator('.role-option:has(input[value="admin"])').click()
    
    # Submit
    page.click('button.login-button')
    
    # Wait for navigation to dashboard
    page.wait_for_url(f"{BASE_URL}/dashboard", timeout=10000)
    
    # Wait for spinner to disappear (content loaded)
    page.wait_for_selector('.spinner', state='hidden', timeout=15000)
    
    return page

@pytest.fixture
def logged_in_user(page: Page, user_credentials):
    """Pre-authenticated regular user"""
    page.goto(f"{BASE_URL}/login")
    page.fill('input[placeholder="Enter username"]', user_credentials['username'])
    page.fill('input[placeholder="Enter password"]', user_credentials['password'])
    page.locator('.role-option:has-text("Operator")').click()
    expect(page.locator('input[type="radio"][value="user"]')).to_be_checked()
    page.click('button.login-button')
    try:

        page.wait_for_url(f"{BASE_URL}/user-dashboard", timeout=5000)
    except Exception:

        error_msg = page.locator('.login-error span')
        if error_msg.is_visible():
            text = error_msg.text_content()
            raise Exception(f"Login Failed with UI Error: {text}")
        
        if "/login" in page.url:
             raise Exception("Login Failed: Stuck on login page without error message.")
        
        raise

    return page

@pytest.fixture
def api_client():
    """HTTP client for API testing"""
    import requests
    session = requests.Session()
    session.headers.update({
        'Content-Type': 'application/json'
    })
    yield session
    session.close()

@pytest.fixture
def wait_for_element(page: Page):
    """Helper to wait for elements"""
    def _wait(selector, timeout=5000):
        return page.wait_for_selector(selector, timeout=timeout)
    return _wait

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (require backend)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full flow)")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "smoke: Quick smoke tests")