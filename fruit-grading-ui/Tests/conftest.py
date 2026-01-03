"""
Pytest Configuration and Fixtures
Shared test setup and utilities
"""

import pytest
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "http://localhost:3000"  # Your React dev server
API_URL = "http://localhost:5000/api"  # Your Flask API


# ============================================================================
# BROWSER FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def browser():
    """
    Session-scoped browser instance
    Reuses browser across all tests
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,  # Set to False to see browser
            slow_mo=0,      # Slow down by N ms for debugging
        )
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def context(browser: Browser):
    """
    Function-scoped context
    Fresh context for each test (isolated state)
    """
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        locale='en-US',
        timezone_id='America/New_York',
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext):
    """
    Function-scoped page
    Fresh page for each test
    """
    page = context.new_page()
    yield page
    page.close()


# ============================================================================
# AUTHENTICATION FIXTURES
# ============================================================================

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


@pytest.fixture
def logged_in_admin(page: Page, admin_credentials):
    """
    Pre-authenticated admin user
    Returns page already logged in as admin
    """
    page.goto(f"{BASE_URL}/login")
    
    # Fill login form
    page.fill('input[name="username"]', admin_credentials['username'])
    page.fill('input[name="password"]', admin_credentials['password'])
    page.select_option('select[name="role"]', admin_credentials['role'])
    
    # Submit
    page.click('button[type="submit"]')
    
    # Wait for navigation
    page.wait_for_url(f"{BASE_URL}/dashboard", timeout=5000)
    
    return page


@pytest.fixture
def logged_in_user(page: Page, user_credentials):
    """
    Pre-authenticated regular user
    Returns page already logged in as user
    """
    page.goto(f"{BASE_URL}/login")
    
    # Fill login form
    page.fill('input[name="username"]', user_credentials['username'])
    page.fill('input[name="password"]', user_credentials['password'])
    page.select_option('select[name="role"]', user_credentials['role'])
    
    # Submit
    page.click('button[type="submit"]')
    
    # Wait for navigation
    page.wait_for_url(f"{BASE_URL}/user-dashboard", timeout=5000)
    
    return page


# ============================================================================
# API FIXTURES
# ============================================================================

@pytest.fixture
def api_client():
    """
    HTTP client for API testing
    """
    import requests
    session = requests.Session()
    session.headers.update({
        'Content-Type': 'application/json'
    })
    yield session
    session.close()


@pytest.fixture
def mock_pipeline_status():
    """Mock pipeline status data"""
    return {
        'status': 'running',
        'progress': 50,
        'current_step': 'preprocessing',
        'steps': [
            {'name': 'database', 'status': 'completed'},
            {'name': 'preprocessing', 'status': 'running'},
            {'name': 'feature_extraction', 'status': 'pending'},
            {'name': 'classification', 'status': 'pending'},
        ]
    }


@pytest.fixture
def mock_dashboard_data():
    """Mock dashboard data"""
    return {
        'total_processed': 1250,
        'accuracy': 94.5,
        'cameras_online': 4,
        'system_uptime': '99.8%'
    }


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def wait_for_element(page: Page):
    """
    Helper to wait for elements
    Usage: wait_for_element('button.submit')
    """
    def _wait(selector, timeout=5000):
        return page.wait_for_selector(selector, timeout=timeout)
    return _wait


@pytest.fixture
def screenshot_on_failure(request, page: Page):
    """
    Automatically take screenshot on test failure
    """
    yield
    if request.node.rep_call.failed:
        screenshot_name = f"failure-{request.node.name}.png"
        page.screenshot(path=f"test-screenshots/{screenshot_name}")


# ============================================================================
# HOOKS
# ============================================================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture test results for screenshot_on_failure
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ============================================================================
# MARKERS
# ============================================================================

def pytest_configure(config):
    """
    Register custom markers
    """
    config.addinivalue_line(
        "markers", "smoke: Quick smoke tests"
    )
    config.addinivalue_line(
        "markers", "regression: Full regression suite"
    )