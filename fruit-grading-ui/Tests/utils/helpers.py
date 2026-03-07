"""
Test Helper Functions
Reusable utilities for frontend testing
"""

from playwright.sync_api import Page, expect
from typing import Dict, Any, Optional
import time



def login_user(page: Page, username: str, password: str, role: str = 'user') -> Page:
    """
    Login a user with given credentials
    
    Args:
        page: Playwright page object
        username: Username
        password: Password
        role: User role (admin or user)
    
    Returns:
        Page object after login
    """
    page.goto("http://localhost:3000/login")
    
    page.fill('input[name="username"]', username)
    page.fill('input[name="password"]', password)
    page.select_option('select[name="role"]', role)
    page.click('button[type="submit"]')
    
    # Wait for navigation
    page.wait_for_load_state("networkidle")
    
    return page


def logout_user(page: Page) -> Page:
    """
    Logout current user
    
    Args:
        page: Playwright page object
    
    Returns:
        Page object after logout
    """
    logout_btn = page.locator('button:has-text("Logout"), a:has-text("Logout")')
    if logout_btn.count() > 0:
        logout_btn.click()
        page.wait_for_url("http://localhost:3000/login")
    
    return page



def navigate_to_page(page: Page, route: str) -> Page:
    """
    Navigate to a specific route
    
    Args:
        page: Playwright page object
        route: Route path (e.g., '/dashboard', '/processing')
    
    Returns:
        Page object after navigation
    """
    base_url = "http://localhost:3000"
    page.goto(f"{base_url}{route}")
    page.wait_for_load_state("domcontentloaded")
    return page


def wait_for_navigation(page: Page, timeout: int = 5000):
    """
    Wait for page navigation to complete
    
    Args:
        page: Playwright page object
        timeout: Maximum wait time in milliseconds
    """
    page.wait_for_load_state("networkidle", timeout=timeout)



def wait_for_element(page: Page, selector: str, timeout: int = 5000):
    """
    Wait for element to be visible
    
    Args:
        page: Playwright page object
        selector: CSS selector
        timeout: Maximum wait time in milliseconds
    
    Returns:
        Element locator
    """
    return page.wait_for_selector(selector, state="visible", timeout=timeout)


def fill_form(page: Page, form_data: Dict[str, str]):
    """
    Fill multiple form fields
    
    Args:
        page: Playwright page object
        form_data: Dictionary of field names to values
    """
    for field_name, value in form_data.items():
        page.fill(f'input[name="{field_name}"]', value)


def select_option_by_text(page: Page, selector: str, text: str):
    """
    Select dropdown option by visible text
    
    Args:
        page: Playwright page object
        selector: Select element selector
        text: Option text to select
    """
    page.select_option(selector, label=text)


def assert_element_visible(page: Page, selector: str, message: str = None):
    """
    Assert element is visible
    
    Args:
        page: Playwright page object
        selector: CSS selector
        message: Optional assertion message
    """
    element = page.locator(selector)
    expect(element).to_be_visible()


def assert_element_not_visible(page: Page, selector: str):
    """
    Assert element is not visible
    
    Args:
        page: Playwright page object
        selector: CSS selector
    """
    element = page.locator(selector)
    expect(element).not_to_be_visible()


def assert_element_has_text(page: Page, selector: str, text: str):
    """
    Assert element contains specific text
    
    Args:
        page: Playwright page object
        selector: CSS selector
        text: Expected text
    """
    element = page.locator(selector)
    expect(element).to_contain_text(text)


def assert_url_contains(page: Page, url_part: str):
    """
    Assert current URL contains specific string
    
    Args:
        page: Playwright page object
        url_part: String that should be in URL
    """
    assert url_part in page.url, f"Expected URL to contain '{url_part}', got '{page.url}'"



def make_api_request(method: str, endpoint: str, data: Optional[Dict] = None, 
                     headers: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make API request and return response
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint
        data: Request body data
        headers: Request headers
    
    Returns:
        Response data as dictionary
    """
    import requests
    
    base_url = "http://localhost:5000/api"
    url = f"{base_url}{endpoint}"
    
    default_headers = {'Content-Type': 'application/json'}
    if headers:
        default_headers.update(headers)
    
    if method.upper() == 'GET':
        response = requests.get(url, headers=default_headers)
    elif method.upper() == 'POST':
        response = requests.post(url, json=data, headers=default_headers)
    elif method.upper() == 'PUT':
        response = requests.put(url, json=data, headers=default_headers)
    elif method.upper() == 'DELETE':
        response = requests.delete(url, headers=default_headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    return {
        'status_code': response.status_code,
        'data': response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text,
        'headers': dict(response.headers)
    }


def assert_api_success(response: Dict[str, Any]):
    """
    Assert API response is successful
    
    Args:
        response: Response from make_api_request
    """
    assert response['status_code'] == 200, \
        f"Expected status 200, got {response['status_code']}"


def assert_api_error(response: Dict[str, Any], expected_status: int):
    """
    Assert API response has expected error status
    
    Args:
        response: Response from make_api_request
        expected_status: Expected HTTP status code
    """
    assert response['status_code'] == expected_status, \
        f"Expected status {expected_status}, got {response['status_code']}"



def generate_mock_pipeline_status(status: str = 'idle', progress: int = 0) -> Dict[str, Any]:
    """
    Generate mock pipeline status data
    
    Args:
        status: Pipeline status
        progress: Progress percentage
    
    Returns:
        Mock pipeline status dictionary
    """
    return {
        'status': status,
        'progress': progress,
        'current_step': 'preprocessing' if progress > 0 else None,
        'steps': [
            {'name': 'database', 'status': 'completed' if progress >= 25 else 'pending'},
            {'name': 'preprocessing', 'status': 'running' if 25 <= progress < 50 else ('completed' if progress >= 50 else 'pending')},
            {'name': 'feature_extraction', 'status': 'running' if 50 <= progress < 75 else ('completed' if progress >= 75 else 'pending')},
            {'name': 'classification', 'status': 'running' if 75 <= progress < 100 else ('completed' if progress == 100 else 'pending')},
        ]
    }


def generate_mock_dashboard_data() -> Dict[str, Any]:
    """
    Generate mock dashboard data
    
    Returns:
        Mock dashboard data dictionary
    """
    return {
        'total_processed': 1250,
        'accuracy': 94.5,
        'cameras_online': 4,
        'system_uptime': '99.8%',
        'recent_activity': [
            {'timestamp': '2024-01-03 10:30:00', 'action': 'Processing completed', 'status': 'success'},
            {'timestamp': '2024-01-03 10:15:00', 'action': 'Processing started', 'status': 'info'},
        ]
    }


def generate_mock_results(count: int = 10) -> list:
    """
    Generate mock classification results
    
    Args:
        count: Number of results to generate
    
    Returns:
        List of mock result dictionaries
    """
    fruits = ['apple', 'orange', 'banana']
    grades = ['premium', 'standard', 'market', 'reject']
    
    results = []
    for i in range(count):
        results.append({
            'id': i + 1,
            'fruit_type': fruits[i % len(fruits)],
            'predicted_grade': grades[i % len(grades)],
            'confidence': round(0.75 + (i % 20) / 100, 2),
            'timestamp': f'2024-01-03 10:{i:02d}:00'
        })
    
    return results


def generate_mock_camera_status() -> list:
    """
    Generate mock camera status data
    
    Returns:
        List of camera status dictionaries
    """
    return [
        {'id': 1, 'name': 'Front', 'status': 'online', 'fps': 30, 'resolution': '1920x1080'},
        {'id': 2, 'name': 'Right', 'status': 'online', 'fps': 30, 'resolution': '1920x1080'},
        {'id': 3, 'name': 'Back', 'status': 'online', 'fps': 30, 'resolution': '1920x1080'},
        {'id': 4, 'name': 'Left', 'status': 'online', 'fps': 30, 'resolution': '1920x1080'},
    ]



def take_screenshot(page: Page, name: str, path: str = "test-screenshots"):
    """
    Take screenshot of current page
    
    Args:
        page: Playwright page object
        name: Screenshot filename
        path: Directory to save screenshot
    """
    import os
    
    os.makedirs(path, exist_ok=True)
    screenshot_path = os.path.join(path, f"{name}.png")
    page.screenshot(path=screenshot_path)
    
    return screenshot_path


def screenshot_element(page: Page, selector: str, name: str, path: str = "test-screenshots"):
    """
    Take screenshot of specific element
    
    Args:
        page: Playwright page object
        selector: CSS selector of element
        name: Screenshot filename
        path: Directory to save screenshot
    """
    import os
    
    os.makedirs(path, exist_ok=True)
    screenshot_path = os.path.join(path, f"{name}.png")
    
    element = page.locator(selector)
    element.screenshot(path=screenshot_path)
    
    return screenshot_path



def measure_page_load_time(page: Page, url: str) -> float:
    """
    Measure page load time
    
    Args:
        page: Playwright page object
        url: URL to load
    
    Returns:
        Load time in seconds
    """
    start_time = time.time()
    page.goto(url)
    page.wait_for_load_state("domcontentloaded")
    end_time = time.time()
    
    return end_time - start_time


def wait_for_condition(condition_func, timeout: int = 5, interval: float = 0.1):
    """
    Wait for a condition to become true
    
    Args:
        condition_func: Function that returns boolean
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds
    
    Returns:
        True if condition met, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    
    return False



def validate_form_error(page: Page, expected_error: str):
    """
    Validate form error message is displayed
    
    Args:
        page: Playwright page object
        expected_error: Expected error text
    """
    error_element = page.locator('[class*="error"]')
    expect(error_element).to_be_visible()
    expect(error_element).to_contain_text(expected_error)


def validate_success_message(page: Page, expected_message: str):
    """
    Validate success message is displayed
    
    Args:
        page: Playwright page object
        expected_message: Expected success text
    """
    success_element = page.locator('[class*="success"]')
    expect(success_element).to_be_visible()
    expect(success_element).to_contain_text(expected_message)


def clear_browser_storage(page: Page):
    """
    Clear localStorage and sessionStorage
    
    Args:
        page: Playwright page object
    """
    page.evaluate("localStorage.clear()")
    page.evaluate("sessionStorage.clear()")


def set_local_storage_item(page: Page, key: str, value: str):
    """
    Set localStorage item
    
    Args:
        page: Playwright page object
        key: Storage key
        value: Storage value
    """
    page.evaluate(f"localStorage.setItem('{key}', '{value}')")


def get_local_storage_item(page: Page, key: str) -> str:
    """
    Get localStorage item
    
    Args:
        page: Playwright page object
        key: Storage key
    
    Returns:
        Storage value
    """
    return page.evaluate(f"localStorage.getItem('{key}')")