import os
import pytest


def pytest_configure(config):
    """
    Called after command line options have been parsed and all plugins loaded.
    This is the earliest hook where we can set up test isolation.
    """
    # Force test environment before any test modules are imported
    os.environ['TESTING'] = 'true'
    os.environ['DB_NAME'] = 'test_fruit_grading'
    os.environ['PYTEST_CURRENT_TEST'] = 'true'
    
    print("\n" + "="*60)
    print("TEST ISOLATION ENABLED")
    print(f"  Database: test_fruit_grading")
    print(f"  Mode: Testing")
    print("="*60 + "\n")


def pytest_unconfigure(config):
    """
    Called before test process exits.
    Clean up test database.
    """
    try:
        from pymongo import MongoClient
        
        connection_string = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
        client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
        
        # Only drop test database
        test_db_name = 'test_fruit_grading'
        client.drop_database(test_db_name)
        client.close()
        
        print(f"\n[Cleanup] Dropped test database: {test_db_name}")
    except Exception as e:
        print(f"\n[Cleanup] Warning: Could not clean test database: {e}")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """
    Called before each test.
    Verify test isolation is active.
    """
    # Double-check test environment
    db_name = os.getenv('DB_NAME', '')
    if db_name == 'fruit_grading':
        pytest.fail(
            "CRITICAL: Test attempting to use production database 'fruit_grading'! "
            "Test isolation has failed. Check your configuration."
        )


@pytest.fixture(autouse=True, scope='session')
def global_test_isolation():
    """
    Session-scoped fixture that runs once for the entire test session.
    Ensures test isolation at the session level.
    """
    # Store original environment
    original_env = {
        'DB_NAME': os.environ.get('DB_NAME'),
        'TESTING': os.environ.get('TESTING'),
    }
    
    # Force test environment
    os.environ['DB_NAME'] = 'test_fruit_grading'
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]