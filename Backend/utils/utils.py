"""
Shared Utilities
Helper functions used across route modules
"""
from functools import wraps
from flask import current_app, g, jsonify

def get_db():
    """Get database instance"""
    if 'db' not in g:
        if current_app.mongo_client is None:
            raise Exception("Database not initialized")
        g.db = current_app.mongo_client[current_app.config['DB_NAME']]
    return g.db

def get_collection(collection_name):
    """Get a specific collection"""
    db = get_db()
    return db[collection_name]

def check_db_connection():
    """Check if database is connected"""
    try:
        if current_app.mongo_client is None:
            return False
        current_app.mongo_client.server_info()
        return True
    except:
        return False


def handle_api_errors(default_response=None, status_code=200):
    """Decorator to handle API errors consistently."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                current_app.logger.error(f"API Error in {func.__name__}: {str(e)}")

                if callable(default_response):
                    response = default_response()
                elif default_response is not None:
                    response = default_response
                else:
                    response = {'error': str(e)}

                return jsonify(response), status_code

        return wrapper
    return decorator