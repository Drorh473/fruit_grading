"""
Shared Utilities
Helper functions used across route modules
"""
from flask import current_app, g
import pymongo

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