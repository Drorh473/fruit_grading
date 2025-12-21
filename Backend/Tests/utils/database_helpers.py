import pymongo
from Tests.test_config import TestConfig


def clear_test_database(db_name=None):
    """Clear test database"""
    if db_name is None:
        db_name = TestConfig.TEST_DB_NAME
    
    client = pymongo.MongoClient(TestConfig.MONGO_CONNECTION_STRING)
    client.drop_database(db_name)
    client.close()


def get_test_collection():
    """Get test collection with indexes"""
    client = pymongo.MongoClient(TestConfig.MONGO_CONNECTION_STRING)
    db = client[TestConfig.TEST_DB_NAME]
    collection = db[TestConfig.TEST_COLLECTION_NAME]
    
    # Create indexes
    collection.create_index('fruit_type')
    collection.create_index('object_id')
    collection.create_index('set_type')
    collection.create_index('camera_id')
    
    return collection


def insert_test_data(collection, num_documents=20):
    """Insert test data into collection"""
    documents = []
    fruit_types = TestConfig.FRUIT_TYPES
    
    for i in range(num_documents):
        fruit_type = fruit_types[i % len(fruit_types)]
        camera_id = i % TestConfig.NUM_OF_CAMERAS
        set_type = "training" if i < num_documents * 0.7 else "testing"
        
        documents.append({
            "path": f"/test/path/{fruit_type}_obj{i:04d}_cam{camera_id}.png",
            "processed_path": f"/test/processed/{fruit_type}_obj{i:04d}_cam{camera_id}.png",
            "fruit_type": fruit_type,
            "object_id": f"obj{i:04d}",
            "camera_id": camera_id,
            "timestamp": f"2025-01-01T00:{i:02d}:00",
            "width": 224,
            "height": 224,
            "color": 3,
            "set_type": set_type,
            "category": "A"
        })
    
    collection.insert_many(documents)
    return documents