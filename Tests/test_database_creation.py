import unittest
import os
import pymongo
from pathlib import Path
from dotenv import load_dotenv
from Streamers.database_creation import (
    create_database,
    collect_images,
    store_in_database,
    split_data
)

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
TEST_DB_NAME = "test_fruit_grading"
TEST_COLLECTION_NAME = "test_images"


class TestDatabaseCreation(unittest.TestCase):
    """Test cases for database creation functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database connection"""
        cls.client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
        cls.db = cls.client[TEST_DB_NAME]
        cls.collection = cls.db[TEST_COLLECTION_NAME]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database after all tests"""
        cls.client.drop_database(TEST_DB_NAME)
        cls.client.close()
    
    def setUp(self):
        """Clear collection before each test"""
        self.collection.delete_many({})
    
    def test_create_database(self):
        """Test database and collection creation"""
        db_name, collection_name = create_database(TEST_DB_NAME, TEST_COLLECTION_NAME)
        
        self.assertEqual(db_name, TEST_DB_NAME)
        self.assertEqual(collection_name, TEST_COLLECTION_NAME)
        
        # Check if indexes were created
        indexes = self.collection.index_information()
        self.assertIn('fruit_type_1', indexes)
        self.assertIn('object_id_1', indexes)
        self.assertIn('set_type_1', indexes)
        self.assertIn('camera_id_1', indexes)
    
    def test_collect_images(self):
        """Test image collection from dataset"""
        dataset_path = os.getenv('ORIGINAL_DATASET_PATH')
        
        if not dataset_path or not os.path.exists(dataset_path):
            self.skipTest("Dataset path not found")
        
        image_data = collect_images(dataset_path)
        
        self.assertIsInstance(image_data, list)
        self.assertGreater(len(image_data), 0)
        
        # Check first image has required fields
        if image_data:
            first_image = image_data[0]
            required_fields = ['path', 'fruit_type', 'object_id', 'camera_id', 
                             'timestamp', 'width', 'height']
            for field in required_fields:
                self.assertIn(field, first_image)
    
    def test_store_in_database(self):
        """Test storing image metadata in database"""
        # Create sample image data
        sample_data = [
            {
                "path": "/test/path/image1.png",
                "fruit_type": "market",
                "object_id": "obj0001",
                "camera_id": 0,
                "timestamp": "2025-01-01T00:00:00",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": ""
            },
            {
                "path": "/test/path/image2.png",
                "fruit_type": "standard",
                "object_id": "obj0002",
                "camera_id": 1,
                "timestamp": "2025-01-01T00:00:01",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": ""
            }
        ]
        
        store_in_database(sample_data, TEST_DB_NAME, TEST_COLLECTION_NAME)
        
        # Check if data was stored
        count = self.collection.count_documents({})
        self.assertEqual(count, len(sample_data))
        
        # Verify data integrity
        stored_doc = self.collection.find_one({"fruit_type": "market"})
        self.assertIsNotNone(stored_doc)
        self.assertEqual(stored_doc["object_id"], "obj0001")
    
    def test_split_data(self):
        """Test data splitting into training and testing sets"""
        # Insert sample data
        sample_data = []
        for i in range(100):
            sample_data.append({
                "path": f"/test/path/image{i}.png",
                "fruit_type": "apple",
                "object_id": f"obj{i:04d}",
                "camera_id": i % 4,
                "timestamp": f"2025-01-01T00:00:{i:02d}",
                "width": 224,
                "height": 224,
                "color": 3,
                "set_type": "",
                "category": ""
            })
        
        self.collection.insert_many(sample_data)
        
        # Split data
        split_data(TEST_DB_NAME, TEST_COLLECTION_NAME, 66, 34)
        
        # Check split results
        training_count = self.collection.count_documents({"set_type": "training"})
        testing_count = self.collection.count_documents({"set_type": "testing"})
        
        self.assertGreater(training_count, 0)
        self.assertGreater(testing_count, 0)
        self.assertEqual(training_count + testing_count, 100)
        
        # Check approximate split ratio
        ratio = training_count / 100
        self.assertGreater(ratio, 0.6)  # At least 60%
        self.assertLess(ratio, 0.68)     # At most 68%


if __name__ == '__main__':
    import sys
    
    # Run tests silently
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=0, buffer=True)
    result = unittest.main(testRunner=runner, exit=False).result
    
    # Only print if there are failures or errors
    if result.failures or result.errors:
        print("\n" + "="*60)
        print("FAILURES AND ERRORS:")
        print("="*60)
        for test, traceback in result.failures:
            print(f"\nFAILURE: {test}")
            print(traceback)
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    else:
        print(f"\nAll {result.testsRun} tests passed successfully!")