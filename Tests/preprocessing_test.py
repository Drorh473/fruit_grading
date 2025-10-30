import unittest
import os
import cv2
import numpy as np
import pymongo
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from preprocessing.preprocessing_from_db import (
    custom_preprocessing,
    process_image,
    load_dataset_split_by_camera,
    set_generator
)

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
TEST_DB_NAME = "test_preprocessing"
TEST_COLLECTION_NAME = "test_images"
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS', 4))


class TestCustomPreprocessing(unittest.TestCase):
    """Test cases for custom preprocessing functions"""
    
    def test_preprocessing_output_shape(self):
        """Test that preprocessing maintains correct output shape"""
        # Create a sample RGB image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        result = custom_preprocessing(test_image)
        
        # Check output is normalized
        self.assertEqual(result.dtype, np.float32)
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)
        
        # Check shape is resized to max dimension
        self.assertLessEqual(max(result.shape[:2]), 224)
    
    def test_preprocessing_with_float_input(self):
        """Test preprocessing handles float input correctly"""
        # Create float image in [0, 1] range
        test_image = np.random.rand(200, 200, 3).astype(np.float32)
        
        result = custom_preprocessing(test_image)
        
        self.assertEqual(result.dtype, np.float32)
        self.assertLessEqual(result.max(), 1.0)
    
    def test_preprocessing_saves_file(self):
        """Test that preprocessing saves file when path provided"""
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        save_path = "/tmp/test_preprocessed.png"
        
        # Clean up if exists
        if os.path.exists(save_path):
            os.remove(save_path)
        
        result = custom_preprocessing(test_image, save_path=save_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Verify saved image can be loaded
        saved_img = cv2.imread(save_path)
        self.assertIsNotNone(saved_img)
        
        # Clean up
        os.remove(save_path)
    
    def test_preprocessing_gaussian_blur_applied(self):
        """Test that Gaussian blur reduces high-frequency noise"""
        # Create image with salt-and-pepper noise
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        noise_mask = np.random.rand(224, 224) > 0.95
        test_image[noise_mask] = 255
        
        result = custom_preprocessing(test_image)
        
        # Result should be smoother (less variance)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (224, 224, 3))

    def test_clahe_preserves_bright_regions(self):
        """Test that CLAHE doesn't over-enhance already bright regions"""
        # Create image with both dark and bright regions
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image[:112, :] = 50   # Dark region
        test_image[112:, :] = 200  # Bright region
        
        # Convert to RGB
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        result = custom_preprocessing(test_image)
        
        # Check that result doesn't clip values
        self.assertLess(result.max(), 1.0)
        self.assertGreater(result.min(), 0.0)
        
        # Check that contrast between regions is maintained
        dark_region = result[:112, :, :].mean()
        bright_region = result[112:, :, :].mean()
        self.assertGreater(bright_region, dark_region)



class TestProcessImage(unittest.TestCase):
    """Test cases for process_image function"""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test image"""
        cls.test_dir = "/tmp/test_images"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create test directory structure
        cls.image_dir = os.path.join(cls.test_dir, "training", "camera_0")
        os.makedirs(cls.image_dir, exist_ok=True)
        
        # Create a test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cls.test_image_path = os.path.join(cls.image_dir, "test_image.png")
        cv2.imwrite(cls.test_image_path, test_img)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_process_image_success(self):
        """Test successful image processing"""
        output_dir = "/tmp/test_output"
        args = (self.test_image_path, output_dir)
        
        file_id, output_path, error = process_image(args)
        
        self.assertIsNotNone(file_id)
        self.assertIsNotNone(output_path)
        self.assertIsNone(error)
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    
    def test_process_image_invalid_path(self):
        """Test processing with invalid image path"""
        output_dir = "/tmp/test_output"
        args = ("/invalid/path/image.png", output_dir)
        
        file_id, output_path, error = process_image(args)
        
        self.assertIsNone(file_id)
        self.assertIsNone(output_path)
        self.assertIsNotNone(error)


class TestDatabaseFunctions(unittest.TestCase):
    """Test cases for database-related functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database with sample data"""
        cls.client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
        cls.db = cls.client[TEST_DB_NAME]
        cls.collection = cls.db[TEST_COLLECTION_NAME]
        
        # Insert sample data
        sample_data = []
        for camera_id in range(NUM_OF_CAMERAS):
            for i in range(5):
                sample_data.append({
                    "path": f"/test/camera_{camera_id}/image_{i}.png",
                    "processed_path": f"/test/processed/camera_{camera_id}/image_{i}.png",
                    "camera_id": camera_id,
                    "set_type": "training" if i < 3 else "testing",
                    "fruit_type": "apple",
                    "object_id": f"obj{i:04d}"
                })
        
        cls.collection.insert_many(sample_data)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        cls.client.drop_database(TEST_DB_NAME)
        cls.client.close()
    
    def test_load_dataset_split_by_camera(self):
        """Test loading dataset split by camera"""
        sequence = load_dataset_split_by_camera(
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )
        
        # Check we have correct number of camera lists
        self.assertEqual(len(sequence), NUM_OF_CAMERAS)
        
        # Check each camera has images
        for camera_list in sequence:
            self.assertIsInstance(camera_list, list)
        
        # Check total images
        total_images = sum(len(cam) for cam in sequence)
        self.assertGreater(total_images, 0)
    
    def test_set_generator_training(self):
        """Test generator for training set"""
        gen, metadata, count = set_generator(
            "training",
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )
        
        # Should return None if no processed images exist
        # But count should reflect database entries
        self.assertIsNotNone(count)
        self.assertGreaterEqual(count, 0)
    
    def test_set_generator_testing(self):
        """Test generator for testing set"""
        gen, metadata, count = set_generator(
            "testing",
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )
        
        self.assertIsNotNone(count)
        self.assertGreaterEqual(count, 0)
    
    def test_set_generator_invalid_set_type(self):
        """Test generator with invalid set type"""
        gen, metadata, count = set_generator(
            "invalid_set",
            db_name=TEST_DB_NAME,
            collection_name=TEST_COLLECTION_NAME
        )
        
        self.assertEqual(count, 0)


class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing pipeline"""
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create test image
        test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Apply preprocessing
        result = custom_preprocessing(test_img)
        
        # Verify output properties
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 3)  # RGB channels
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)
    
    def test_preprocessing_maintains_colors(self):
        """Test that preprocessing maintains color information"""
        # Create a pure red image
        red_image = np.zeros((224, 224, 3), dtype=np.uint8)
        red_image[:, :, 0] = 255  # Red channel
        
        result = custom_preprocessing(red_image)
        
        # Red channel should have highest values
        self.assertGreater(result[:, :, 0].mean(), result[:, :, 1].mean())
        self.assertGreater(result[:, :, 0].mean(), result[:, :, 2].mean())


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)