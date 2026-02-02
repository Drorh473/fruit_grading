"""Image preprocessing and data augmentation for fruit classification."""
import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from dotenv import load_dotenv
import pymongo
import time
from bson.objectid import ObjectId
import random

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

MONGODB_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not MONGODB_CONNECTION_STRING:
    raise ValueError("MongoDB connection string not found in .env file")

PROCESSED_DATASET_PATH = os.getenv('PROCESSED_DATASET_PATH')
STORED_DATASET_PATH = os.getenv('STORED_DATASET_PATH')
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')
DB_NAME = os.getenv('DB_NAME', 'fruit_rotation_db')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
IMAGE_SIZE = (224, 224)
NUM_OF_CAMERAS = int(os.getenv('NUM_OF_CAMERAS'))

def custom_preprocessing(image, save_path=None):
    """Apply Gaussian filtering and CLAHE histogram equalization."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    h, w = image.shape[:2]
    max_dim = 224
    if h != max_dim or w != max_dim:
        ratio = max_dim / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))

    original = image.copy()
    blurred = cv2.GaussianBlur(original, (3, 3), 0.001)

    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if save_path:
        save_img = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, save_img)

    enhanced = enhanced.astype(np.float32) / 255.0
    return enhanced


def apply_augmentation(image):
    """Apply random augmentations: rotation, flip, brightness, contrast, noise, color jitter, zoom."""
    is_float = image.dtype == np.float32 and image.max() <= 1.0
    if is_float:
        image = (image * 255).astype(np.uint8)

    augmented = image.copy()
    h, w = augmented.shape[:2]

    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    if random.random() > 0.7:
        augmented = cv2.flip(augmented, 0)

    if random.random() > 0.5:
        brightness_factor = random.uniform(0.5, 1.5)
        augmented = np.clip(augmented.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        contrast_factor = random.uniform(0.6, 1.4)
        mean = np.mean(augmented, axis=(0, 1), keepdims=True)
        augmented = np.clip((augmented - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

    if random.random() > 0.6:
        noise = np.random.normal(0, random.uniform(5, 15), augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-20, 20), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() > 0.6:
        zoom_factor = random.uniform(0.85, 0.95)
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        y1 = random.randint(0, h - new_h)
        x1 = random.randint(0, w - new_w)
        cropped = augmented[y1:y1+new_h, x1:x1+new_w]
        augmented = cv2.resize(cropped, (w, h))

    if is_float:
        augmented = augmented.astype(np.float32) / 255.0

    return augmented


def process_image(image_path, output_dir, metadata):
    """Process a single image and save the preprocessed version."""
    try:
        set_type = metadata.get('set_type')
        camera_id = metadata.get('camera_id')
        file_id = str(metadata.get('_id'))

        _, ext = os.path.splitext(image_path)
        if not ext:
            ext = '.jpg'

        if set_type and camera_id is not None:
            output_subdir = os.path.join(output_dir, set_type, f"camera_{camera_id}")
        else:
            output_subdir = os.path.join(output_dir, "unknown")

        output_path = os.path.join(output_subdir, f"{file_id}{ext}")
        os.makedirs(output_subdir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            return None, None, f"Could not read {image_path}"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        custom_preprocessing(img, save_path=output_path)

        return file_id, output_path, None
    except Exception as e:
        return None, None, f"Error processing {image_path}: {e}"


def preprocess_and_save_dataset(sequanceofcameras, output_dir=None, db_name=os.getenv('DB_NAME', "fruit_grading"), collection_name="images"):
    """Preprocess all images and return organized paths for training and testing."""
    output_dir = output_dir or PROCESSED_DATASET_PATH

    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]

    all_docs = list(collection.find())
    path_info = {}
    for doc in all_docs:
        original_path = doc.get('path')
        path_info[original_path] = {
            'set_type': doc.get('set_type'),
            'fruit_type': doc.get('fruit_type'),
            'object_id': doc.get('object_id'),
            'camera_id': doc.get('camera_id'),
            'timestamp': doc.get('timestamp'),
            '_id': doc.get('_id')
        }
    client.close()

    total_images = sum(len(camera) for camera in sequanceofcameras if camera)
    if total_images == 0:
        return [], [], {}

    training_paths = []
    testing_paths = []
    metadata_dict = {}
    all_results = []

    for cam_idx, camera in enumerate(sequanceofcameras):
        if not camera:
            continue

        process_args = []
        for path in camera:
            metadata = path_info.get(path, {})
            process_args.append((path, output_dir, metadata))

        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(process_image, process_args),
                total=len(camera),
                desc=f"Camera {cam_idx}"
            ))

        updates = []
        for i, (file_id, output_path, error) in enumerate(results):
            if file_id and output_path:
                try:
                    object_id = ObjectId(file_id)
                    updates.append(pymongo.UpdateOne(
                        {'_id': object_id},
                        {'$set': {'processed_path': output_path}}
                    ))

                    original_path = camera[i]
                    if original_path in path_info:
                        info = path_info[original_path]
                        set_type = info['set_type']

                        if set_type == 'training':
                            training_paths.append(output_path)
                        elif set_type == 'testing':
                            testing_paths.append(output_path)

                        metadata_dict[output_path] = {
                            'fruit_type': info['fruit_type'],
                            'object_id': info['object_id'],
                            'camera_id': info['camera_id'],
                            'timestamp': info['timestamp']
                        }
                except Exception:
                    pass

        if updates:
            client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
            db = client[db_name]
            collection = db[collection_name]

            batch_size = 500
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                try:
                    collection.bulk_write(batch, ordered=False)
                except Exception:
                    pass
            client.close()

        all_results.extend(results)

    return training_paths, testing_paths, metadata_dict


def load_dataset_split_by_camera(db_name=os.getenv('DB_NAME'), collection_name="images"):
    """Load image paths grouped by camera ID."""
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client[db_name]
    collection = db[collection_name]
    sequenceofcamera = [[] for _ in range(NUM_OF_CAMERAS)]
    for image in collection.find():
        camera_id = image.get('camera_id')
        if 0 <= camera_id <= NUM_OF_CAMERAS - 1:
            sequenceofcamera[camera_id].append(image.get('path'))
    return sequenceofcamera

def set_generator(image_paths, metadata_dict, augment=False, augment_multiplier=3):
    """Create a batch generator for the given image paths with optional augmentation."""
    if not image_paths:
        return None, {}, 0

    effective_count = len(image_paths) * augment_multiplier if augment else len(image_paths)

    def batch_generator():
        if augment:
            indices = [(i, aug_idx) for i in range(len(image_paths))
                      for aug_idx in range(augment_multiplier)]
        else:
            indices = [(i, 0) for i in range(len(image_paths))]

        np.random.shuffle(indices)
        effective_count = len(indices)
        num_batches = (effective_count + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, effective_count)
            batch_indices = indices[start_idx:end_idx]

            batch_images = []
            batch_metadata = []

            for img_idx, aug_version in batch_indices:
                img_path = image_paths[img_idx]

                if not os.path.exists(img_path):
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, IMAGE_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if augment and aug_version > 0:
                        img = apply_augmentation(img)

                    batch_images.append(img)
                    img_data_dict = metadata_dict.get(img_path, {})
                    batch_metadata.append(img_data_dict)
                except Exception:
                    continue

            if not batch_images:
                continue

            batch_x = np.array(batch_images)
            batch_y = np.array(batch_metadata)
            yield batch_x, batch_y

    batch_generator.samples = effective_count
    batch_generator.num_batches = (effective_count + BATCH_SIZE - 1) // BATCH_SIZE

    return batch_generator, {}, effective_count

def load_dataset_with_preprocessing():
    """Load dataset with custom preprocessing and return train/test generators."""
    processed_dir = PROCESSED_DATASET_PATH or "processed_dataset"
    sequence_camera = load_dataset_split_by_camera()

    if not os.path.isdir(processed_dir):
        training_paths, testing_paths, metadata_dict = preprocess_and_save_dataset(
            sequence_camera, processed_dir
        )
    else:
        client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db["images"]

        training_docs = list(collection.find({"set_type": "training"}))
        training_paths = [doc.get('processed_path') or doc.get('path')
                         for doc in training_docs]

        testing_docs = list(collection.find({"set_type": "testing"}))
        testing_paths = [doc.get('processed_path') or doc.get('path')
                        for doc in testing_docs]

        metadata_dict = {}
        for doc in training_docs + testing_docs:
            path = doc.get('processed_path') or doc.get('path')
            metadata_dict[path] = {
                'fruit_type': doc.get('fruit_type'),
                'object_id': doc.get('object_id'),
                'camera_id': doc.get('camera_id'),
                'timestamp': doc.get('timestamp')
            }
        client.close()

    train_gen, _, _ = set_generator(training_paths, metadata_dict, augment=True, augment_multiplier=5)
    test_gen, _, _ = set_generator(testing_paths, metadata_dict, augment=False)

    return train_gen, test_gen