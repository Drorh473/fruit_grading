from flask import Blueprint, jsonify, request, current_app
import os
import sys
import shutil
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processes.data_insertion import process_new_fruit_folder

add_fruit_bp = Blueprint('add_fruit', __name__)

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads_temp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

"""
Handles multi-angle fruit image upload and classification.
Receives image files organized by camera angle (an_0 through an_3), saves them to a temporary
batch folder, runs the ML processing pipeline, and returns the predicted fruit grade with confidence.
Cleans up temporary files after processing regardless of success or failure.
"""
@add_fruit_bp.route('/upload-process', methods=['POST'])
def upload_and_process():
    temp_dir = None
    try:
        if 'dataset' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        files = request.files.getlist('dataset')
        
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'error': 'No selected files'}), 400

        batch_id = str(uuid.uuid4())
        temp_dir = os.path.join(UPLOAD_FOLDER, batch_id)
        os.makedirs(temp_dir, exist_ok=True)

        saved_count = 0
        valid_angles = ['an_0', 'an_1', 'an_2', 'an_3', 'angle_0', 'angle_1', 'angle_2', 'angle_3']

        for file in files:
            raw_path = file.filename.replace('\\', '/')
            parts = raw_path.split('/')
            
            target_angle = None
            for part in parts:
                if part in valid_angles:
                    target_angle = part
                    break
            
            if target_angle and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                angle_dir = os.path.join(temp_dir, target_angle)
                os.makedirs(angle_dir, exist_ok=True)
                
                filename = secure_filename(parts[-1])
                save_path = os.path.join(angle_dir, filename)
                
                file.save(save_path)
                saved_count += 1

        if saved_count == 0:
            shutil.rmtree(temp_dir)
            return jsonify({
                'success': False, 
                'error': 'No valid images found in angle folders'
            }), 400

        db_name = current_app.config.get('DB_NAME', 'fruit_grading')
        
        result = process_new_fruit_folder(
            folder_path=temp_dir,
            db_name=db_name,
            collection_name='images',
            run_tests=False
        )
        
        if result is None:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return jsonify({
                'success': False,
                'error': 'Processing pipeline failed'
            }), 500

        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        return jsonify({
            'success': True,
            'objectId': result.get('object_id', 'unknown'),
            'predictedType': result.get('predicted_type', 'unknown'),
            'confidence': result.get('confidence', 0.0),
            'imagesProcessed': result.get('images_count', saved_count),
            'processingTime': result.get('processing_time', 0.0)
        }), 200

    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
                
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
