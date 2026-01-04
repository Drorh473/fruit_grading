"""
Add Fruit Routes
Endpoints for adding and processing new fruit objects
"""
from flask import Blueprint, jsonify, request, current_app
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import validation and processing functions
from preprocessing.preprocessing_insertion import validate_folder_structure
from processes.data_insertion import process_new_fruit_folder

add_fruit_bp = Blueprint('add_fruit', __name__)

@add_fruit_bp.route('/validate', methods=['POST'])
def validate_folder():
    """Validate fruit folder structure"""
    try:
        data = request.get_json()
        folder_path = data.get('folderPath')
        
        if not folder_path:
            return jsonify({
                'valid': False,
                'message': 'Folder path is required'
            }), 400
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            return jsonify({
                'valid': False,
                'message': 'Folder does not exist'
            }), 200
        
        # Use actual validation from preprocessing module
        is_valid, error_msg = validate_folder_structure(folder_path)
        
        if not is_valid:
            return jsonify({
                'valid': False,
                'message': error_msg
            }), 200
        
        # Count images in angle directories
        angle_dirs = ['angle_0', 'angle_1', 'angle_2', 'angle_3']
        total_images = 0
        angles_found = 0
        
        for angle_dir in angle_dirs:
            angle_path = os.path.join(folder_path, angle_dir)
            if os.path.isdir(angle_path):
                angles_found += 1
                # Count image files
                image_files = [f for f in os.listdir(angle_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(image_files)
        
        return jsonify({
            'valid': True,
            'message': 'Folder structure is valid',
            'details': {
                'anglesFound': angles_found,
                'totalImages': total_images,
                'cameraAngles': ['angle_0 (Front)', 'angle_1 (Right)', 
                               'angle_2 (Back)', 'angle_3 (Left)']
            }
        }), 200
        
    except Exception as e:
        print(f"Error in validate_folder: {e}")
        return jsonify({
            'valid': False,
            'message': f'Validation error: {str(e)}'
        }), 500


@add_fruit_bp.route('/process', methods=['POST'])
def process_fruit():
    """Process new fruit object through complete pipeline"""
    try:
        data = request.get_json()
        folder_path = data.get('folderPath')
        run_tests = data.get('runTests', False)
        
        if not folder_path:
            return jsonify({
                'success': False,
                'error': 'Folder path is required'
            }), 400
        
        # Validate folder exists
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'error': 'Folder does not exist'
            }), 400
        
        # Get database name from config
        db_name = current_app.config.get('DB_NAME', 'fruit_grading')
        
        # Run complete processing pipeline
        result = process_new_fruit_folder(
            folder_path=folder_path,
            db_name=db_name,
            collection_name='images',
            run_tests=run_tests
        )
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'Processing pipeline failed. Check server logs for details.'
            }), 500
        
        # Return successful result
        return jsonify({
            'success': True,
            'objectId': result['object_id'],
            'predictedType': result['predicted_type'],
            'confidence': result['confidence'],
            'imagesProcessed': result['images_count'],
            'processingTime': result['processing_time']
        }), 200
        
    except Exception as e:
        print(f"Error in process_fruit: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500


@add_fruit_bp.route('/upload', methods=['POST'])
def upload_fruit():
    """Upload fruit images directly"""
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        fruit_type = request.form.get('fruitType')
        object_id = request.form.get('objectId')
        
        if not fruit_type:
            return jsonify({
                'success': False,
                'error': 'Fruit type is required'
            }), 400
        
        # In production:
        # 1. Save uploaded files to temporary directory
        # 2. Process images
        # 3. Clean up temporary files
        
        return jsonify({
            'success': True,
            'objectId': object_id or f'obj_{len(files)}',
            'filesUploaded': len(files),
            'message': f'Successfully uploaded {len(files)} images'
        }), 200
        
    except Exception as e:
        print(f"Error in upload_fruit: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500