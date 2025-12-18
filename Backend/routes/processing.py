"""
Processing Pipeline Routes
Endpoints for ML pipeline control and monitoring
"""
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
import threading
import traceback
import sys
from pathlib import Path
from shared_state import pipeline_state

processing_bp = Blueprint('processing', __name__)

# Import build_model functions
try:
    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from processes.build_model import (
        run_tests, setup_database, preprocess_data,
        extract_features, train_classifier,
        generate_confusion_matrix
    )
except ImportError as e:
    print(f"Warning: Could not import build_model functions: {e}")
    run_tests = None
    setup_database = None
    preprocess_data = None
    extract_features = None
    train_classifier = None
    generate_confusion_matrix = None


def run_pipeline_background(config):
    """Background task for running pipeline"""
    try:
        pipeline_state.update_state(status='running', progress=0, running=True)
        pipeline_state.add_log("Pipeline started", 'info')
        
        # Extract config
        skip_tests = config.get('skipTests', False)
        setup_database_con = config.get('setupDatabase', True)
        preprocess_data_flag = config.get('preprocessData', True) 
        extract_features_flag = config.get('extractFeatures', True)
        train_classifier_flag = config.get('trainClassifier', True)
        generate_confusion_matrix_flag = config.get('generateConfusionMatrix', True)
        
        # Training hyperparameters
        hidden_dim = config.get('hiddenDim', 16)
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learningRate', 0.0005)
        lambda_reg = config.get('lambdaReg', 0.001)
        
        # Initialize variables that might be used later
        train_gen = None
        test_gen = None
        train_features = None
        test_features = None
        params = None
        results = None  # Initialize results to None
        
        # Step 0: Tests (optional)
        if not skip_tests:
            pipeline_state.add_log("Running test suite...", 'info')
            test_success = run_tests()
            if not test_success:
                pipeline_state.add_log("Tests failed! Aborting pipeline.", 'error')
                pipeline_state.update_state(status='failed', running=False)
                return
        
        # Step 1: Database Setup
        if setup_database_con:
            pipeline_state.update_step(1, 'processing')
            pipeline_state.add_log("Step 1/5: Setting up database...", 'info')
            
            success = setup_database()  # Call the actual function
            if not success:
                pipeline_state.add_log("Database setup failed!", 'error')
                pipeline_state.update_step(1, 'failed')
                pipeline_state.update_state(status='failed', running=False)
                return
            
            pipeline_state.update_step(1, 'completed')
            pipeline_state.add_log("Database setup complete", 'success')
        
        # Step 2: Preprocessing
        if preprocess_data_flag:
            pipeline_state.update_step(2, 'processing')
            pipeline_state.add_log("Step 2/5: Preprocessing data...", 'info')
            
            train_gen, test_gen = preprocess_data()
            if not train_gen or not test_gen:
                pipeline_state.add_log("Preprocessing failed!", 'error')
                pipeline_state.update_step(2, 'failed')
                pipeline_state.update_state(status='failed', running=False)
                return
            
            pipeline_state.update_step(2, 'completed')
            pipeline_state.add_log(f"Preprocessing complete", 'success')
        
        # Step 3: Feature Extraction
        if extract_features_flag:
            pipeline_state.update_step(3, 'processing')
            pipeline_state.add_log("Step 3/5: Extracting features...", 'info')
            
            train_features, test_features = extract_features(train_gen, test_gen)
            if not train_features or not test_features:
                pipeline_state.add_log("Feature extraction failed!", 'error')
                pipeline_state.update_step(3, 'failed')
                pipeline_state.update_state(status='failed', running=False)
                return
            
            pipeline_state.update_step(3, 'completed')
            pipeline_state.add_log(f"Feature extraction complete", 'success')
        
        # Step 4: Model Training
        if train_classifier_flag:
            pipeline_state.update_step(4, 'processing')
            pipeline_state.add_log(f"Step 4/5: Training classifier...", 'info')
            
            params, results = train_classifier(
                train_features, test_features,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg
            )
            
            if params is None or results is None:
                pipeline_state.add_log("Classifier training failed!", 'error')
                pipeline_state.update_step(4, 'failed')
                pipeline_state.update_state(status='failed', running=False)
                return
            
            pipeline_state.update_step(4, 'completed')
            pipeline_state.add_log(f"Training complete - Accuracy: {results['test_accuracy']*100:.2f}%", 'success')
        
        # Step 5: Evaluation
        if generate_confusion_matrix_flag and results is not None:
            pipeline_state.update_step(5, 'processing')
            pipeline_state.add_log("Step 5/5: Generating evaluation metrics...", 'info')
            
            cm = generate_confusion_matrix(results)
            if cm is not None:
                results['confusion_matrix'] = cm
            
            pipeline_state.update_step(5, 'completed')
            pipeline_state.add_log("Evaluation complete", 'success')
        
        # Step 6: Save dashboard metadata (NEW)
        if results is not None and train_features is not None and test_features is not None:
            pipeline_state.add_log("Step 6/6: Saving dashboard metadata...", 'info')
            try:
                from model_metadata import save_dashboard_metadata
                
                # Get label mapping
                label_mapping = results.get('label_mapping', {
                    'market': 0,
                    'standard': 1,
                    'premium': 2
                })
                
                # Get confusion matrix
                cm = results.get('confusion_matrix', None)
                
                save_dashboard_metadata(
                    results=results,
                    train_features=train_features,
                    test_features=test_features,
                    params=params,
                    label_mapping=label_mapping,
                    confusion_matrix=cm
                )
                
                pipeline_state.add_log("Dashboard metadata saved successfully", 'success')
            except Exception as e:
                pipeline_state.add_log(f"Warning: Could not save dashboard metadata: {str(e)}", 'warning')
        
        # Store results in pipeline_state (only if results exist)
        if results is not None:
            pipeline_state.set_results({
                'train_accuracy': float(results['train_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'train_loss': float(results['train_loss']),
                'test_loss': float(results['test_loss']),
                'totalProcessed': len(train_features) + len(test_features) if train_features and test_features else 0,
                'timestamp': datetime.now().isoformat()
            })
        
        # Complete
        pipeline_state.update_state(status='completed', progress=100, running=False)
        pipeline_state.add_log(f"Pipeline completed successfully!", 'success')
        
    except Exception as e:
        pipeline_state.add_log(f"Pipeline error: {str(e)}", 'error')
        pipeline_state.add_log(traceback.format_exc(), 'error')
        pipeline_state.update_state(status='failed', running=False)
        
        # Mark current step as failed
        state = pipeline_state.get_state()
        if state['currentStep'] > 0:
            pipeline_state.update_step(state['currentStep'], 'failed')

@processing_bp.route('/start', methods=['POST'])
def start_pipeline():
    """Start processing pipeline"""
    try:
        if pipeline_state.is_running():
            return jsonify({
                'success': False,
                'error': 'Pipeline already running'
            }), 400
        
        config = request.get_json() or {}
        
        # Reset pipeline state
        pipeline_state.reset_pipeline()
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_pipeline_background,
            args=(config,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'pipelineId': 'pipeline_' + datetime.now().strftime('%Y%m%d%H%M%S'),
            'status': 'started'
        }), 200
        
    except Exception as e:
        print(f"Error starting pipeline: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@processing_bp.route('/stop', methods=['POST'])
def stop_pipeline():
    """Stop pipeline"""
    try:
        pipeline_state.update_state(running=False, status='stopped')
        pipeline_state.add_log('Pipeline stopped by user', 'warning')
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"Error stopping pipeline: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@processing_bp.route('/status', methods=['GET'])
def get_pipeline_status():
    """Get pipeline status"""
    try:
        state = pipeline_state.get_state()
        
        response = {
            'running': state['running'],
            'status': state['status'],
            'currentStep': state['currentStep'],
            'progress': state['progress'],
            'steps': state['steps']
        }
        
        # Add results if completed
        results = pipeline_state.get_results()
        if state['status'] == 'completed' and results:
            response['totalProcessed'] = results['totalProcessed']
            response['accuracy'] = results['test_accuracy']
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error getting pipeline status: {e}")
        return jsonify({
            'running': False,
            'status': 'idle',
            'currentStep': 0,
            'progress': 0,
            'steps': []
        }), 200


@processing_bp.route('/logs', methods=['GET'])
def get_pipeline_logs():
    """Get pipeline logs"""
    try:
        limit = int(request.args.get('limit', 100))
        logs = pipeline_state.get_logs(limit)
        
        return jsonify(logs), 200
        
    except Exception as e:
        print(f"Error getting pipeline logs: {e}")
        return jsonify([]), 200


@processing_bp.route('/config', methods=['GET'])
def get_pipeline_config():
    """Get pipeline configuration"""
    try:
        config = pipeline_state.get_config()
        return jsonify(config), 200
        
    except Exception as e:
        print(f"Error getting pipeline config: {e}")
        return jsonify({
            'hiddenDim': 16,
            'epochs': 100,
            'learningRate': 0.0005,
            'lambdaReg': 0.001,
            'batchSize': 32
        }), 200


@processing_bp.route('/config', methods=['PUT'])
def update_pipeline_config():
    """Update pipeline configuration"""
    try:
        config = request.get_json()
        
        # Update config
        updates = {}
        if 'hiddenDim' in config:
            updates['hiddenDim'] = config['hiddenDim']
        if 'epochs' in config:
            updates['epochs'] = config['epochs']
        if 'learningRate' in config:
            updates['learningRate'] = config['learningRate']
        if 'lambdaReg' in config:
            updates['lambdaReg'] = config['lambdaReg']
        if 'batchSize' in config:
            updates['batchSize'] = config['batchSize']
        
        pipeline_state.update_config(**updates)
        
        return jsonify(pipeline_state.get_config()), 200
        
    except Exception as e:
        print(f"Error updating pipeline config: {e}")
        return jsonify({'error': str(e)}), 500