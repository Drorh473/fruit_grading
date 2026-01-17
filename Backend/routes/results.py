"""
Results Routes
Endpoints for viewing and exporting classification results
"""
import os
import json
from flask import Blueprint, jsonify, request, Response
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from utils.utils import get_collection

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models')

results_bp = Blueprint('results', __name__)


def load_dashboard_metadata():
    """Load dashboard metadata from saved model directory"""
    metadata_path = os.path.join(MODEL_DIR, 'dashboard_metadata.json')
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dashboard metadata: {e}")
        return None


@results_bp.route('/list', methods=['GET'])
def get_results_list():
    """Get results list with filtering and pagination"""
    try:
        try:
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid limit or offset parameter'
            }), 400
        if limit < 0 or offset < 0:
            return jsonify({
                'error': 'Limit and offset must be non-negative'
            }), 400
        collection = get_collection('images')
        
        search = request.args.get('search', '')
        fruit_type = request.args.get('type', 'all')
        batch = request.args.get('batch', 'all')
        
        query = {}
        if search:
            query['object_id'] = {'$regex': search, '$options': 'i'}
        if fruit_type != 'all':
            query['fruit_type'] = fruit_type
        if batch != 'all':
            query['batch_id'] = batch
        
        pipeline = [
            {'$match': query},
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'batch_id': {'$first': '$batch_id'},
                'image_count': {'$sum': 1}
            }}
        ]
        
        results = list(collection.aggregate(pipeline))
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        total = len(results)
        paginated = results[offset:offset + limit]
        
        formatted_results = []
        for r in paginated:
            formatted_results.append({
                'id': r['_id'],
                'type': r['fruit_type'],
                'timestamp': r['timestamp'].isoformat() if hasattr(r['timestamp'], 'isoformat') else str(r['timestamp']),
                'batch': r.get('batch_id', 'N/A'),
                'imageCount': r['image_count']
            })
        
        return jsonify({
            'results': formatted_results,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        print(f"Error in get_results_list: {e}")
        return jsonify({
            'results': [], 
            'total': 0, 
            'limit': 100, 
            'offset': 0
        }), 500


@results_bp.route('/kpis', methods=['GET'])
def get_kpis():
    """Get key performance indicators"""
    try:
        collection = get_collection('images')
        
        pipeline = [
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'confidence': {'$first': '$confidence'},
                'timestamp': {'$first': '$timestamp'}
            }}
        ]
        objects = list(collection.aggregate(pipeline))
        
        total_processed = len(objects)
        
        quality_count = sum(1 for obj in objects if obj['fruit_type'] != 'reject')
        quality_rate = round((quality_count / total_processed * 100) if total_processed > 0 else 0, 1)
        
        now = datetime.now()
        recent_objects = [obj for obj in objects if hasattr(obj['timestamp'], 'timestamp') or isinstance(obj['timestamp'], datetime)]
        if recent_objects:
            hours = 24
            recent_count = sum(1 for obj in recent_objects if (now - obj['timestamp']).total_seconds() < hours * 3600)
            processing_speed = round(recent_count / hours, 1)
        else:
            processing_speed = 0
        
        yesterday_start = now - timedelta(days=1)
        yesterday_end = now
        day_before_start = now - timedelta(days=2)
        day_before_end = now - timedelta(days=1)
        
        yesterday_pipeline = [
            {'$match': {
                'timestamp': {
                    '$gte': yesterday_start,
                    '$lt': yesterday_end
                }
            }},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'}
            }}
        ]
        yesterday_objects = list(collection.aggregate(yesterday_pipeline))
        
        day_before_pipeline = [
            {'$match': {
                'timestamp': {
                    '$gte': day_before_start,
                    '$lt': day_before_end
                }
            }},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'}
            }}
        ]
        day_before_objects = list(collection.aggregate(day_before_pipeline))
        
        trends = {}
        
        yesterday_total = len(yesterday_objects)
        day_before_total = len(day_before_objects)
        if day_before_total > 0:
            total_change = ((yesterday_total - day_before_total) / day_before_total) * 100
            trends['totalProcessed'] = f"{'+' if total_change >= 0 else ''}{total_change:.1f}%"
        else:
            trends['totalProcessed'] = None
        
        yesterday_quality = sum(1 for obj in yesterday_objects if obj['fruit_type'] != 'reject')
        yesterday_quality_rate = (yesterday_quality / yesterday_total * 100) if yesterday_total > 0 else 0
        
        day_before_quality = sum(1 for obj in day_before_objects if obj['fruit_type'] != 'reject')
        day_before_quality_rate = (day_before_quality / day_before_total * 100) if day_before_total > 0 else 0
        
        if day_before_quality_rate > 0:
            quality_change = yesterday_quality_rate - day_before_quality_rate
            trends['qualityRate'] = f"{'+' if quality_change >= 0 else ''}{quality_change:.1f}%"
        else:
            trends['qualityRate'] = None
        
        yesterday_speed = round(yesterday_total / 24, 1) if yesterday_total > 0 else 0
        day_before_speed = round(day_before_total / 24, 1) if day_before_total > 0 else 0
        
        if day_before_speed > 0:
            speed_change = ((yesterday_speed - day_before_speed) / day_before_speed) * 100
            trends['processingSpeed'] = f"{'+' if speed_change >= 0 else ''}{speed_change:.1f}%"
        else:
            trends['processingSpeed'] = None
        
        return jsonify({
            'totalProcessed': total_processed,
            'qualityRate': quality_rate,
            'processingSpeed': processing_speed,
            'trends': trends
        }), 200
        
    except Exception as e:
        print(f"Error in get_kpis: {e}")
        return jsonify({
            'totalProcessed': 0,
            'qualityRate': 0,
            'processingSpeed': 0,
            'trends': {}
        }), 200


@results_bp.route('/quality-distribution', methods=['GET'])
def get_quality_distribution():
    """Get quality distribution statistics (excluding reject)"""
    try:
        collection = get_collection('images')
        
        pipeline = [
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'}
            }}
        ]
        objects = list(collection.aggregate(pipeline))
        
        type_counts = Counter(obj['fruit_type'] for obj in objects)
        total = sum(type_counts.get(t, 0) for t in ['market', 'standard', 'premium'])
        
        distribution = {}
        for quality_type in ['market', 'standard', 'premium']:
            count = type_counts.get(quality_type, 0)
            percentage = round((count / total * 100) if total > 0 else 0, 1)
            distribution[quality_type] = {
                'count': count,
                'percentage': percentage
            }
        
        return jsonify(distribution), 200
        
    except Exception as e:
        print(f"Error in get_quality_distribution: {e}")
        return jsonify({
            'market': {'count': 0, 'percentage': 0},
            'standard': {'count': 0, 'percentage': 0},
            'premium': {'count': 0, 'percentage': 0}
        }), 200


@results_bp.route('/alerts', methods=['GET'])
def get_quality_alerts():
    """Get quality alerts based on recent processing"""
    try:
        collection = get_collection('images')
        
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        pipeline = [
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'confidence': {'$first': '$confidence'},
                'timestamp': {'$first': '$timestamp'}
            }}
        ]
        objects = list(collection.aggregate(pipeline))
        
        alerts = []
        
        recent_objects = [obj for obj in objects if isinstance(obj['timestamp'], datetime) and obj['timestamp'] > one_hour_ago]
        if recent_objects:
            reject_rate = sum(1 for obj in recent_objects if obj['fruit_type'] == 'reject') / len(recent_objects)
            if reject_rate > 0.3:
                alerts.append({
                    'id': 'high_reject_rate',
                    'type': 'warning',
                    'title': 'High Reject Rate',
                    'message': f'Reject rate at {reject_rate*100:.1f}% in the last hour'
                })
        
        if not alerts:
            alerts.append({
                'id': 'all_good',
                'type': 'success',
                'title': 'System Operating Normally',
                'message': 'All quality metrics within acceptable ranges'
            })
        
        return jsonify(alerts), 200
        
    except Exception as e:
        print(f"Error in get_quality_alerts: {e}")
        return jsonify([]), 200


@results_bp.route('/batches', methods=['GET'])
def get_batches():
    """Get list of unique batch IDs"""
    try:
        collection = get_collection('images')
        batches = collection.distinct('batch_id')
        batches = sorted([b for b in batches if b is not None])
        return jsonify(batches), 200
        
    except Exception as e:
        print(f"Error in get_batches: {e}")
        return jsonify([]), 200


@results_bp.route('/training-history', methods=['GET'])
def get_training_history():
    """Get training history from saved model metadata"""
    try:
        metadata = load_dashboard_metadata()
        
        if not metadata or 'training_history' not in metadata:
            return jsonify({
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }), 200
        
        history = metadata['training_history']
        
        return jsonify({
            'train_loss': history.get('train_loss', []),
            'train_accuracy': history.get('train_accuracy', []),
            'val_loss': history.get('val_loss', []),
            'val_accuracy': history.get('val_accuracy', [])
        }), 200
        
    except Exception as e:
        print(f"Error in get_training_history: {e}")
        return jsonify({
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }), 200


@results_bp.route('/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data from saved model metadata"""
    try:
        metadata = load_dashboard_metadata()
        
        if not metadata:
            return jsonify({
                'classes': [],
                'matrix': [],
                'normalized': [],
                'metrics': {}
            }), 200
        
        # Get confusion matrix from metadata
        cm = metadata.get('confusion_matrix')
        label_mapping = metadata.get('label_mapping', {})
        
        if not cm or not label_mapping:
            return jsonify({
                'classes': [],
                'matrix': [],
                'normalized': [],
                'metrics': {}
            }), 200
        
        # Get class names sorted by index (excluding reject)
        classes = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1]) 
                   if name.lower() != 'reject']
        
        # Filter out reject from matrix
        reject_idx = None
        for name, idx in label_mapping.items():
            if name.lower() == 'reject':
                reject_idx = idx
                break
        
        matrix = cm
        if reject_idx is not None:
            matrix = [row[:reject_idx] + row[reject_idx+1:] for i, row in enumerate(cm) if i != reject_idx]
        
        # Calculate normalized matrix
        normalized = []
        for row in matrix:
            row_sum = sum(row)
            if row_sum > 0:
                normalized.append([round(val / row_sum, 4) for val in row])
            else:
                normalized.append([0.0] * len(row))
        
        # Get metrics from metadata or calculate
        perf = metadata.get('performance', {})
        accuracy = perf.get('test_accuracy', 0)
        
        return jsonify({
            'classes': classes,
            'matrix': matrix,
            'normalized': normalized,
            'metrics': {
                'accuracy': accuracy
            }
        }), 200
        
    except Exception as e:
        print(f"Error in get_confusion_matrix: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'classes': [],
            'matrix': [],
            'normalized': [],
            'metrics': {}
        }), 200


@results_bp.route('/all', methods=['GET'])
def get_all_results():
    """Legacy endpoint - redirects to /list"""
    return get_results_list()


@results_bp.route('/details/<object_id>', methods=['GET'])
def get_result_details(object_id):
    """Get detailed information for specific result"""
    try:
        collection = get_collection('images')
        
        images = list(collection.find({'object_id': object_id}))
        
        if not images:
            return jsonify({'error': 'Result not found'}), 404
        
        first_image = images[0]
        
        images_by_camera = {}
        for img in images:
            camera_id = img.get('camera_id', 0)
            images_by_camera[camera_id] = {
                'camera_id': camera_id,
                'angle': img.get('camera_angle', f'Camera {camera_id}'),
                'image_path': img.get('file_path', ''),
                'timestamp': str(img.get('timestamp', ''))
            }
        
        result = {
            'object_id': object_id,
            'fruit_type': first_image.get('fruit_type', 'unknown'),
            'confidence': first_image.get('confidence', 0.0),
            'timestamp': str(first_image.get('timestamp', '')),
            'batch_id': first_image.get('batch_id', ''),
            'image_count': len(images),
            'images': list(images_by_camera.values())
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in get_result_details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@results_bp.route('/stats', methods=['GET'])
def get_results_stats():
    """Get overall statistics"""
    try:
        collection = get_collection('images')
        
        type_counts = list(collection.aggregate([
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]))
        
        unique_objects = len(collection.distinct('object_id'))
        total_images = collection.count_documents({})
        
        return jsonify({
            'totalObjects': unique_objects,
            'totalImages': total_images,
            'marketCount': next((r['count'] for r in type_counts if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in type_counts if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0)
        }), 200
    except Exception as e:
        print(f"Error in get_results_stats: {e}")
        return jsonify({
            'totalObjects': 0, 
            'totalImages': 0,
            'marketCount': 0,
            'standardCount': 0,
            'premiumCount': 0
        }), 200


@results_bp.route('/export', methods=['GET'])
def export_results():
    """Export results as CSV"""
    try:
        import io
        import csv
        
        collection = get_collection('images')
        
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'batch_id': {'$first': '$batch_id'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['Object ID', 'Grade', 'Timestamp', 'Batch ID', 'Image Count'])
        
        for result in results:
            writer.writerow([
                result.get('_id', ''),
                result.get('fruit_type', ''),
                str(result.get('timestamp', '')),
                result.get('batch_id', ''),
                result.get('image_count', 0)
            ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=results_export.csv'
            }
        )
        
    except Exception as e:
        print(f"Error in export_results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500