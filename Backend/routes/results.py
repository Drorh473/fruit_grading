"""
Results Routes
Endpoints for viewing and exporting classification results
"""
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from collections import Counter
from utils.utils import get_collection
from utils.model_metadata import (
    load_dashboard_metadata,
    generate_predictions_from_confusion_matrix,
    calculate_per_class_metrics
)

results_bp = Blueprint('results', __name__)


@results_bp.route('/list', methods=['GET'])
def get_results_list():
    """Get results list with filtering and pagination"""
    try:
        try:
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid limit or offset parameter'}), 400
        if limit < 0 or offset < 0:
            return jsonify({'error': 'Limit and offset must be non-negative'}), 400
        
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
        return jsonify({'results': [], 'total': 0, 'limit': 100, 'offset': 0}), 500


@results_bp.route('/kpis', methods=['GET'])
def get_kpis():
    """Get KPIs from model metadata."""
    try:
        metadata = load_dashboard_metadata()

        if metadata:
            dataset_info = metadata.get('dataset_info', {})
            performance = metadata.get('performance', {})

            total_processed = dataset_info.get('total_objects', 0)
            test_accuracy = performance.get('test_accuracy', 0)
            model_accuracy = round(test_accuracy * 100, 1)

            avg_conf = performance.get('avg_confidence')
            avg_confidence = round(avg_conf * 100 if avg_conf and avg_conf <= 1 else (avg_conf or 0), 1)

            cm = metadata.get('confusion_matrix', [])
            if cm:
                correct_count = sum(cm[i][i] for i in range(len(cm)))
                total_with_labels = sum(sum(row) for row in cm)
            else:
                testing_count = dataset_info.get('testing_count', 0)
                correct_count = int(test_accuracy * testing_count)
                total_with_labels = testing_count
        else:
            total_processed = 0
            model_accuracy = 0
            avg_confidence = 0
            correct_count = 0
            total_with_labels = 0

        return jsonify({
            'totalProcessed': total_processed,
            'modelAccuracy': model_accuracy,
            'avgConfidence': avg_confidence,
            'correctCount': correct_count,
            'totalWithLabels': total_with_labels,
            'trends': {}
        }), 200

    except Exception as e:
        return jsonify({
            'totalProcessed': 0,
            'modelAccuracy': 0,
            'avgConfidence': 0,
            'correctCount': 0,
            'totalWithLabels': 0,
            'trends': {}
        }), 200


@results_bp.route('/quality-distribution', methods=['GET'])
def get_quality_distribution():
    """Get quality distribution statistics from metadata"""
    try:
        metadata = load_dashboard_metadata()

        if metadata:
            class_dist = metadata.get('dataset_info', {}).get('class_distribution', {})
            total_objects = metadata.get('dataset_info', {}).get('total_objects', 0)

            distribution = {}
            for quality_type in ['market', 'standard', 'premium', 'reject']:
                type_data = class_dist.get(quality_type, {})
                count = type_data.get('total', 0)
                percentage = round((count / total_objects * 100) if total_objects > 0 else 0, 1)
                distribution[quality_type] = {'count': count, 'percentage': percentage}

            return jsonify(distribution), 200

        return jsonify({
            'market': {'count': 0, 'percentage': 0},
            'standard': {'count': 0, 'percentage': 0},
            'premium': {'count': 0, 'percentage': 0},
            'reject': {'count': 0, 'percentage': 0}
        }), 200

    except Exception as e:
        return jsonify({
            'market': {'count': 0, 'percentage': 0},
            'standard': {'count': 0, 'percentage': 0},
            'premium': {'count': 0, 'percentage': 0},
            'reject': {'count': 0, 'percentage': 0}
        }), 200


@results_bp.route('/test-predictions', methods=['GET'])
def get_test_predictions():
    """Get test set predictions from model metadata"""
    try:
        metadata = load_dashboard_metadata()

        if not metadata:
            return jsonify({
                'predictions': [],
                'total': 0,
                'correct_count': 0,
                'accuracy': 0
            }), 200

        filters = {
            'search': request.args.get('search', ''),
            'actual': request.args.get('actual', 'all'),
            'predicted': request.args.get('predicted', 'all'),
            'correct': request.args.get('correct', 'all'),
            'include_actual_label': True
        }

        predictions = generate_predictions_from_confusion_matrix(metadata, filters=filters)

        total = len(predictions)
        correct_count = sum(1 for p in predictions if p.get('correct'))
        accuracy = round((correct_count / total * 100) if total > 0 else 0, 1)

        return jsonify({
            'predictions': predictions,
            'total': total,
            'correct_count': correct_count,
            'accuracy': accuracy
        }), 200

    except Exception as e:
        return jsonify({
            'predictions': [],
            'total': 0,
            'correct_count': 0,
            'accuracy': 0
        }), 200


@results_bp.route('/training-history', methods=['GET'])
def get_training_history():
    """Get training history from model metadata"""
    try:
        metadata = load_dashboard_metadata()
        
        if not metadata:
            return jsonify({
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }), 200
        
        return jsonify(metadata.get('training_history', {})), 200
        
    except Exception as e:
        return jsonify({
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }), 200


@results_bp.route('/alerts', methods=['GET'])
def get_quality_alerts():
    """Get quality alerts"""
    try:
        collection = get_collection('images')
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        pipeline = [
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
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
        return jsonify([]), 200


@results_bp.route('/hourly-trend', methods=['GET'])
def get_hourly_trend():
    """Get hourly processing trend"""
    try:
        hours = int(request.args.get('hours', 24))
        collection = get_collection('images')
        now = datetime.now()
        trend_data = []
        
        for i in range(hours):
            hour_start = now - timedelta(hours=hours-i)
            hour_end = hour_start + timedelta(hours=1)
            
            pipeline = [
                {'$match': {'timestamp': {'$gte': hour_start, '$lt': hour_end}}},
                {'$group': {'_id': '$object_id', 'fruit_type': {'$first': '$fruit_type'}}}
            ]
            
            objects = list(collection.aggregate(pipeline))
            processed = len(objects)
            quality_count = sum(1 for obj in objects if obj['fruit_type'] != 'reject')
            quality_rate = round((quality_count / processed * 100) if processed > 0 else 0, 1)
            
            trend_data.append({
                'hour': hour_start.strftime('%H:00'),
                'processed': processed,
                'qualityRate': quality_rate
            })
        
        return jsonify(trend_data), 200
        
    except Exception as e:
        return jsonify([]), 200


@results_bp.route('/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix from model metadata"""
    try:
        metadata = load_dashboard_metadata()

        if not metadata:
            return jsonify({'classes': [], 'matrix': [], 'normalized': [], 'metrics': {}}), 200

        label_mapping = metadata.get('label_mapping', {})
        classes = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
        matrix = metadata.get('confusion_matrix', [])

        normalized = []
        if matrix:
            for row in matrix:
                row_sum = sum(row)
                if row_sum > 0:
                    normalized.append([round(val / row_sum, 3) for val in row])
                else:
                    normalized.append([0.0] * len(row))

            total = sum(sum(row) for row in matrix)
            correct = sum(matrix[i][i] for i in range(len(matrix)))
            accuracy = round((correct / total) if total > 0 else 0, 3)

            per_class = calculate_per_class_metrics(matrix, classes)
        else:
            accuracy = metadata.get('performance', {}).get('test_accuracy', 0)
            per_class = metadata.get('per_class_performance', {})

        return jsonify({
            'classes': classes,
            'matrix': matrix,
            'normalized': normalized,
            'metrics': {
                'accuracy': accuracy,
                'per_class': per_class
            }
        }), 200

    except Exception as e:
        return jsonify({'classes': [], 'matrix': [], 'normalized': [], 'metrics': {}}), 200


@results_bp.route('/all', methods=['GET'])
def get_all_results():
    """Legacy endpoint"""
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
        
        # Handle null/None confidence - use default if not a valid number
        conf = first_image.get('confidence')
        if conf is None or conf == 0:
            conf = 0.90

        return jsonify({
            'object_id': object_id,
            'fruit_type': first_image.get('fruit_type', 'unknown'),
            'confidence': conf,
            'timestamp': str(first_image.get('timestamp', '')),
            'batch_id': first_image.get('batch_id', ''),
            'image_count': len(images),
            'images': list(images_by_camera.values())
        }), 200
        
    except Exception as e:
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
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0),
            'rejectCount': next((r['count'] for r in type_counts if r['_id'] == 'reject'), 0)
        }), 200
    except Exception as e:
        return jsonify({
            'totalObjects': 0, 'totalImages': 0,
            'marketCount': 0, 'standardCount': 0, 'premiumCount': 0, 'rejectCount': 0
        }), 200


@results_bp.route('/export', methods=['GET'])
def export_results():
    """Export results as CSV"""
    try:
        from flask import Response
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
        writer.writerow(['Object ID', 'Fruit Type', 'Timestamp', 'Confidence', 'Batch ID', 'Image Count'])
        
        for result in results:
            writer.writerow([
                result.get('_id', ''),
                result.get('fruit_type', ''),
                str(result.get('timestamp', '')),
                result.get('confidence', ''),
                result.get('batch_id', ''),
                result.get('image_count', 0)
            ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=results_export.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500