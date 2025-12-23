"""
Results Routes
Endpoints for viewing and exporting classification results
"""
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from collections import Counter
from utils import get_collection

results_bp = Blueprint('results', __name__)


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
        
        # Get query parameters

        search = request.args.get('search', '')
        fruit_type = request.args.get('type', 'all')
        batch = request.args.get('batch', 'all')
        
        # Build query
        query = {}
        if search:
            query['object_id'] = {'$regex': search, '$options': 'i'}
        if fruit_type != 'all':
            query['fruit_type'] = fruit_type
        if batch != 'all':
            query['batch_id'] = batch
        
        # Aggregate to get unique objects
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
        
        # Get results
        results = list(collection.aggregate(pipeline))
        
        # Sort and paginate
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        total = len(results)
        paginated = results[offset:offset + limit]
        
        # Format results
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
        
        # Get unique objects
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
        
        # Quality rate (non-reject)
        quality_count = sum(1 for obj in objects if obj['fruit_type'] != 'reject')
        quality_rate = round((quality_count / total_processed * 100) if total_processed > 0 else 0, 1)
        
        # Processing speed (objects per hour) - estimate based on recent data
        now = datetime.now()
        recent_objects = [obj for obj in objects if hasattr(obj['timestamp'], 'timestamp') or isinstance(obj['timestamp'], datetime)]
        if recent_objects:
            hours = 24  # Last 24 hours
            recent_count = sum(1 for obj in recent_objects if (now - obj['timestamp']).total_seconds() < hours * 3600)
            processing_speed = round(recent_count / hours, 1)
        else:
            processing_speed = 0
        
        # Calculate trends by comparing today vs yesterday
        yesterday_start = now - timedelta(days=1)
        yesterday_end = now
        day_before_start = now - timedelta(days=2)
        day_before_end = now - timedelta(days=1)
        
        # Get yesterday's data
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
        
        # Get day before yesterday's data
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
        
        # Calculate trends
        trends = {}
        
        # Total Processed trend
        yesterday_total = len(yesterday_objects)
        day_before_total = len(day_before_objects)
        if day_before_total > 0:
            total_change = ((yesterday_total - day_before_total) / day_before_total) * 100
            trends['totalProcessed'] = f"{'+' if total_change >= 0 else ''}{total_change:.1f}%"
        else:
            trends['totalProcessed'] = None
        
        # Quality Rate trend
        yesterday_quality = sum(1 for obj in yesterday_objects if obj['fruit_type'] != 'reject')
        yesterday_quality_rate = (yesterday_quality / yesterday_total * 100) if yesterday_total > 0 else 0
        
        day_before_quality = sum(1 for obj in day_before_objects if obj['fruit_type'] != 'reject')
        day_before_quality_rate = (day_before_quality / day_before_total * 100) if day_before_total > 0 else 0
        
        if day_before_quality_rate > 0:
            quality_change = yesterday_quality_rate - day_before_quality_rate
            trends['qualityRate'] = f"{'+' if quality_change >= 0 else ''}{quality_change:.1f}%"
        else:
            trends['qualityRate'] = None
        
        # Processing Speed trend (objects per hour)
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
    """Get quality distribution statistics"""
    try:
        collection = get_collection('images')
        
        # Get unique objects with their types
        pipeline = [
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'}
            }}
        ]
        objects = list(collection.aggregate(pipeline))
        
        # Count by type
        type_counts = Counter(obj['fruit_type'] for obj in objects)
        total = len(objects)
        
        distribution = {}
        for quality_type in ['market', 'standard', 'premium', 'reject']:
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
            'premium': {'count': 0, 'percentage': 0},
            'reject': {'count': 0, 'percentage': 0}
        }), 200


@results_bp.route('/alerts', methods=['GET'])
def get_quality_alerts():
    """Get quality alerts based on recent processing"""
    try:
        collection = get_collection('images')
        
        # Get recent objects (last hour)
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
        
        # Check for high reject rate
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
        
        # Add success message if no alerts
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
        # Filter out None/null values and sort
        batches = sorted([b for b in batches if b is not None])
        return jsonify(batches), 200
        
    except Exception as e:
        print(f"Error in get_batches: {e}")
        return jsonify([]), 200


@results_bp.route('/hourly-trend', methods=['GET'])
def get_hourly_trend():
    """Get hourly processing trend"""
    try:
        hours = int(request.args.get('hours', 24))
        collection = get_collection('images')
        
        # Get objects grouped by hour
        now = datetime.now()
        trend_data = []
        
        for i in range(hours):
            hour_start = now - timedelta(hours=hours-i)
            hour_end = hour_start + timedelta(hours=1)
            
            pipeline = [
                {'$match': {
                    'timestamp': {
                        '$gte': hour_start,
                        '$lt': hour_end
                    }
                }},
                {'$group': {
                    '_id': '$object_id',
                    'fruit_type': {'$first': '$fruit_type'}
                }}
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
        print(f"Error in get_hourly_trend: {e}")
        return jsonify([]), 200


@results_bp.route('/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data (mock for now)"""
    try:
        # This would require actual prediction vs ground truth data
        # For now, return a placeholder structure
        return jsonify({
            'classes': ['market', 'standard', 'premium', 'reject'],
            'matrix': [
                [45, 3, 1, 1],  # market
                [2, 38, 2, 0],  # standard
                [1, 2, 35, 1],  # premium
                [0, 1, 0, 18]   # reject
            ],
            'metrics': {
                'accuracy': 0.91,
                'precision': {'market': 0.94, 'standard': 0.86, 'premium': 0.92, 'reject': 0.90},
                'recall': {'market': 0.90, 'standard': 0.90, 'premium': 0.90, 'reject': 0.95}
            }
        }), 200
        
    except Exception as e:
        print(f"Error in get_confusion_matrix: {e}")
        return jsonify({'classes': [], 'matrix': [], 'metrics': {}}), 200


# Keep existing endpoints for backward compatibility
@results_bp.route('/all', methods=['GET'])
def get_all_results():
    """Legacy endpoint - redirects to /list"""
    return get_results_list()


@results_bp.route('/details/<object_id>', methods=['GET'])
def get_result_details(object_id):
    """Get detailed information for specific result"""
    try:
        collection = get_collection('images')
        
        # Find all images for this object
        images = list(collection.find({'object_id': object_id}))
        
        # If no images found, return 404
        if not images:
            return jsonify({'error': 'Result not found'}), 404
        
        # Get first image for basic info
        first_image = images[0]
        
        # Organize images by camera
        images_by_camera = {}
        for img in images:
            camera_id = img.get('camera_id', 0)
            images_by_camera[camera_id] = {
                'camera_id': camera_id,
                'angle': img.get('camera_angle', f'Camera {camera_id}'),
                'image_path': img.get('file_path', ''),
                'timestamp': str(img.get('timestamp', ''))
            }
        
        # Build response
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
            # ADD THESE FIELDS:
            'marketCount': next((r['count'] for r in type_counts if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in type_counts if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0),
            'rejectCount': next((r['count'] for r in type_counts if r['_id'] == 'reject'), 0)
        }), 200
    except Exception as e:
        print(f"Error in get_results_stats: {e}")
        return jsonify({
            'totalObjects': 0, 
            'totalImages': 0,
            'marketCount': 0,
            'standardCount': 0,
            'premiumCount': 0,
            'rejectCount': 0
        }), 200

@results_bp.route('/export', methods=['GET'])
def export_results():
    """Export results as CSV"""
    try:
        from flask import Response
        import io
        import csv
        
        # Use get_collection instead of current_app.mongo_client
        collection = get_collection('images')
        
        # Aggregate results
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
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Object ID', 'Fruit Type', 'Timestamp', 
                        'Confidence', 'Batch ID', 'Image Count'])
        
        # Write data
        for result in results:
            writer.writerow([
                result.get('_id', ''),
                result.get('fruit_type', ''),
                str(result.get('timestamp', '')),
                result.get('confidence', ''),
                result.get('batch_id', ''),
                result.get('image_count', 0)
            ])
        
        # Create response
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