"""
Results Routes
Endpoints for viewing and exporting classification results
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
from utils import get_collection

results_bp = Blueprint('results', __name__)

@results_bp.route('/all', methods=['GET'])
def get_all_results():
    """Get all results with optional filtering and pagination"""
    try:
        collection = get_collection('images')
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        fruit_type = request.args.get('type')
        search = request.args.get('search')
        
        # Build query
        query = {}
        if fruit_type:
            query['fruit_type'] = fruit_type
        if search:
            query['object_id'] = {'$regex': search, '$options': 'i'}
        
        # Calculate skip
        skip = (page - 1) * limit
        
        # Get total count for pagination
        total = collection.count_documents(query)
        
        # Aggregate to get unique objects
        pipeline = [
            {'$match': query},
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}},
            {'$skip': skip},
            {'$limit': limit}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['_id'],
                'type': result['fruit_type'],
                'confidence': result.get('confidence', 0.90),
                'timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result['timestamp'], datetime) else str(result['timestamp']),
                'imageCount': result['image_count']
            })
        
        return jsonify({
            'results': formatted_results,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit
            }
        }), 200
        
    except Exception as e:
        print(f"Error in get_all_results: {e}")
        return jsonify({
            'results': [],
            'pagination': {
                'page': 1,
                'limit': 50,
                'total': 0,
                'pages': 0
            }
        }), 200


@results_bp.route('/<object_id>', methods=['GET'])
def get_result_details(object_id):
    """Get detailed results for a specific object"""
    try:
        collection = get_collection('images')
        
        # Get all images for this object
        images = list(collection.find({'object_id': object_id}))
        
        if not images:
            return jsonify({'error': 'Object not found'}), 404
        
        # Format result
        result = {
            'objectId': object_id,
            'fruitType': images[0].get('fruit_type'),
            'confidence': images[0].get('confidence', 0.90),
            'timestamp': images[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(images[0]['timestamp'], datetime) else str(images[0]['timestamp']),
            'imageCount': len(images),
            'images': []
        }
        
        # Add image details
        for img in images:
            result['images'].append({
                'cameraId': img.get('camera_id'),
                'angle': img.get('angle'),
                'frameNumber': img.get('frame_number'),
                'path': img.get('image_path')
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in get_result_details: {e}")
        return jsonify({'error': str(e)}), 500


@results_bp.route('/stats', methods=['GET'])
def get_results_stats():
    """Get overall statistics for results"""
    try:
        collection = get_collection('images')
        
        # Get counts by type
        type_pipeline = [
            {'$group': {'_id': '$fruit_type', 'count': {'$sum': 1}}}
        ]
        type_counts = list(collection.aggregate(type_pipeline))
        
        # Get unique objects
        unique_objects = len(collection.distinct('object_id'))
        
        # Get total images
        total_images = collection.count_documents({})
        
        stats = {
            'totalObjects': unique_objects,
            'totalImages': total_images,
            'marketCount': next((r['count'] for r in type_counts if r['_id'] == 'market'), 0),
            'standardCount': next((r['count'] for r in type_counts if r['_id'] == 'standard'), 0),
            'premiumCount': next((r['count'] for r in type_counts if r['_id'] == 'premium'), 0),
            'rejectCount': next((r['count'] for r in type_counts if r['_id'] == 'reject'), 0)
        }
        
        return jsonify(stats), 200
        
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
        collection = get_collection('images')
        
        # Get all unique objects
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$object_id',
                'fruit_type': {'$first': '$fruit_type'},
                'timestamp': {'$first': '$timestamp'},
                'confidence': {'$first': '$confidence'},
                'image_count': {'$sum': 1}
            }},
            {'$sort': {'timestamp': -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        # Create CSV
        csv_lines = ['Object ID,Fruit Type,Confidence,Timestamp,Image Count']
        
        for r in results:
            timestamp_str = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(r['timestamp'], datetime) else str(r['timestamp'])
            csv_lines.append(f"{r['_id']},{r['fruit_type']},{r.get('confidence', 0.0):.4f},{timestamp_str},{r['image_count']}")
        
        csv_content = '\n'.join(csv_lines)
        
        return csv_content, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=fruit_results.csv'
        }
        
    except Exception as e:
        print(f"Error in export_results: {e}")
        return jsonify({'error': str(e)}), 500