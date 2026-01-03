"""
Shared Pipeline State with Smart Caching
Manages ML pipeline state across all routes with efficient caching
"""
from datetime import datetime
from threading import Lock
import json
import os
from pathlib import Path

class PipelineState:
    """Thread-safe pipeline state manager with smart caching"""
    
    def __init__(self):
        self._lock = Lock()
        self._state = {
            'running': False,
            'status': 'idle',
            'currentStep': 0,
            'progress': 0,
            'logs': [],
            'steps': [
                {'id': 1, 'name': 'Database Setup', 'status': 'pending'},
                {'id': 2, 'name': 'Data Preprocessing', 'status': 'pending'},
                {'id': 3, 'name': 'Feature Extraction', 'status': 'pending'},
                {'id': 4, 'name': 'Model Training', 'status': 'pending'},
                {'id': 5, 'name': 'Evaluation', 'status': 'pending'}
            ],
            'config': {
                'hiddenDim': 16,
                'epochs': 100,
                'learningRate': 0.0005,
                'lambdaReg': 0.001,
                'batchSize': 32
            },
            'results': None,
            'pipeline_thread': None
        }
        
        # Caching system
        self._results_cache = None
        self._cache_timestamp = None
        self._metadata_path = Path(os.getenv('MODEL_DIR', 'saved_models')) / 'dashboard_metadata.json'
        
        # Load initial results from disk (once on startup)
        self._load_results_from_disk()
    
    def _load_results_from_disk(self):
        """Load results from disk file - called only on startup or when explicitly triggered"""
        try:
            if self._metadata_path.exists():
                with open(self._metadata_path, 'r') as f:
                    data = json.load(f)
                    self._results_cache = data
                    self._cache_timestamp = datetime.now()
                    print(f"Dashboard metadata loaded from {self._metadata_path} (startup)")
            else:
                print(f"No metadata file found at {self._metadata_path}")
        except Exception as e:
            print(f"Error loading dashboard metadata: {e}")
            self._results_cache = None
    
    def invalidate_cache(self):
        """Force reload of results from disk (call after pipeline completion)"""
        print("Invalidating results cache - reloading from disk")
        self._load_results_from_disk()
    
    def get_state(self):
        """Get current state"""
        with self._lock:
            return self._state.copy()
    
    def update_state(self, **kwargs):
        """Update state fields"""
        with self._lock:
            self._state.update(kwargs)
    
    def add_log(self, message, log_type='info'):
        """Add log entry"""
        with self._lock:
            self._state['logs'].append({
                'message': message,
                'type': log_type,
                'timestamp': datetime.now().isoformat()
            })
        print(f"[{log_type.upper()}] {message}")
    
    def update_step(self, step_id, status):
        """Update step status"""
        with self._lock:
            for step in self._state['steps']:
                if step['id'] == step_id:
                    step['status'] = status
                    break
            self._state['currentStep'] = step_id
            self._state['progress'] = int((step_id / len(self._state['steps'])) * 100 - 20)
    
    def reset_pipeline(self):
        """Reset pipeline for new run"""
        with self._lock:
            self._state['running'] = False
            self._state['status'] = 'idle'
            self._state['currentStep'] = 0
            self._state['progress'] = 0
            self._state['logs'] = []

            for step in self._state['steps']:
                step['status'] = 'pending'
    
    def get_config(self):
        """Get pipeline configuration"""
        with self._lock:
            return self._state['config'].copy()
    
    def update_config(self, **kwargs):
        """Update pipeline configuration"""
        with self._lock:
            self._state['config'].update(kwargs)
    
    def get_results(self):
        """
        Get pipeline results from cache (no disk I/O)
        Returns cached results loaded at startup or after pipeline completion
        """
        return self._results_cache
    
    def set_results(self, results):
        """
        Set pipeline results (called after training completes)
        Updates both memory cache and triggers disk save
        """
        with self._lock:
            self._state['results'] = results
            self._results_cache = results
            self._cache_timestamp = datetime.now()
        
        # Save to disk
        try:
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Dashboard metadata saved to {self._metadata_path}")
        except Exception as e:
            print(f"Error saving dashboard metadata: {e}")
    
    def is_running(self):
        """Check if pipeline is running"""
        with self._lock:
            return self._state['running']
    
    def get_logs(self, limit=100):
        """Get recent logs"""
        with self._lock:
            return self._state['logs'][-limit:]
    
    def get_cache_info(self):
        """Get cache metadata for debugging"""
        return {
            'cached': self._results_cache is not None,
            'timestamp': self._cache_timestamp.isoformat() if self._cache_timestamp else None,
            'file_exists': self._metadata_path.exists()
        }

# Global pipeline state instance
pipeline_state = PipelineState()