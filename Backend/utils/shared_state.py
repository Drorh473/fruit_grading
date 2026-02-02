from datetime import datetime
from threading import Lock

class PipelineState:
    """Thread-safe pipeline state manager"""
    
    def __init__(self):
        self._lock = Lock()
        self._state = {
            'running': False,
            'status': 'idle',
            'currentStep': 1,
            'progress': 0,
            'logs': [],
            'steps': [
                {'id': 1, 'name': 'Testing', 'status': 'pending'},
                {'id': 2, 'name': 'Database Setup', 'status': 'pending'},
                {'id': 3, 'name': 'Data Preprocessing', 'status': 'pending'},
                {'id': 4, 'name': 'Feature Extraction', 'status': 'pending'},
                {'id': 5, 'name': 'Model Training', 'status': 'pending'},
                {'id': 6, 'name': 'Evaluation', 'status': 'pending'}
            ],
            'config': {
                'hiddenDim': 16,
                'epochs': 100,
                'learningRate': 0.0005,
                'lambdaReg': 0.001,
                'batchSize': 32,
                'pcaComponents': 20,
                'dropoutRate': 0.2
            },
            'results': None,
            'pipeline_thread': None
        }
    
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
    
    def _calculate_progress(self):
        """Calculate progress based on completed steps only"""
        completed_count = sum(1 for step in self._state['steps'] if step['status'] == 'completed')
        total_steps = len(self._state['steps'])
        return int((completed_count / total_steps) * 100)
    
    def update_step(self, step_id, status):
        """Update step status - progress only updates when step completes"""
        with self._lock:
            for step in self._state['steps']:
                if step['id'] == step_id:
                    step['status'] = status
                    break
            self._state['currentStep'] = step_id
            self._state['progress'] = self._calculate_progress()
    
    def reset_pipeline(self):
        """Reset pipeline for new run"""
        with self._lock:
            self._state['running'] = False
            self._state['status'] = 'idle'
            self._state['currentStep'] = 1
            self._state['progress'] = 0
            self._state['logs'] = []
            self._state['results'] = None
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
        """Get pipeline results"""
        with self._lock:
            return self._state['results']
    
    def set_results(self, results):
        """Set pipeline results"""
        with self._lock:
            self._state['results'] = results
    
    def is_running(self):
        """Check if pipeline is running"""
        with self._lock:
            return self._state['running']
    
    def get_logs(self, limit=100):
        """Get recent logs"""
        with self._lock:
            return self._state['logs'][-limit:]

# Global pipeline state instance
pipeline_state = PipelineState()