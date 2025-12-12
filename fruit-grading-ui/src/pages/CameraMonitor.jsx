import React, { useState, useEffect } from 'react';
import { FiVideo, FiVideoOff, FiRefreshCw } from 'react-icons/fi';
import './CameraMonitor.css';

const CameraMonitor = ({ systemStatus }) => {
  const [cameraFeeds, setCameraFeeds] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(0);

  useEffect(() => {
    // Initialize camera feeds
    initializeCameras();
  }, []);

  const initializeCameras = () => {
    const feeds = Array.from({ length: 4 }, (_, i) => ({
      id: i,
      name: `Camera ${i}`,
      status: systemStatus.cameras[i],
      angle: ['Front View', 'Right View', 'Back View', 'Left View'][i],
      fps: 30,
      resolution: '224x224',
      lastFrame: null
    }));
    setCameraFeeds(feeds);
  };

  const handleRefresh = (cameraId) => {
    // Refresh specific camera feed
    console.log(`Refreshing camera ${cameraId}`);
  };

  const handleRefreshAll = () => {
    // Refresh all camera feeds
    console.log('Refreshing all cameras');
  };

  return (
    <div className="camera-monitor">
      <div className="page-header">
        <div>
          <h1>Camera Monitor</h1>
          <p className="page-subtitle">Real-time camera feed visualization</p>
        </div>
        <button className="btn btn-primary" onClick={handleRefreshAll}>
          <FiRefreshCw />
          Refresh All
        </button>
      </div>

      {/* Camera Grid View */}
      <div className="camera-grid">
        {cameraFeeds.map((feed) => (
          <div 
            key={feed.id} 
            className={`camera-card ${selectedCamera === feed.id ? 'selected' : ''}`}
            onClick={() => setSelectedCamera(feed.id)}
          >
            <div className="camera-header">
              <div className="camera-info">
                <h3>{feed.name}</h3>
                <span className="camera-angle">{feed.angle}</span>
              </div>
              <button 
                className="btn-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRefresh(feed.id);
                }}
              >
                <FiRefreshCw />
              </button>
            </div>

            <div className="camera-feed">
              {feed.status ? (
                <div className="feed-placeholder">
                  <FiVideo size={48} />
                  <p>Camera Feed Active</p>
                  <div className="recording-indicator">
                    <span className="recording-dot" />
                    <span>LIVE</span>
                  </div>
                </div>
              ) : (
                <div className="feed-placeholder offline">
                  <FiVideoOff size={48} />
                  <p>Camera Offline</p>
                </div>
              )}
            </div>

            <div className="camera-footer">
              <div className="camera-stat">
                <span className="stat-label">FPS</span>
                <span className="stat-value">{feed.fps}</span>
              </div>
              <div className="camera-stat">
                <span className="stat-label">Resolution</span>
                <span className="stat-value">{feed.resolution}</span>
              </div>
              <div className="camera-stat">
                <span className="stat-label">Status</span>
                <span className={`status-badge ${feed.status ? 'status-success' : 'status-error'}`}>
                  {feed.status ? 'Active' : 'Offline'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Selected Camera Details */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Camera {selectedCamera} Details</h2>
          <span className="card-subtitle">{cameraFeeds[selectedCamera]?.angle}</span>
        </div>
        <div className="camera-details">
          <div className="detail-grid">
            <div className="detail-item">
              <span className="detail-label">Camera ID</span>
              <span className="detail-value">{selectedCamera}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">View Angle</span>
              <span className="detail-value">{cameraFeeds[selectedCamera]?.angle}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Frame Rate</span>
              <span className="detail-value">{cameraFeeds[selectedCamera]?.fps} FPS</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Resolution</span>
              <span className="detail-value">{cameraFeeds[selectedCamera]?.resolution}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Status</span>
              <span className={`status-badge ${cameraFeeds[selectedCamera]?.status ? 'status-success' : 'status-error'}`}>
                {cameraFeeds[selectedCamera]?.status ? 'Active' : 'Offline'}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Preprocessing</span>
              <span className="detail-value">Gaussian Blur + CLAHE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Camera Configuration */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Multi-View Configuration</h2>
        </div>
        <div className="config-info">
          <div className="config-item">
            <div className="config-icon">🔄</div>
            <div>
              <h4>Rotation Capture</h4>
              <p>Captures fruit from 4 angles during 360° rotation</p>
            </div>
          </div>
          <div className="config-item">
            <div className="config-icon">📊</div>
            <div>
              <h4>Temporal Pooling</h4>
              <p>Averages features across multiple frames per angle</p>
            </div>
          </div>
          <div className="config-item">
            <div className="config-icon">🔗</div>
            <div>
              <h4>Multi-View Fusion</h4>
              <p>Concatenates features from all 4 camera angles</p>
            </div>
          </div>
          <div className="config-item">
            <div className="config-icon">🎯</div>
            <div>
              <h4>Feature Extraction</h4>
              <p>ShuffleNetV2 pre-trained on ImageNet</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraMonitor;
