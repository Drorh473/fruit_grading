import React, { useState, useEffect } from "react";
import { FiVideo, FiVideoOff, FiRefreshCw } from "react-icons/fi";
import "./CameraMonitor.css";

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
      angle: ["Front View", "Right View", "Back View", "Left View"][i],
      fps: 30,
      resolution: "224x224",
      lastFrame: null,
    }));
    setCameraFeeds(feeds);
  };

  const handleRefresh = (cameraId) => {
    // Refresh specific camera feed
    console.log(`Refreshing camera ${cameraId}`);
  };

  const handleRefreshAll = () => {
    // Refresh all camera feeds
    console.log("Refreshing all cameras");
  };

  return (
    <div className="camera-monitor">
      <div className="page-header">
        <div>
          <h1>Camera Monitor</h1>
          <p className="page-subtitle">
            Real-time camera feed visualization and system health
          </p>
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
            className={`camera-card ${
              selectedCamera === feed.id ? "selected" : ""
            }`}
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
                <span
                  className={`status-badge ${
                    feed.status ? "status-success" : "status-error"
                  }`}
                >
                  {feed.status ? "Active" : "Offline"}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Camera System Health */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Camera System Health</h2>
          <p className="card-subtitle">
            Detailed diagnostics and quality metrics
          </p>
        </div>

        <div className="camera-health-grid">
          {cameraFeeds.map((feed) => (
            <div key={feed.id} className="camera-health-card">
              <div className="camera-health-header">
                <div className="camera-health-title">
                  <span className="camera-health-name">📹 {feed.name}</span>
                  <span className="camera-health-angle">{feed.angle}</span>
                </div>
                <div
                  className={`health-indicator ${
                    feed.status ? "health-good" : "health-error"
                  }`}
                ></div>
              </div>

              <div className="camera-health-metrics">
                <div className="health-metric">
                  <span className="health-metric-label">Frame Rate</span>
                  <span className="health-metric-value">{feed.fps} FPS</span>
                </div>
                <div className="health-metric">
                  <span className="health-metric-label">Capture Success</span>
                  <span
                    className={`health-metric-value ${
                      feed.status ? "metric-success" : "metric-warning"
                    }`}
                  >
                    {feed.status ? "99.8%" : "96.2%"}
                  </span>
                </div>
                <div className="health-metric">
                  <span className="health-metric-label">Avg Quality</span>
                  <span className="health-metric-value">
                    {feed.status
                      ? feed.id === 1
                        ? "89/100"
                        : "92/100"
                      : "85/100"}
                  </span>
                </div>
                <div className="health-metric">
                  <span className="health-metric-label">Last Error</span>
                  <span
                    className={`health-metric-value ${
                      feed.status ? "metric-success" : "metric-warning"
                    }`}
                  >
                    {feed.status ? "None" : "2 min ago"}
                  </span>
                </div>
              </div>

              <div className="camera-health-footer">
                <button className="btn btn-secondary btn-sm">View Logs</button>
                <button className="btn btn-secondary btn-sm">
                  Diagnostics
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CameraMonitor;
