import React, { useState, useEffect } from "react";
import {
  FiVideo,
  FiVideoOff,
  FiRefreshCw,
  FiAlertCircle,
  FiCheckCircle,
} from "react-icons/fi";
import "./CameraMonitor.css";

const CameraMonitor = ({ systemStatus }) => {
  const [cameraFeeds, setCameraFeeds] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    initializeCameras();

    // Simulate periodic updates every 5 seconds
    const interval = setInterval(() => {
      updateCameraMetrics();
      setLastUpdate(new Date());
    }, 5000);

    return () => clearInterval(interval);
  }, [systemStatus]);

  const initializeCameras = () => {
    const feeds = Array.from({ length: 4 }, (_, i) => ({
      id: i,
      name: `Camera ${i}`,
      status: systemStatus.cameras[i],
      angle: ["Front View", "Right View", "Back View", "Left View"][i],
      fps: systemStatus.cameras[i] ? 30 : 0,
      resolution: "224x224",
      lastFrame: systemStatus.cameras[i] ? new Date().toISOString() : null,
      captureSuccess: systemStatus.cameras[i] ? 99.8 - i * 0.5 : 96.2,
      quality: systemStatus.cameras[i] ? 92 - (i === 1 ? 3 : 0) : 85,
      lastError: systemStatus.cameras[i] ? null : "Connection timeout",
      uptime: systemStatus.cameras[i] ? Math.floor(Math.random() * 24) + 12 : 0,
      framesProcessed: systemStatus.cameras[i]
        ? Math.floor(Math.random() * 10000) + 50000
        : 0,
      errorCount: systemStatus.cameras[i] ? Math.floor(Math.random() * 5) : 12,
    }));
    setCameraFeeds(feeds);
  };

  const updateCameraMetrics = () => {
    setCameraFeeds((prev) =>
      prev.map((feed) => ({
        ...feed,
        framesProcessed: feed.status
          ? feed.framesProcessed + Math.floor(Math.random() * 150) + 100
          : feed.framesProcessed,
        lastFrame: feed.status ? new Date().toISOString() : feed.lastFrame,
        uptime: feed.status ? feed.uptime + 1 / 720 : feed.uptime, // Increment by 5 seconds
      }))
    );
  };

  const handleRefresh = (cameraId) => {
    setCameraFeeds((prev) =>
      prev.map((feed) =>
        feed.id === cameraId
          ? { ...feed, lastFrame: new Date().toISOString() }
          : feed
      )
    );
  };

  const handleRefreshAll = () => {
    initializeCameras();
  };

  const formatUptime = (hours) => {
    if (hours < 1) return "< 1h";
    if (hours >= 24)
      return `${Math.floor(hours / 24)}d ${Math.floor(hours % 24)}h`;
    return `${Math.floor(hours)}h`;
  };

  const getHealthStatus = (feed) => {
    if (!feed.status) return "error";
    if (feed.captureSuccess < 98) return "warning";
    if (feed.quality < 90) return "warning";
    return "good";
  };

  return (
    <div className="camera-monitor">
      <div className="page-header">
        <div>
          <h1>Camera Monitor</h1>
          <p className="page-subtitle">
            Real-time camera feed visualization and system health
          </p>
          <p className="last-update">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
        <div className="header-actions">
          <button className="btn btn-primary" onClick={handleRefreshAll}>
            <FiRefreshCw />
            Refresh All
          </button>
        </div>
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
          {cameraFeeds.map((feed) => {
            const healthStatus = getHealthStatus(feed);
            return (
              <div
                key={feed.id}
                className={`camera-health-card health-${healthStatus}`}
              >
                <div className="camera-health-header">
                  <div className="camera-health-title">
                    <span className="camera-health-name">
                      {feed.status ? <FiCheckCircle /> : <FiAlertCircle />}
                      {feed.name}
                    </span>
                    <span className="camera-health-angle">{feed.angle}</span>
                  </div>
                  <div className={`health-indicator health-${healthStatus}`}>
                    {healthStatus === "good" && "●"}
                    {healthStatus === "warning" && "◐"}
                    {healthStatus === "error" && "○"}
                  </div>
                </div>

                <div className="camera-health-metrics">
                  <div className="health-metric">
                    <span className="health-metric-label">Capture Success</span>
                    <span
                      className={`health-metric-value ${
                        feed.captureSuccess >= 99
                          ? "metric-success"
                          : feed.captureSuccess >= 97
                          ? "metric-warning"
                          : "metric-error"
                      }`}
                    >
                      {feed.captureSuccess.toFixed(1)}%
                    </span>
                  </div>

                  <div className="health-metric">
                    <span className="health-metric-label">Avg Quality</span>
                    <span
                      className={`health-metric-value ${
                        feed.quality >= 90 ? "metric-success" : "metric-warning"
                      }`}
                    >
                      {feed.quality}/100
                    </span>
                  </div>

                  <div className="health-metric">
                    <span className="health-metric-label">Uptime</span>
                    <span className="health-metric-value metric-info">
                      {formatUptime(feed.uptime)}
                    </span>
                  </div>

                  <div className="health-metric">
                    <span className="health-metric-label">
                      Frames Processed
                    </span>
                    <span className="health-metric-value metric-info">
                      {feed.framesProcessed.toLocaleString()}
                    </span>
                  </div>

                  <div className="health-metric">
                    <span className="health-metric-label">Error Count</span>
                    <span
                      className={`health-metric-value ${
                        feed.errorCount === 0
                          ? "metric-success"
                          : feed.errorCount < 5
                          ? "metric-warning"
                          : "metric-error"
                      }`}
                    >
                      {feed.errorCount}
                    </span>
                  </div>

                  <div className="health-metric">
                    <span className="health-metric-label">Last Error</span>
                    <span
                      className={`health-metric-value ${
                        !feed.lastError ? "metric-success" : "metric-warning"
                      }`}
                    >
                      {feed.lastError || "None"}
                    </span>
                  </div>
                </div>

              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default CameraMonitor;
