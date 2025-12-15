import React, { useState, useEffect } from "react";
import { FiVideo, FiVideoOff, FiRefreshCw, FiPower } from "react-icons/fi";
import "./CameraMonitor.css";

const CameraMonitor = ({ systemStatus, setSystemStatus }) => {
  const [cameraFeeds, setCameraFeeds] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(0);

  useEffect(() => {
    // Initialize camera feeds whenever systemStatus changes
    initializeCameras();
  }, [systemStatus]);

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

  const toggleCamera = (cameraId, e) => {
    e.stopPropagation();

    const currentStatus = systemStatus.cameras[cameraId];
    const action = currentStatus ? "shutdown" : "start";

    if (
      window.confirm(`Are you sure you want to ${action} Camera ${cameraId}?`)
    ) {
      // Update system status
      const newCameras = [...systemStatus.cameras];
      newCameras[cameraId] = !newCameras[cameraId];

      setSystemStatus({
        ...systemStatus,
        cameras: newCameras,
      });

      console.log(
        `Camera ${cameraId} ${newCameras[cameraId] ? "started" : "shutdown"}`
      );
    }
  };

  const handleRefresh = (cameraId) => {
    // Refresh specific camera feed
    console.log(`Refreshing camera ${cameraId}`);
  };

  const handleRefreshAll = () => {
    // Refresh all camera feeds
    console.log("Refreshing all cameras");
  };

  const startAllCameras = () => {
    if (window.confirm("Start all cameras?")) {
      setSystemStatus({
        ...systemStatus,
        cameras: [true, true, true, true],
      });
    }
  };

  const shutdownAllCameras = () => {
    if (
      window.confirm("Shutdown all cameras? This will affect system operation.")
    ) {
      setSystemStatus({
        ...systemStatus,
        cameras: [false, false, false, false],
      });
    }
  };

  const activeCamerasCount = systemStatus.cameras.filter((c) => c).length;

  return (
    <div className="camera-monitor">
      <div className="page-header">
        <div>
          <h1>Camera Monitor</h1>
          <p className="page-subtitle">
            Real-time camera feed visualization and control
          </p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={shutdownAllCameras}>
            <FiPower />
            Shutdown All
          </button>
          <button className="btn btn-primary" onClick={startAllCameras}>
            <FiPower />
            Start All
          </button>
          <button className="btn btn-primary" onClick={handleRefreshAll}>
            <FiRefreshCw />
            Refresh All
          </button>
        </div>
      </div>

      {/* System Status Alert */}
      {activeCamerasCount < 4 && (
        <div
          className="card"
          style={{
            background: "rgba(243, 156, 18, 0.1)",
            border: "1px solid var(--warning)",
          }}
        >
          <div
            style={{
              padding: "var(--spacing-md)",
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-md)",
            }}
          >
            <FiVideoOff
              style={{ fontSize: "1.5rem", color: "var(--warning)" }}
            />
            <div>
              <strong style={{ color: "var(--warning)" }}>Warning: </strong>
              <span style={{ color: "var(--text-primary)" }}>
                {4 - activeCamerasCount} camera
                {4 - activeCamerasCount > 1 ? "s" : ""} offline. Multi-view
                fusion requires all 4 cameras for optimal classification.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Camera Grid View */}
      <div className="camera-grid">
        {cameraFeeds.map((feed) => (
          <div
            key={feed.id}
            className={`camera-card ${
              selectedCamera === feed.id ? "selected" : ""
            } ${!feed.status ? "camera-offline" : ""}`}
            onClick={() => setSelectedCamera(feed.id)}
          >
            <div className="camera-header">
              <div className="camera-info">
                <h3>{feed.name}</h3>
                <span className="camera-angle">{feed.angle}</span>
              </div>
              <div style={{ display: "flex", gap: "var(--spacing-sm)" }}>
                <button
                  className="btn-icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRefresh(feed.id);
                  }}
                  disabled={!feed.status}
                  title="Refresh camera"
                >
                  <FiRefreshCw />
                </button>
                <button
                  className={`btn-icon ${
                    !feed.status ? "btn-icon-start" : "btn-icon-shutdown"
                  }`}
                  onClick={(e) => toggleCamera(feed.id, e)}
                  title={feed.status ? "Shutdown camera" : "Start camera"}
                >
                  <FiPower />
                </button>
              </div>
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
                  <button
                    className="btn btn-sm btn-primary"
                    style={{ marginTop: "var(--spacing-md)" }}
                    onClick={(e) => toggleCamera(feed.id, e)}
                  >
                    <FiPower />
                    Start Camera
                  </button>
                </div>
              )}
            </div>

            <div className="camera-footer">
              <div className="camera-stat">
                <span className="stat-label">FPS</span>
                <span className="stat-value">
                  {feed.status ? feed.fps : "--"}
                </span>
              </div>
              <div className="camera-stat">
                <span className="stat-label">Resolution</span>
                <span className="stat-value">
                  {feed.status ? feed.resolution : "--"}
                </span>
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

      {/* Selected Camera Details */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Camera {selectedCamera} Details</h2>
          <span className="card-subtitle">
            {cameraFeeds[selectedCamera]?.angle}
          </span>
        </div>
        <div className="camera-details">
          <div className="detail-grid">
            <div className="detail-item">
              <span className="detail-label">Camera ID</span>
              <span className="detail-value">{selectedCamera}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">View Angle</span>
              <span className="detail-value">
                {cameraFeeds[selectedCamera]?.angle}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Frame Rate</span>
              <span className="detail-value">
                {cameraFeeds[selectedCamera]?.status
                  ? `${cameraFeeds[selectedCamera]?.fps} FPS`
                  : "N/A"}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Resolution</span>
              <span className="detail-value">
                {cameraFeeds[selectedCamera]?.status
                  ? cameraFeeds[selectedCamera]?.resolution
                  : "N/A"}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Status</span>
              <span
                className={`status-badge ${
                  cameraFeeds[selectedCamera]?.status
                    ? "status-success"
                    : "status-error"
                }`}
              >
                {cameraFeeds[selectedCamera]?.status ? "Active" : "Offline"}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Preprocessing</span>
              <span className="detail-value">
                {cameraFeeds[selectedCamera]?.status
                  ? "Gaussian Blur + CLAHE"
                  : "N/A"}
              </span>
            </div>
          </div>
          <div
            style={{
              marginTop: "var(--spacing-lg)",
              display: "flex",
              gap: "var(--spacing-md)",
            }}
          >
            {cameraFeeds[selectedCamera]?.status ? (
              <button
                className="btn btn-danger"
                onClick={(e) => toggleCamera(selectedCamera, e)}
              >
                <FiPower />
                Shutdown Camera {selectedCamera}
              </button>
            ) : (
              <button
                className="btn btn-primary"
                onClick={(e) => toggleCamera(selectedCamera, e)}
              >
                <FiPower />
                Start Camera {selectedCamera}
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={() => handleRefresh(selectedCamera)}
              disabled={!cameraFeeds[selectedCamera]?.status}
            >
              <FiRefreshCw />
              Refresh Feed
            </button>
          </div>
        </div>
      </div>

      {/* Camera Configuration */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Multi-View Configuration</h2>
          <span
            className={`status-badge ${
              activeCamerasCount === 4 ? "status-success" : "status-warning"
            }`}
          >
            {activeCamerasCount} / 4 Cameras Active
          </span>
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
