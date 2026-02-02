import React, { useState, useEffect, useRef } from "react";
import {
  FiDatabase,
  FiCpu,
  FiActivity,
  FiCheckCircle,
  FiRefreshCw,
} from "react-icons/fi";
import {
  getSystemStatus,
  getProcessingStats,
  getRecentResults,
  getDatasetInfo,
  getModelPerformance,
} from "../utils/AdminDashboardApi";
import "./Dashboard.css";

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState({
    database: "loading",
    model: "loading",
    cameras: [false, false, false, false],
  });
  const [processingStats, setProcessingStats] = useState({
    totalProcessed: 0,
    accuracy: 0,
    lastUpdate: null,
  });
  const [recentResults, setRecentResults] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Track last known model timestamp to detect changes
  const lastModelTimestamp = useRef(null);
  const autoRefreshInterval = useRef(null);

  useEffect(() => {
    loadDashboardData();

    // Smart auto-refresh: only refresh system status and processing stats
    // Model data doesn't change unless pipeline runs
    autoRefreshInterval.current = setInterval(() => {
      refreshLiveData();
    }, 30000); // 30 seconds

    return () => {
      if (autoRefreshInterval.current) {
        clearInterval(autoRefreshInterval.current);
      }
    };
  }, []);

  const loadDashboardData = async () => {
    try {
      setError(null);

      // Load all data (full refresh)
      const [statusData, statsData, resultsData, datasetData, modelData] =
        await Promise.all([
          getSystemStatus(),
          getProcessingStats(),
          getRecentResults(3),
          getDatasetInfo(),
          getModelPerformance(),
        ]);

      setSystemStatus(statusData);
      setProcessingStats(statsData);
      setRecentResults(resultsData);
      setDatasetInfo(datasetData);
      setModelPerformance(modelData);

      // Track model timestamp
      if (statsData.lastUpdate) {
        lastModelTimestamp.current = statsData.lastUpdate;
      }
    } catch (err) {
      setError(
        `Failed to load dashboard data: ${err.message || "Please try again."}`
      );
    } finally {
      setLoading(false);
    }
  };

  const refreshLiveData = async () => {
    // Only refresh dynamic data (system status, processing stats, recent results)
    // Skip model performance and dataset info unless model changed
    try {
      const [statusData, statsData, resultsData] = await Promise.all([
        getSystemStatus(),
        getProcessingStats(),
        getRecentResults(3),
      ]);

      setSystemStatus(statusData);
      setProcessingStats(statsData);
      setRecentResults(resultsData);

      if (
        statsData.lastUpdate &&
        statsData.lastUpdate !== lastModelTimestamp.current
      ) {
        lastModelTimestamp.current = statsData.lastUpdate;

        // Reload model-specific data only when model changes
        const [datasetData, modelData] = await Promise.all([
          getDatasetInfo(),
          getModelPerformance(),
        ]);

        setDatasetInfo(datasetData);
        setModelPerformance(modelData);
      }
    } catch (err) {
      // Silent fail for auto-refresh
    }
  };

  const handleRefresh = () => {
    setLoading(true);
    loadDashboardData();
  };

  const getStatusColor = (status) => {
    if (status === "connected" || status === "loaded") return "status-success";
    if (status === "loading") return "status-info";
    return "status-error";
  };

  const getStatusText = (status) => {
    if (status === "connected" || status === "loaded") return status;
    if (status === "loading") return "checking...";
    return "disconnected";
  };

  if (loading && systemStatus.database === "loading") {
    return (
      <div className="dashboard">
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "400px",
          }}
        >
          <div className="spinner" />
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <h1>System Dashboard</h1>
          <p className="page-subtitle">
            Monitor your fruit grading system status and performance
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={handleRefresh}
          disabled={loading}
        >
          <FiRefreshCw className={loading ? "spinning" : ""} />
          Refresh
        </button>
      </div>

      {error && (
        <div
          className="card"
          style={{
            background: "rgba(231, 76, 60, 0.1)",
            border: "1px solid var(--error)",
          }}
        >
          <p style={{ color: "var(--error)", margin: 0 }}>{error}</p>
        </div>
      )}

      {/* System Status Cards */}
      <div className="grid grid-4">
        <div className="stat-card">
          <div
            className="stat-icon"
            style={{ background: "rgba(22, 160, 133, 0.1)" }}
          >
            <FiDatabase style={{ color: "var(--accent-primary)" }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Database</p>
            <h3 className="stat-value">
              <span
                className={`status-badge ${getStatusColor(
                  systemStatus.database
                )}`}
              >
                {getStatusText(systemStatus.database)}
              </span>
            </h3>
          </div>
        </div>

        <div className="stat-card">
          <div
            className="stat-icon"
            style={{ background: "rgba(52, 152, 219, 0.1)" }}
          >
            <FiCpu style={{ color: "var(--accent-secondary)" }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Model Status</p>
            <h3 className="stat-value">
              <span
                className={`status-badge ${getStatusColor(systemStatus.model)}`}
              >
                {getStatusText(systemStatus.model)}
              </span>
            </h3>
          </div>
        </div>

        <div className="stat-card">
          <div
            className="stat-icon"
            style={{ background: "rgba(243, 156, 18, 0.1)" }}
          >
            <FiActivity style={{ color: "var(--warning)" }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Processed Today</p>
            <h3 className="stat-value">{processingStats.totalProcessed}</h3>
          </div>
        </div>

        <div className="stat-card">
          <div
            className="stat-icon"
            style={{ background: "rgba(39, 174, 96, 0.1)" }}
          >
            <FiCheckCircle style={{ color: "var(--success)" }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Model Accuracy</p>
            <h3 className="stat-value">
              {processingStats.accuracy > 0
                ? `${(processingStats.accuracy * 100).toFixed(1)}%`
                : "N/A"}
            </h3>
          </div>
        </div>
      </div>

      {/* Camera Status Grid */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Camera Status</h2>
        </div>
        <div className="grid grid-4">
          {systemStatus.cameras.map((status, index) => (
            <div key={index} className="camera-status-item">
              <div
                className={`camera-indicator ${status ? "active" : "inactive"}`}
              >
                <FiActivity />
              </div>
              <div>
                <p className="camera-label">Camera {index}</p>
                <span
                  className={`status-badge ${
                    status ? "status-success" : "status-error"
                  }`}
                >
                  {status ? "Active" : "Offline"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Results */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Processing Results</h2>
          <span className="card-subtitle">Latest classified fruits</span>
        </div>
        {recentResults.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Classification</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {recentResults.map((result) => (
                  <tr key={result.id}>
                    <td>
                      <code>{result.id}</code>
                    </td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td className="timestamp">{result.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p>No recent results available</p>
          </div>
        )}
      </div>

      {/* System Information */}
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Dataset Information</h2>
          </div>
          {datasetInfo ? (
            <div className="info-list">
              <div className="info-item">
                <span className="info-label">Training Samples</span>
                <span className="info-value">
                  {datasetInfo.trainingCount} objects
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Testing Samples</span>
                <span className="info-value">
                  {datasetInfo.testingCount} objects
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Total Images</span>
                <span className="info-value">
                  {datasetInfo.totalImages} images
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Feature Dimension</span>
                <span className="info-value">
                  {datasetInfo.featureDim.toLocaleString()}
                </span>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>Loading dataset info...</p>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Model Performance</h2>
          </div>
          {modelPerformance ? (
            <div className="info-list">
              <div className="info-item">
                <span className="info-label">Architecture</span>
                <span className="info-value">
                  {modelPerformance.architecture}
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Training Accuracy</span>
                <span className="info-value">
                  {(modelPerformance.trainAccuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Test Accuracy</span>
                <span className="info-value">
                  {(modelPerformance.testAccuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Classes</span>
                <span className="info-value">
                  {modelPerformance.classes} (market, standard, premium)
                </span>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>Loading model performance...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
