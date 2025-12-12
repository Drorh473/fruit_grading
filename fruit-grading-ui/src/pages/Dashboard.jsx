import React, { useState, useEffect } from 'react';
import { FiDatabase, FiCpu, FiActivity, FiCheckCircle } from 'react-icons/fi';
import './Dashboard.css';

const Dashboard = ({ systemStatus, processingStats }) => {
  const [recentResults, setRecentResults] = useState([]);

  useEffect(() => {
    // Fetch recent results from API
    fetchRecentResults();
  }, []);

  const fetchRecentResults = async () => {
    // Mock data - replace with actual API call
    setRecentResults([
      { id: 'obj0014', type: 'market', confidence: 0.95, timestamp: '2025-12-12 14:30:22' },
      { id: 'obj0013', type: 'standard', confidence: 0.88, timestamp: '2025-12-12 14:28:15' },
      { id: 'obj0012', type: 'reject', confidence: 0.92, timestamp: '2025-12-12 14:25:08' }
    ]);
  };

  const getStatusColor = (status) => {
    return status ? 'status-success' : 'status-error';
  };

  const getStatusText = (status) => {
    return status ? 'Connected' : 'Disconnected';
  };

  return (
    <div className="dashboard">
      <div className="page-header">
        <h1>System Dashboard</h1>
        <p className="page-subtitle">Monitor your fruit grading system status and performance</p>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-4">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(22, 160, 133, 0.1)' }}>
            <FiDatabase style={{ color: 'var(--accent-primary)' }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Database</p>
            <h3 className="stat-value">
              <span className={`status-badge ${getStatusColor(systemStatus.database === 'connected')}`}>
                {systemStatus.database}
              </span>
            </h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(52, 152, 219, 0.1)' }}>
            <FiCpu style={{ color: 'var(--accent-secondary)' }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Model Status</p>
            <h3 className="stat-value">
              <span className={`status-badge ${getStatusColor(systemStatus.model === 'loaded')}`}>
                {systemStatus.model}
              </span>
            </h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(243, 156, 18, 0.1)' }}>
            <FiActivity style={{ color: 'var(--warning)' }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Processed Today</p>
            <h3 className="stat-value">{processingStats.totalProcessed}</h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(39, 174, 96, 0.1)' }}>
            <FiCheckCircle style={{ color: 'var(--success)' }} />
          </div>
          <div className="stat-content">
            <p className="stat-label">Model Accuracy</p>
            <h3 className="stat-value">
              {processingStats.accuracy > 0 ? `${(processingStats.accuracy * 100).toFixed(1)}%` : 'N/A'}
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
              <div className={`camera-indicator ${status ? 'active' : 'inactive'}`}>
                <FiActivity />
              </div>
              <div>
                <p className="camera-label">Camera {index}</p>
                <span className={`status-badge ${getStatusColor(status)}`}>
                  {getStatusText(status)}
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
                  <th>Confidence</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {recentResults.map((result) => (
                  <tr key={result.id}>
                    <td><code>{result.id}</code></td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill" 
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                        <span className="confidence-text">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
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
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">Training Samples</span>
              <span className="info-value">9 objects</span>
            </div>
            <div className="info-item">
              <span className="info-label">Testing Samples</span>
              <span className="info-value">5 objects</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Images</span>
              <span className="info-value">165 images</span>
            </div>
            <div className="info-item">
              <span className="info-label">Feature Dimension</span>
              <span className="info-value">200,704</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Model Performance</h2>
          </div>
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">Architecture</span>
              <span className="info-value">ShuffleNetV2 + FC</span>
            </div>
            <div className="info-item">
              <span className="info-label">Training Accuracy</span>
              <span className="info-value">100%</span>
            </div>
            <div className="info-item">
              <span className="info-label">Test Accuracy</span>
              <span className="info-value">36.36%</span>
            </div>
            <div className="info-item">
              <span className="info-label">Classes</span>
              <span className="info-value">3 (market, standard, reject)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
