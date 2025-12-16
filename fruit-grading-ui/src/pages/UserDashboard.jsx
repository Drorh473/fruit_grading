import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FiBarChart2, FiClock, FiCheckCircle } from "react-icons/fi";
import { useAuth } from "../context/AuthContext";
import "./UserDashboard.css";

const UserDashboard = () => {
  const [recentResults, setRecentResults] = useState([]);
  const [stats, setStats] = useState({
    totalToday: 0,
    marketCount: 0,
    standardCount: 0,
    premiumCount: 0,
  });

  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    fetchRecentResults();
    fetchStats();
  }, []);

  const fetchRecentResults = async () => {
    // Mock data - replace with actual API call
    const mockResults = [
      {
        id: "obj0014",
        type: "market",
        confidence: 0.95,
        timestamp: "2025-12-12 14:30:22",
      },
      {
        id: "obj0013",
        type: "standard",
        confidence: 0.88,
        timestamp: "2025-12-12 14:28:15",
      },
      {
        id: "obj0012",
        type: "premium",
        confidence: 0.92,
        timestamp: "2025-12-12 14:25:08",
      },
      {
        id: "obj0011",
        type: "market",
        confidence: 0.89,
        timestamp: "2025-12-12 14:20:45",
      },
      {
        id: "obj0010",
        type: "standard",
        confidence: 0.91,
        timestamp: "2025-12-12 14:18:33",
      },
    ];
    setRecentResults(mockResults);
  };

  const fetchStats = async () => {
    // Mock stats - replace with actual API call
    setStats({
      totalToday: 47,
      marketCount: 28,
      standardCount: 13,
      premiumCount: 6,
    });
  };

  return (
    <div className="user-dashboard">
      <div className="page-header">
        <div>
          <h1>Welcome, {user?.username}</h1>
          <p className="page-subtitle">
            Operator Dashboard - View Classification Results
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-4">
        <div className="summary-card">
          <div
            className="summary-icon"
            style={{ background: "rgba(52, 152, 219, 0.1)" }}
          >
            <FiClock style={{ color: "var(--info)" }} />
          </div>
          <div className="summary-content">
            <p className="summary-label">Processed Today</p>
            <h3 className="summary-value">{stats.totalToday}</h3>
          </div>
        </div>

        <div className="summary-card">
          <div
            className="summary-icon"
            style={{ background: "rgba(39, 174, 96, 0.1)" }}
          >
            <FiCheckCircle style={{ color: "var(--success)" }} />
          </div>
          <div className="summary-content">
            <p className="summary-label">Market Grade</p>
            <h3 className="summary-value">{stats.marketCount}</h3>
          </div>
        </div>

        <div className="summary-card">
          <div
            className="summary-icon"
            style={{ background: "rgba(52, 152, 219, 0.1)" }}
          >
            <FiCheckCircle style={{ color: "var(--info)" }} />
          </div>
          <div className="summary-content">
            <p className="summary-label">Standard Grade</p>
            <h3 className="summary-value">{stats.standardCount}</h3>
          </div>
        </div>

        <div className="summary-card">
          <div
            className="summary-icon"
            style={{ background: "rgba(155, 89, 182, 0.1)" }}
          >
            <FiCheckCircle style={{ color: "#9b59b6" }} />
          </div>
          <div className="summary-content">
            <p className="summary-label">Premium Grade</p>
            <h3 className="summary-value">{stats.premiumCount}</h3>
          </div>
        </div>
      </div>

      {/* Recent Results */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Classification Results</h2>
          <button
            className="btn btn-primary"
            onClick={() => navigate("/results")}
          >
            <FiBarChart2 />
            View All Results
          </button>
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
                    <td>
                      <code>{result.id}</code>
                    </td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td>
                      <div className="confidence-bar-small">
                        <div
                          className="confidence-fill"
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                        <span className="confidence-text">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
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

      {/* Info Box */}
      <div className="card info-card">
        <div className="card-header">
          <h2 className="card-title">ðŸ“Š Classification Guide</h2>
        </div>
        <div className="classification-guide">
          <div className="guide-item">
            <span className="type-badge type-market">Market</span>
            <p>Premium quality fruits suitable for market sale</p>
          </div>
          <div className="guide-item">
            <span className="type-badge type-standard">Standard</span>
            <p>Good quality fruits suitable for processing</p>
          </div>
          <div className="guide-item">
            <span className="type-badge type-premium">Premium</span>
            <p>Highest quality fruits with superior characteristics</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserDashboard;
