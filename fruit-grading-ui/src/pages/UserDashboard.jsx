import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiBarChart2,
  FiClock,
  FiCheckCircle,
  FiAlertCircle,
  FiRefreshCw,
} from "react-icons/fi";
import { useAuth } from "../context/AuthContext";
import { getUserDashboardStats, getRecentResults } from "../utils/api";
import "./UserDashboard.css";

const UserDashboard = () => {
  const [recentResults, setRecentResults] = useState([]);
  const [stats, setStats] = useState({
    totalToday: 0,
    marketCount: 0,
    standardCount: 0,
    rejectCount: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const navigate = useNavigate();
  const { user } = useAuth();

  // Fetch data on mount
  useEffect(() => {
    fetchData();
  }, []);

  // Fetch all dashboard data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch stats and recent results in parallel
      const [statsData, resultsData] = await Promise.all([
        getUserDashboardStats(), // Note: DashboardStats
        getRecentResults(),
      ]);

      setStats(statsData);
      setRecentResults(resultsData);
    } catch (err) {
      try {
        const data = await getUserDashboardStats();
        setStats(data);
      } catch (err) {
        console.error("Failed:", err);
        setError("Failed to load dashboard data. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  // Loading state
  if (loading) {
    return (
      <div className="user-dashboard">
        <div className="page-header">
          <div>
            <h1>Welcome, {user?.username}</h1>
            <p className="page-subtitle">Loading dashboard data...</p>
          </div>
        </div>
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "3rem",
            }}
          >
            <div className="spinner"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="user-dashboard">
      <div className="page-header">
        <div>
          <h1>Welcome, {user?.username}</h1>
          <p className="page-subtitle">
            Operator Dashboard - View Classification Results
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <FiRefreshCw className={refreshing ? "spinning" : ""} />
          {refreshing ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="alert alert-error">
          <FiAlertCircle />
          <span>{error}</span>
          <button
            className="btn btn-sm btn-secondary"
            onClick={() => setError(null)}
            style={{ marginLeft: "auto" }}
          >
            Dismiss
          </button>
        </div>
      )}

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
            style={{ background: "rgba(231, 76, 60, 0.1)" }}
          >
            <FiCheckCircle style={{ color: "var(--error)" }} />
          </div>
          <div className="summary-content">
            <p className="summary-label">Rejected</p>
            <h3 className="summary-value">{stats.rejectCount}</h3>
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
            <p
              style={{
                fontSize: "0.875rem",
                color: "var(--text-secondary)",
                marginTop: "0.5rem",
              }}
            >
              Results will appear here once fruits are processed
            </p>
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="card info-card">
        <div className="card-header">
          <h2 className="card-title">📊 Classification Guide</h2>
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
            <span className="type-badge type-reject">Reject</span>
            <p>Fruits that do not meet quality standards</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserDashboard;
