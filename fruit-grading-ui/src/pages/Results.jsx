import React, { useState, useEffect } from "react";
import {
  FiDownload,
  FiFilter,
  FiSearch,
  FiCalendar,
  FiFileText,
  FiTrendingUp,
  FiTrendingDown,
} from "react-icons/fi";
import "./Results.css";

const Results = () => {
  const [results, setResults] = useState([]);
  const [filteredResults, setFilteredResults] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [filterBatch, setFilterBatch] = useState("all");
  const [filterConfidence, setFilterConfidence] = useState("all");

  // KPI Data
  const [kpis, setKpis] = useState({
    totalProcessed: 1247,
    qualityRate: 0.942,
    avgConfidence: 0.913,
    processingSpeed: 4.2,
    trends: {
      totalProcessed: 0.125,
      qualityRate: 0.021,
      avgConfidence: -0.012,
      processingSpeed: 0.3,
    },
  });

  // Quality Distribution
  const [qualityDist, setQualityDist] = useState({
    market: { count: 645, percentage: 52 },
    standard: { count: 524, percentage: 42 },
    reject: { count: 78, percentage: 6 },
  });

  // Alerts
  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: "warning",
      title: "Low Confidence Batch",
      message: "Batch #247 has avg confidence of 67%",
      icon: "⚠️",
    },
    {
      id: 2,
      type: "error",
      title: "High Rejection Rate",
      message: "Last hour: 15% rejection (3x normal)",
      icon: "🔴",
    },
    {
      id: 3,
      type: "info",
      title: "Performance Insight",
      message: "Peak efficiency at 10-11 AM (4.8 fruits/min)",
      icon: "ℹ️",
    },
  ]);

  useEffect(() => {
    fetchResults();
  }, []);

  useEffect(() => {
    filterResults();
  }, [searchTerm, filterType, filterBatch, filterConfidence, results]);

  const fetchResults = async () => {
    // Mock data - replace with actual API call
    const mockResults = [
      {
        id: "obj0247",
        batch: "#247",
        type: "market",
        confidence: 0.952,
        processingTime: 4.1,
        timestamp: "2025-12-15 14:30:22",
      },
      {
        id: "obj0246",
        batch: "#247",
        type: "standard",
        confidence: 0.887,
        processingTime: 3.9,
        timestamp: "2025-12-15 14:30:18",
      },
      {
        id: "obj0245",
        batch: "#246",
        type: "reject",
        confidence: 0.924,
        processingTime: 4.2,
        timestamp: "2025-12-15 14:30:14",
      },
      {
        id: "obj0244",
        batch: "#246",
        type: "market",
        confidence: 0.673,
        processingTime: 4.5,
        timestamp: "2025-12-15 14:30:09",
      },
      {
        id: "obj0243",
        batch: "#246",
        type: "standard",
        confidence: 0.911,
        processingTime: 4.0,
        timestamp: "2025-12-15 14:30:05",
      },
      {
        id: "obj0242",
        batch: "#245",
        type: "market",
        confidence: 0.887,
        processingTime: 3.8,
        timestamp: "2025-12-15 14:29:58",
      },
      {
        id: "obj0241",
        batch: "#245",
        type: "standard",
        confidence: 0.856,
        processingTime: 4.1,
        timestamp: "2025-12-15 14:29:52",
      },
      {
        id: "obj0240",
        batch: "#245",
        type: "reject",
        confidence: 0.903,
        processingTime: 4.3,
        timestamp: "2025-12-15 14:29:45",
      },
    ];
    setResults(mockResults);
  };

  const filterResults = () => {
    let filtered = results;

    if (searchTerm) {
      filtered = filtered.filter(
        (r) =>
          r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          r.batch.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType !== "all") {
      filtered = filtered.filter((r) => r.type === filterType);
    }

    if (filterBatch !== "all") {
      filtered = filtered.filter((r) => r.batch === filterBatch);
    }

    if (filterConfidence !== "all") {
      filtered = filtered.filter((r) => {
        if (filterConfidence === "high") return r.confidence > 0.9;
        if (filterConfidence === "medium")
          return r.confidence >= 0.7 && r.confidence <= 0.9;
        if (filterConfidence === "low") return r.confidence < 0.7;
        return true;
      });
    }

    setFilteredResults(filtered);
  };

  const exportResults = () => {
    const csv = [
      [
        "Object ID",
        "Batch",
        "Type",
        "Confidence",
        "Processing Time",
        "Timestamp",
      ],
      ...filteredResults.map((r) => [
        r.id,
        r.batch,
        r.type,
        r.confidence,
        r.processingTime,
        r.timestamp,
      ]),
    ]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "results.csv";
    a.click();
  };

  return (
    <div className="results">
      <div className="page-header">
        <div>
          <h1>Classification Results</h1>
          <p className="page-subtitle">
            Production metrics, quality trends, and business intelligence
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Total Processed</span>
            <div
              className="kpi-icon"
              style={{
                background: "rgba(22, 160, 133, 0.1)",
                color: "var(--accent-primary)",
              }}
            >
              🎯
            </div>
          </div>
          <div className="kpi-value">
            {kpis.totalProcessed.toLocaleString()}
          </div>
          <div
            className={`kpi-trend ${
              kpis.trends.totalProcessed > 0 ? "trend-up" : "trend-down"
            }`}
          >
            {kpis.trends.totalProcessed > 0 ? (
              <FiTrendingUp />
            ) : (
              <FiTrendingDown />
            )}
            {Math.abs(kpis.trends.totalProcessed * 100).toFixed(1)}% vs
            yesterday
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Quality Rate</span>
            <div
              className="kpi-icon"
              style={{
                background: "rgba(39, 174, 96, 0.1)",
                color: "var(--success)",
              }}
            >
              ✓
            </div>
          </div>
          <div className="kpi-value">
            {(kpis.qualityRate * 100).toFixed(1)}%
          </div>
          <div
            className={`kpi-trend ${
              kpis.trends.qualityRate > 0 ? "trend-up" : "trend-down"
            }`}
          >
            {kpis.trends.qualityRate > 0 ? (
              <FiTrendingUp />
            ) : (
              <FiTrendingDown />
            )}
            {Math.abs(kpis.trends.qualityRate * 100).toFixed(1)}% vs yesterday
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Avg Confidence</span>
            <div
              className="kpi-icon"
              style={{
                background: "rgba(52, 152, 219, 0.1)",
                color: "var(--info)",
              }}
            >
              📈
            </div>
          </div>
          <div className="kpi-value">
            {(kpis.avgConfidence * 100).toFixed(1)}%
          </div>
          <div
            className={`kpi-trend ${
              kpis.trends.avgConfidence > 0 ? "trend-up" : "trend-down"
            }`}
          >
            {kpis.trends.avgConfidence > 0 ? (
              <FiTrendingUp />
            ) : (
              <FiTrendingDown />
            )}
            {Math.abs(kpis.trends.avgConfidence * 100).toFixed(1)}% vs yesterday
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-label">Processing Speed</span>
            <div
              className="kpi-icon"
              style={{
                background: "rgba(243, 156, 18, 0.1)",
                color: "var(--warning)",
              }}
            >
              ⚡
            </div>
          </div>
          <div className="kpi-value">{kpis.processingSpeed.toFixed(1)}s</div>
          <div
            className={`kpi-trend ${
              kpis.trends.processingSpeed > 0 ? "trend-down" : "trend-up"
            }`}
          >
            {kpis.trends.processingSpeed > 0 ? (
              <FiTrendingUp />
            ) : (
              <FiTrendingDown />
            )}
            {Math.abs(kpis.trends.processingSpeed).toFixed(1)}s vs yesterday
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="chart-grid">
        {/* Time Series Chart */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Hourly Processing Trend</h2>
              <p className="card-subtitle">Last 24 hours</p>
            </div>
          </div>
          <div className="chart-container">
            <div className="chart-placeholder">
              <p style={{ fontSize: "2rem", marginBottom: "8px" }}>📈</p>
              <p
                style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}
              >
                Line Chart
              </p>
              <p
                style={{
                  fontSize: "0.75rem",
                  marginTop: "8px",
                  color: "var(--text-secondary)",
                }}
              >
                Shows fruits processed per hour
                <br />
                with quality rate overlay
              </p>
            </div>
          </div>
        </div>

        {/* Quality Distribution */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Quality Distribution</h2>
              <p className="card-subtitle">Today's breakdown</p>
            </div>
          </div>
          <div className="quality-bars">
            <div className="quality-item type-market">
              <div className="quality-header">
                <span className="quality-label">Market Grade</span>
                <span
                  className="quality-value"
                  style={{ color: "var(--success)" }}
                >
                  {qualityDist.market.count}
                </span>
              </div>
              <div className="quality-bar">
                <div
                  className="quality-fill"
                  style={{
                    width: `${qualityDist.market.percentage}%`,
                    background: "var(--success)",
                  }}
                ></div>
              </div>
              <span
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                {qualityDist.market.percentage}% of total
              </span>
            </div>

            <div className="quality-item type-standard">
              <div className="quality-header">
                <span className="quality-label">Standard Grade</span>
                <span
                  className="quality-value"
                  style={{ color: "var(--info)" }}
                >
                  {qualityDist.standard.count}
                </span>
              </div>
              <div className="quality-bar">
                <div
                  className="quality-fill"
                  style={{
                    width: `${qualityDist.standard.percentage}%`,
                    background: "var(--info)",
                  }}
                ></div>
              </div>
              <span
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                {qualityDist.standard.percentage}% of total
              </span>
            </div>

            <div className="quality-item type-reject">
              <div className="quality-header">
                <span className="quality-label">Rejected</span>
                <span
                  className="quality-value"
                  style={{ color: "var(--error)" }}
                >
                  {qualityDist.reject.count}
                </span>
              </div>
              <div className="quality-bar">
                <div
                  className="quality-fill"
                  style={{
                    width: `${qualityDist.reject.percentage}%`,
                    background: "var(--error)",
                  }}
                ></div>
              </div>
              <span
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                {qualityDist.reject.percentage}% of total
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Analysis & Alerts */}
      <div className="chart-grid">
        {/* Confidence Distribution */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Confidence Score Distribution</h2>
              <p className="card-subtitle">Classification reliability</p>
            </div>
          </div>
          <div className="chart-container">
            <div className="chart-placeholder">
              <p style={{ fontSize: "2rem", marginBottom: "8px" }}>📊</p>
              <p
                style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}
              >
                Histogram Chart
              </p>
              <p
                style={{
                  fontSize: "0.75rem",
                  marginTop: "8px",
                  color: "var(--text-secondary)",
                }}
              >
                Shows distribution of confidence scores
                <br />
                0-100% with bucket sizes
              </p>
            </div>
          </div>
        </div>

        {/* Quality Alerts */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Quality Alerts</h2>
              <p className="card-subtitle">Requires attention</p>
            </div>
          </div>
          <div className="alert-list">
            {alerts.map((alert) => (
              <div key={alert.id} className={`alert-item alert-${alert.type}`}>
                <span className="alert-icon">{alert.icon}</span>
                <div className="alert-content">
                  <div className="alert-title">{alert.title}</div>
                  <div className="alert-message">{alert.message}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Batch Comparison */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">Batch Performance Comparison</h2>
            <p className="card-subtitle">Current vs Historical Average</p>
          </div>
        </div>
        <div className="chart-container">
          <div className="chart-placeholder">
            <p style={{ fontSize: "2rem", marginBottom: "8px" }}>📊</p>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
              Bar Chart Comparison
            </p>
            <p
              style={{
                fontSize: "0.75rem",
                marginTop: "8px",
                color: "var(--text-secondary)",
              }}
            >
              Compare key metrics:
              <br />
              Processing speed, Quality rate, Confidence, Throughput
            </p>
          </div>
        </div>
      </div>

      {/* Detailed Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            Detailed Results ({filteredResults.length} items)
          </h2>
        </div>

        {/* Filters */}
        <div
          className="filters-container"
          style={{ marginBottom: "var(--spacing-lg)" }}
        >
          <div className="search-box">
            <FiSearch />
            <input
              type="text"
              placeholder="Search by Object ID, Batch..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
          <div className="filter-group">
            <FiFilter />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Types</option>
              <option value="market">Market</option>
              <option value="standard">Standard</option>
              <option value="reject">Reject</option>
            </select>
          </div>
          <select
            value={filterBatch}
            onChange={(e) => setFilterBatch(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Batches</option>
            <option value="#247">Batch #247</option>
            <option value="#246">Batch #246</option>
            <option value="#245">Batch #245</option>
          </select>
          <select
            value={filterConfidence}
            onChange={(e) => setFilterConfidence(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Confidence</option>
            <option value="high">High (&gt;90%)</option>
            <option value="medium">Medium (70-90%)</option>
            <option value="low">Low (&lt;70%)</option>
          </select>
        </div>

        {filteredResults.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Batch</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Processing Time</th>
                  <th>Timestamp</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((result) => (
                  <tr key={result.id}>
                    <td>
                      <code>{result.id}</code>
                    </td>
                    <td>{result.batch}</td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td>{(result.confidence * 100).toFixed(1)}%</td>
                    <td>{result.processingTime}s</td>
                    <td className="timestamp">{result.timestamp}</td>
                    <td>
                      {result.confidence < 0.7 ? (
                        <a href="#" style={{ color: "var(--warning)" }}>
                          ⚠️ Review
                        </a>
                      ) : (
                        <a href="#" style={{ color: "var(--accent-primary)" }}>
                          View
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p>No results found</p>
          </div>
        )}
      </div>

      {/* Export Options */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">📤 Export & Reporting Options</h2>
        </div>
        <div className="export-options">
          <button className="btn btn-secondary">📄 Export PDF Report</button>
          <button className="btn btn-secondary">
            📊 Export Excel with Charts
          </button>
          <button className="btn btn-secondary">
            📧 Schedule Email Report
          </button>
          <button className="btn btn-secondary">📋 Export CSV Data</button>
        </div>
      </div>
    </div>
  );
};

export default Results;
