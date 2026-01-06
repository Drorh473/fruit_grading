import React, { useState, useEffect, useCallback } from "react";
import {
  FiDownload,
  FiFilter,
  FiSearch,
  FiCalendar,
  FiFileText,
  FiTrendingUp,
  FiTrendingDown,
  FiAlertCircle,
} from "react-icons/fi";
import "./Results.css";
import {
  getResultsList,
  getKPIs,
  getQualityDistribution,
  getQualityAlerts,
  getBatches,
  exportResultsCSV,
  exportResultsPDF,
  exportResultsExcel,
  downloadCSV,
  downloadBlob,
} from "../utils/ResultsApi";

const Results = () => {
  // Data states
  const [results, setResults] = useState([]);
  const [kpis, setKpis] = useState(null);
  const [qualityDist, setQualityDist] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [batches, setBatches] = useState([]);

  // Filter states
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [filterBatch, setFilterBatch] = useState("all");

  // UI states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  // Fetch all data on mount
  useEffect(() => {
    fetchAllData();
  }, []);

  // Re-fetch results when filters change
  useEffect(() => {
    fetchResults();
  }, [searchTerm, filterType, filterBatch]);

  /**
   * Fetch all initial data
   */
  const fetchAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch all data in parallel
      const [kpisData, qualityData, alertsData, batchesData] =
        await Promise.all([
          getKPIs().catch((err) => {
            console.error("KPIs fetch failed:", err);
            return null;
          }),
          getQualityDistribution().catch((err) => {
            console.error("Quality distribution fetch failed:", err);
            return null;
          }),
          getQualityAlerts().catch((err) => {
            console.error("Alerts fetch failed:", err);
            return [];
          }),
          getBatches().catch((err) => {
            console.error("Batches fetch failed:", err);
            return [];
          }),
        ]);

      setKpis(kpisData);
      setQualityDist(qualityData);
      setAlerts(alertsData);
      setBatches(batchesData);

      // Fetch initial results
      await fetchResults();
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to load results data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  /**
   * Fetch results with current filters
   */
  const fetchResults = async () => {
    try {
      const filters = {
        search: searchTerm || undefined,
        type: filterType,
        batch: filterBatch,
        limit: 100,
      };

      const data = await getResultsList(filters);
      setResults(data.results || []);
    } catch (err) {
      console.error("Error fetching results:", err);
      // Don't show error for filter updates, just log it
      if (!results.length) {
        setError("Failed to load results. Please try again.");
      }
    }
  };

  /**
   * Export results as CSV
   */
  const handleExportCSV = async () => {
    setExporting(true);
    try {
      const filters = {
        search: searchTerm || undefined,
        type: filterType !== "all" ? filterType : undefined,
        batch: filterBatch !== "all" ? filterBatch : undefined,
      };

      const csvContent = await exportResultsCSV(filters);
      const timestamp = new Date().toISOString().split("T")[0];
      downloadCSV(csvContent, `results_${timestamp}.csv`);
    } catch (err) {
      console.error("CSV export failed:", err);
      alert("Failed to export CSV. Please try again.");
    } finally {
      setExporting(false);
    }
  };

  /**
   * Export results as PDF
   */
  const handleExportPDF = async () => {
    setExporting(true);
    try {
      const blob = await exportResultsPDF({ includeCharts: true });
      const timestamp = new Date().toISOString().split("T")[0];
      downloadBlob(blob, `results_report_${timestamp}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
      alert("Failed to export PDF. Please try again.");
    } finally {
      setExporting(false);
    }
  };

  /**
   * Export results as Excel
   */
  const handleExportExcel = async () => {
    setExporting(true);
    try {
      const blob = await exportResultsExcel({ includeCharts: true });
      const timestamp = new Date().toISOString().split("T")[0];
      downloadBlob(blob, `results_${timestamp}.xlsx`);
    } catch (err) {
      console.error("Excel export failed:", err);
      alert("Failed to export Excel. Please try again.");
    } finally {
      setExporting(false);
    }
  };

  /**
   * Handle schedule email report
   */
  const handleScheduleEmail = () => {
    alert("Email scheduling feature coming soon!");
  };

  // Loading state
  if (loading) {
    return (
      <div className="results">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="results">
        <div className="error-container">
          <FiAlertCircle size={48} />
          <h2>Error Loading Results</h2>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={fetchAllData}>
            Retry
          </button>
        </div>
      </div>
    );
  }

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
      {kpis && (
        <div
          className="kpi-grid"
          style={{ gridTemplateColumns: "repeat(3, 1fr)" }}
        >
          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Total Processed</span>
            </div>
            <div className="kpi-value">
              {kpis.totalProcessed?.toLocaleString() || 0}
            </div>
            {kpis.trends?.totalProcessed && (
              <div
                className={`kpi-trend ${
                  kpis.trends.totalProcessed.startsWith("+")
                    ? "trend-up"
                    : "trend-down"
                }`}
              >
                {kpis.trends.totalProcessed.startsWith("+") ? (
                  <FiTrendingUp />
                ) : (
                  <FiTrendingDown />
                )}
                {kpis.trends.totalProcessed} vs yesterday
              </div>
            )}
          </div>

          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Quality Rate</span>
            </div>
            <div className="kpi-value">{kpis.qualityRate || 0}%</div>
            {kpis.trends?.qualityRate && (
              <div
                className={`kpi-trend ${
                  kpis.trends.qualityRate.startsWith("+")
                    ? "trend-up"
                    : "trend-down"
                }`}
              >
                {kpis.trends.qualityRate.startsWith("+") ? (
                  <FiTrendingUp />
                ) : (
                  <FiTrendingDown />
                )}
                {kpis.trends.qualityRate} vs yesterday
              </div>
            )}
          </div>

          <div className="kpi-card">
            <div className="kpi-header">
              <span className="kpi-label">Processing Speed(per day)</span>
            </div>
            <div className="kpi-value">{kpis.processingSpeed || 0} obj/hr</div>
            {kpis.trends?.processingSpeed && (
              <div
                className={`kpi-trend ${
                  kpis.trends.processingSpeed.startsWith("+")
                    ? "trend-up"
                    : "trend-down"
                }`}
              >
                {kpis.trends.processingSpeed.startsWith("+") ? (
                  <FiTrendingUp />
                ) : (
                  <FiTrendingDown />
                )}
                {kpis.trends.processingSpeed} vs yesterday
              </div>
            )}
          </div>
        </div>
      )}

      {/* Quality Analytics Grid */}
      <div className="charts-alerts-grid">
        {/* Left Column - Quality Distribution */}
        <div className="charts-column">
          {qualityDist && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 className="card-title">Quality Distribution</h2>
                  <p className="card-subtitle">
                    Current batch classification breakdown
                  </p>
                </div>
              </div>
              <div className="pie-chart-container">
                <svg viewBox="0 0 200 200" className="pie-chart">
                  {(() => {
                    const circumference = 502.65;
                    const marketPct = qualityDist.market?.percentage || 0;
                    const standardPct = qualityDist.standard?.percentage || 0;
                    const premiumPct = qualityDist.premium?.percentage || 0;
                    const rejectPct = qualityDist.reject?.percentage || 0;

                    const marketArc = (marketPct * circumference) / 100;
                    const standardArc = (standardPct * circumference) / 100;
                    const premiumArc = (premiumPct * circumference) / 100;
                    const rejectArc = (rejectPct * circumference) / 100;

                    let currentRotation = -90;

                    return (
                      <>
                        {/* Market */}
                        {marketPct > 0 && (
                          <circle
                            cx="100"
                            cy="100"
                            r="80"
                            fill="none"
                            stroke="var(--success)"
                            strokeWidth="60"
                            strokeDasharray={`${marketArc} ${circumference}`}
                            transform={`rotate(${currentRotation} 100 100)`}
                          />
                        )}
                        {/* Standard */}
                        {standardPct > 0 && (
                          <circle
                            cx="100"
                            cy="100"
                            r="80"
                            fill="none"
                            stroke="var(--info)"
                            strokeWidth="60"
                            strokeDasharray={`${standardArc} ${circumference}`}
                            transform={`rotate(${
                              currentRotation +
                              (marketArc / circumference) * 360
                            } 100 100)`}
                          />
                        )}
                        {/* Premium */}
                        {premiumPct > 0 && (
                          <circle
                            cx="100"
                            cy="100"
                            r="80"
                            fill="none"
                            stroke="#9b59b6"
                            strokeWidth="60"
                            strokeDasharray={`${premiumArc} ${circumference}`}
                            transform={`rotate(${
                              currentRotation +
                              ((marketArc + standardArc) / circumference) * 360
                            } 100 100)`}
                          />
                        )}
                        {/* Reject */}
                        {rejectPct > 0 && (
                          <circle
                            cx="100"
                            cy="100"
                            r="80"
                            fill="none"
                            stroke="var(--error)"
                            strokeWidth="60"
                            strokeDasharray={`${rejectArc} ${circumference}`}
                            transform={`rotate(${
                              currentRotation +
                              ((marketArc + standardArc + premiumArc) /
                                circumference) *
                                360
                            } 100 100)`}
                          />
                        )}
                        {/* Center circle for donut effect */}
                        <circle
                          cx="100"
                          cy="100"
                          r="50"
                          fill="var(--bg-medium)"
                        />
                      </>
                    );
                  })()}
                </svg>

                <div className="pie-chart-legend">
                  <div className="legend-item">
                    <div
                      className="legend-color"
                      style={{ background: "var(--success)" }}
                    ></div>
                    <div className="legend-info">
                      <span className="legend-label">Market Grade</span>
                      <span className="legend-value">
                        {qualityDist.market?.count || 0} (
                        {qualityDist.market?.percentage || 0}%)
                      </span>
                    </div>
                  </div>
                  <div className="legend-item">
                    <div
                      className="legend-color"
                      style={{ background: "var(--info)" }}
                    ></div>
                    <div className="legend-info">
                      <span className="legend-label">Standard Grade</span>
                      <span className="legend-value">
                        {qualityDist.standard?.count || 0} (
                        {qualityDist.standard?.percentage || 0}%)
                      </span>
                    </div>
                  </div>
                  <div className="legend-item">
                    <div
                      className="legend-color"
                      style={{ background: "#9b59b6" }}
                    ></div>
                    <div className="legend-info">
                      <span className="legend-label">Premium Grade</span>
                      <span className="legend-value">
                        {qualityDist.premium?.count || 0} (
                        {qualityDist.premium?.percentage || 0}%)
                      </span>
                    </div>
                  </div>
                  <div className="legend-item">
                    <div
                      className="legend-color"
                      style={{ background: "var(--error)" }}
                    ></div>
                    <div className="legend-info">
                      <span className="legend-label">Reject</span>
                      <span className="legend-value">
                        {qualityDist.reject?.count || 0} (
                        {qualityDist.reject?.percentage || 0}%)
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Quality Alerts Sidebar */}
        <div className="alerts-sidebar">
          <div className="card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Quality Alerts</h2>
                <p className="card-subtitle">Requires attention</p>
              </div>
            </div>
            <div className="alert-list">
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`alert-item alert-${alert.type}`}
                  >
                    <div className="alert-content">
                      <div className="alert-title">{alert.title}</div>
                      <div className="alert-message">{alert.message}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <p
                    style={{
                      fontSize: "0.875rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    No alerts at this time
                  </p>
                </div>
              )}
            </div>
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
              Processing speed, Quality rate, Throughput
            </p>
          </div>
        </div>
      </div>

      {/* Detailed Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            Detailed Results ({results.length} items)
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
              <option value="premium">Premium</option>
              <option value="reject">Reject</option>
            </select>
          </div>
          <select
            value={filterBatch}
            onChange={(e) => setFilterBatch(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Batches</option>
            {batches.map((batch) => (
              <option key={batch} value={batch}>
                {batch}
              </option>
            ))}
          </select>
        </div>

        {results.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Batch</th>
                  <th>Classification</th>
                  <th>Image Count</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.id}>
                    <td>
                      <code>{result.id}</code>
                    </td>
                    <td>{result.batch || "N/A"}</td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td>{result.imageCount || 0}</td>
                    <td className="timestamp">{result.timestamp || "N/A"}</td>
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
          <h2 className="card-title">Export & Reporting Options</h2>
        </div>
        <div className="export-options">
          <button
            className="btn btn-secondary"
            onClick={handleExportPDF}
            disabled={exporting}
          >
            {exporting ? "Exporting..." : "Export PDF Report"}
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleExportExcel}
            disabled={exporting}
          >
            {exporting ? "Exporting..." : "Export Excel with Charts"}
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleScheduleEmail}
            disabled={exporting}
          >
            Schedule Email Report
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleExportCSV}
            disabled={exporting}
          >
            {exporting ? "Exporting..." : "Export CSV Data"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;
