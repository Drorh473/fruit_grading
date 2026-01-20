import React, { useState, useEffect } from "react";
import {
  FiDownload,
  FiFilter,
  FiSearch,
  FiCheck,
  FiX,
  FiTrendingUp,
  FiTrendingDown,
  FiAlertCircle,
} from "react-icons/fi";
import "./Results.css";
import {
  getTestPredictions,
  getKPIs,
  getQualityDistribution,
  getQualityAlerts,
  exportResultsCSV,
  downloadCSV,
} from "../utils/ResultsApi";

const Results = () => {
  // Data states
  const [predictions, setPredictions] = useState([]);
  const [predictionStats, setPredictionStats] = useState({
    total: 0,
    accuracy: 0,
    correct_count: 0,
  });
  const [kpis, setKpis] = useState(null);
  const [qualityDist, setQualityDist] = useState(null);
  const [alerts, setAlerts] = useState([]);

  // Filter states
  const [searchTerm, setSearchTerm] = useState("");
  const [filterActual, setFilterActual] = useState("all");
  const [filterPredicted, setFilterPredicted] = useState("all");
  const [filterCorrect, setFilterCorrect] = useState("all");

  // UI states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  // Fetch all data on mount
  useEffect(() => {
    fetchAllData();
  }, []);

  // Re-fetch predictions when filters change
  useEffect(() => {
    fetchPredictions();
  }, [searchTerm, filterActual, filterPredicted, filterCorrect]);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [kpisData, qualityData, alertsData] = await Promise.all([
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
      ]);

      setKpis(kpisData);
      setQualityDist(qualityData);
      setAlerts(alertsData);

      await fetchPredictions();
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to load results data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictions = async () => {
    try {
      const filters = {
        search: searchTerm || undefined,
        actual: filterActual,
        predicted: filterPredicted,
        correct: filterCorrect,
      };

      const data = await getTestPredictions(filters);
      setPredictions(data.predictions || []);
      setPredictionStats({
        total: data.total || 0,
        accuracy: data.accuracy || 0,
        correct_count: data.correct_count || 0,
      });
    } catch (err) {
      console.error("Error fetching predictions:", err);
      if (!predictions.length) {
        setError("Failed to load predictions. Please try again.");
      }
    }
  };

  const handleExportCSV = async () => {
    setExporting(true);
    try {
      const csvContent = await exportResultsCSV();
      const timestamp = new Date().toISOString().split("T")[0];
      downloadCSV(csvContent, `results_${timestamp}.csv`);
    } catch (err) {
      console.error("CSV export failed:", err);
      alert("Failed to export CSV. Please try again.");
    } finally {
      setExporting(false);
    }
  };

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
          <p className="page-subtitle">Model predictions on test set objects</p>
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
              <span className="kpi-label">Model Accuracy</span>
            </div>
            <div className="kpi-value">{predictionStats.accuracy}%</div>
            <div className="kpi-trend trend-neutral">
              {predictionStats.correct_count}/{predictionStats.total} correct
            </div>
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
        </div>
      )}

      {/* Main Content Grid */}
      <div className="results-grid">
        {/* Left Column - Quality Distribution */}
        <div className="charts-column">
          {qualityDist && (
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 className="card-title">Quality Distribution</h2>
                  <p className="card-subtitle">By grade classification</p>
                </div>
              </div>
              <div className="chart-container">
                <div className="donut-chart">
                  <div className="donut-center">
                    <span className="donut-value">
                      {(qualityDist.market?.count || 0) +
                        (qualityDist.standard?.count || 0) +
                        (qualityDist.premium?.count || 0)}
                    </span>
                    <span className="donut-label">Total</span>
                  </div>
                </div>
                <div className="chart-legend">
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
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Alerts */}
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

      {/* Test Predictions Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            Test Set Predictions ({predictionStats.total} objects)
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
              placeholder="Search by Object ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
          <div className="filter-group">
            <FiFilter />
            <select
              value={filterActual}
              onChange={(e) => setFilterActual(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Actual</option>
              <option value="market">Market</option>
              <option value="standard">Standard</option>
              <option value="premium">Premium</option>
            </select>
          </div>
          <select
            value={filterPredicted}
            onChange={(e) => setFilterPredicted(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Predicted</option>
            <option value="market">Market</option>
            <option value="standard">Standard</option>
            <option value="premium">Premium</option>
          </select>
          <select
            value={filterCorrect}
            onChange={(e) => setFilterCorrect(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Results</option>
            <option value="correct">Correct Only</option>
            <option value="incorrect">Incorrect Only</option>
          </select>
        </div>

        {predictions.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Actual Label</th>
                  <th>Predicted Label</th>
                  <th>Confidence</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((pred) => (
                  <tr
                    key={pred.object_id}
                    className={pred.correct ? "" : "row-incorrect"}
                  >
                    <td>
                      <code>{pred.object_id}</code>
                    </td>
                    <td>
                      <span className={`type-badge type-${pred.actual_label}`}>
                        {pred.actual_label}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`type-badge type-${pred.predicted_label}`}
                      >
                        {pred.predicted_label}
                      </span>
                    </td>
                    <td>
                      <div className="confidence-cell">
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{ width: `${pred.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="confidence-value">
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td>
                      {pred.correct ? (
                        <span className="result-badge result-correct">
                          <FiCheck /> Correct
                        </span>
                      ) : (
                        <span className="result-badge result-incorrect">
                          <FiX /> Wrong
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p>No predictions found. Run the model pipeline first.</p>
          </div>
        )}
      </div>

      {/* Export Options */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Export Options</h2>
        </div>
        <div className="export-options">
          <button
            className="btn btn-secondary"
            onClick={handleExportCSV}
            disabled={exporting}
          >
            <FiDownload />
            {exporting ? "Exporting..." : "Export CSV Data"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;
