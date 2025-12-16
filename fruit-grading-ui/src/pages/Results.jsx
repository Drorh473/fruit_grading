import React, { useState, useEffect } from "react";
import {
  FiDownload,
  FiFilter,
  FiSearch,
  FiAlertCircle,
  FiRefreshCw,
} from "react-icons/fi";
import {
  getResultsList,
  getConfusionMatrix,
  exportResults,
  downloadCSV,
} from "../utils/api";
import "./Results.css";

const Results = () => {
  const [results, setResults] = useState([]);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [filters, setFilters] = useState({
    search: "",
    type: "all",
    limit: 50,
    offset: 0,
  });
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadResults();
  }, [filters]);

  useEffect(() => {
    loadConfusionMatrix();
  }, []);

  const loadResults = async () => {
    try {
      setError(null);
      setLoading(true);

      const data = await getResultsList(filters);
      setResults(data.results);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to load results:", err);
      setError("Failed to load results. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const loadConfusionMatrix = async () => {
    try {
      const data = await getConfusionMatrix();
      setConfusionMatrix(data);
    } catch (err) {
      console.error("Failed to load confusion matrix:", err);
    }
  };

  const handleSearch = (value) => {
    setFilters((prev) => ({ ...prev, search: value, offset: 0 }));
  };

  const handleTypeFilter = (type) => {
    setFilters((prev) => ({ ...prev, type, offset: 0 }));
  };

  const handleExport = async () => {
    try {
      setExporting(true);
      const csv = await exportResults({
        search: filters.search,
        type: filters.type,
      });
      downloadCSV(csv, `results_${new Date().toISOString().split("T")[0]}.csv`);
    } catch (err) {
      console.error("Failed to export results:", err);
      setError("Failed to export results. Please try again.");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="results">
      <div className="page-header">
        <div>
          <h1>Classification Results</h1>
          <p className="page-subtitle">
            View and analyze fruit grading results
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button
            className="btn btn-secondary"
            onClick={loadResults}
            disabled={loading}
          >
            <FiRefreshCw />
            Refresh
          </button>
          <button
            className="btn btn-primary"
            onClick={handleExport}
            disabled={exporting || results.length === 0}
          >
            {exporting ? (
              <>
                <div className="spinner-small" />
                Exporting...
              </>
            ) : (
              <>
                <FiDownload />
                Export Results
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: "1.5rem" }}>
          <FiAlertCircle />
          <span>{error}</span>
        </div>
      )}

      {/* Filters */}
      <div className="card">
        <div className="filters-container">
          <div className="search-box">
            <FiSearch />
            <input
              type="text"
              placeholder="Search by Object ID..."
              value={filters.search}
              onChange={(e) => handleSearch(e.target.value)}
              className="search-input"
            />
          </div>
          <div className="filter-group">
            <FiFilter />
            <select
              value={filters.type}
              onChange={(e) => handleTypeFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Types</option>
              <option value="market">Market</option>
              <option value="standard">Standard</option>
              <option value="reject">Reject</option>
            </select>
          </div>
          <span
            style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}
          >
            {total} result{total !== 1 ? "s" : ""} found
          </span>
        </div>
      </div>

      {/* Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            Results ({results.length} of {total})
          </h2>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "3rem" }}>
            <div className="spinner" style={{ margin: "0 auto" }}></div>
            <p style={{ marginTop: "1rem", color: "var(--text-secondary)" }}>
              Loading results...
            </p>
          </div>
        ) : results.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Classification</th>
                  <th>Images</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.id}>
                    <td>
                      <code>{result.id}</code>
                    </td>
                    <td>
                      <span className={`type-badge type-${result.type}`}>
                        {result.type}
                      </span>
                    </td>
                    <td>{result.images}</td>
                    <td className="timestamp">{result.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p>No results found matching your filters</p>
          </div>
        )}
      </div>

      {/* Confusion Matrix */}
      {confusionMatrix && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Confusion Matrix</h2>
            <span className="card-subtitle">
              Model classification performance
            </span>
          </div>
          <div className="confusion-matrix-container">
            <div className="matrix-grid">
              <div className="matrix-header"></div>
              {confusionMatrix.classes.map((cls) => (
                <div key={cls} className="matrix-header">
                  <span className={`type-badge type-${cls}`}>{cls}</span>
                </div>
              ))}
              {confusionMatrix.matrix.map((row, i) => (
                <React.Fragment key={i}>
                  <div className="matrix-row-header">
                    <span
                      className={`type-badge type-${confusionMatrix.classes[i]}`}
                    >
                      {confusionMatrix.classes[i]}
                    </span>
                  </div>
                  {row.map((value, j) => (
                    <div
                      key={j}
                      className={`matrix-cell ${i === j ? "diagonal" : ""}`}
                      style={{
                        background:
                          i === j
                            ? `rgba(39, 174, 96, ${value / 10})`
                            : `rgba(231, 76, 60, ${value / 10})`,
                      }}
                    >
                      {value}
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>

          {/* Metrics */}
          <div className="metrics-grid">
            {Object.entries(confusionMatrix.metrics).map(
              ([className, metrics]) => (
                <div key={className} className="metric-card">
                  <h4 className={`metric-title type-${className}`}>
                    {className.toUpperCase()}
                  </h4>
                  <div className="metric-values">
                    <div className="metric-item">
                      <span className="metric-label">Precision</span>
                      <span className="metric-value">
                        {(metrics.precision * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Recall</span>
                      <span className="metric-value">
                        {(metrics.recall * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">F1-Score</span>
                      <span className="metric-value">
                        {(metrics.f1 * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;
