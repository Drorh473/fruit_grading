import React, { useState, useEffect } from "react";
import {
  FiDownload,
  FiFilter,
  FiSearch,
  FiTrendingUp,
  FiTrendingDown,
  FiAlertCircle,
  FiCheckCircle,
} from "react-icons/fi";
import "./Results.css";
import {
  getResultsList,
  getKPIs,
  getQualityAlerts,
  exportResultsCSV,
  downloadCSV,
  getConfusionMatrix,
  getTrainingHistory,
} from "../utils/ResultsApi";

const Results = () => {
  // Data states
  const [results, setResults] = useState([]);
  const [kpis, setKpis] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);

  // Filter states
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");

  // UI states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    fetchAllData();
  }, []);

  useEffect(() => {
    fetchResults();
  }, [searchTerm, filterType]);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      const [kpisData, alertsData, confusionData, historyData] =
        await Promise.all([
          getKPIs().catch((err) => {
            console.error("KPIs fetch failed:", err);
            return null;
          }),
          getQualityAlerts().catch((err) => {
            console.error("Alerts fetch failed:", err);
            return [];
          }),
          getConfusionMatrix().catch((err) => {
            console.error("Confusion matrix fetch failed:", err);
            return null;
          }),
          getTrainingHistory().catch((err) => {
            console.error("Training history fetch failed:", err);
            return null;
          }),
        ]);

      setKpis(kpisData);
      setAlerts(alertsData);
      setConfusionMatrix(confusionData);
      setTrainingHistory(historyData);

      await fetchResults();
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to load results data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const fetchResults = async () => {
    try {
      const filters = {
        search: searchTerm || undefined,
        type: filterType,
        limit: 100,
      };

      const data = await getResultsList(filters);
      setResults(data.results || []);
    } catch (err) {
      console.error("Error fetching results:", err);
      if (!results.length) {
        setError("Failed to load results. Please try again.");
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

  // Loss Chart Component
  const LossChart = ({ data }) => {
    if (!data || !data.train_loss || data.train_loss.length === 0) {
      return (
        <div className="chart-placeholder">
          <p>No training history available</p>
        </div>
      );
    }

    const epochs = data.train_loss.length;
    const maxLoss = Math.max(...data.train_loss, ...(data.val_loss || []));
    const chartWidth = 500;
    const chartHeight = 250;
    const padding = { top: 20, right: 30, bottom: 50, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;

    const xScale = (i) =>
      padding.left + (i / Math.max(epochs - 1, 1)) * innerWidth;
    const yScale = (v) => padding.top + (1 - v / maxLoss) * innerHeight;

    const createPath = (values) => {
      if (!values || values.length === 0) return "";
      return values
        .map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(v)}`)
        .join(" ");
    };

    const getEpochTicks = () => {
      if (epochs <= 10) return [...Array(epochs).keys()];
      const step = Math.ceil(epochs / 5);
      const ticks = [];
      for (let i = 0; i < epochs; i += step) {
        ticks.push(i);
      }
      if (ticks[ticks.length - 1] !== epochs - 1) {
        ticks.push(epochs - 1);
      }
      return ticks;
    };

    return (
      <div className="single-chart-container">
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="line-chart"
        >
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={padding.left}
                y1={yScale(tick * maxLoss)}
                x2={chartWidth - padding.right}
                y2={yScale(tick * maxLoss)}
                stroke="var(--border)"
                strokeDasharray="2,2"
              />
              <text
                x={padding.left - 8}
                y={yScale(tick * maxLoss)}
                className="axis-tick"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {(tick * maxLoss).toFixed(2)}
              </text>
            </g>
          ))}
          {getEpochTicks().map((epochIdx) => (
            <text
              key={epochIdx}
              x={xScale(epochIdx)}
              y={chartHeight - padding.bottom + 15}
              className="axis-tick"
              textAnchor="middle"
            >
              {epochIdx + 1}
            </text>
          ))}
          <path
            d={createPath(data.train_loss)}
            fill="none"
            stroke="var(--accent-primary)"
            strokeWidth="2.5"
          />
          {data.val_loss && (
            <path
              d={createPath(data.val_loss)}
              fill="none"
              stroke="var(--warning)"
              strokeWidth="2.5"
            />
          )}
          <text x={chartWidth / 2} y={chartHeight - 5} className="axis-label">
            Epoch
          </text>
        </svg>
        <div className="chart-legend">
          <span className="legend-item">
            <span
              className="legend-dot"
              style={{ background: "var(--accent-primary)" }}
            ></span>
            Train Loss
          </span>
          {data.val_loss && (
            <span className="legend-item">
              <span
                className="legend-dot"
                style={{ background: "var(--warning)" }}
              ></span>
              Val Loss
            </span>
          )}
        </div>
      </div>
    );
  };

  // Accuracy Chart Component
  const AccuracyChart = ({ data }) => {
    if (!data || !data.train_accuracy || data.train_accuracy.length === 0) {
      return (
        <div className="chart-placeholder">
          <p>No training history available</p>
        </div>
      );
    }

    const epochs = data.train_accuracy.length;
    const chartWidth = 500;
    const chartHeight = 250;
    const padding = { top: 20, right: 30, bottom: 50, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;

    const xScale = (i) =>
      padding.left + (i / Math.max(epochs - 1, 1)) * innerWidth;
    const yScale = (v) => padding.top + (1 - v) * innerHeight;

    const createPath = (values) => {
      if (!values || values.length === 0) return "";
      return values
        .map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(v)}`)
        .join(" ");
    };

    const getEpochTicks = () => {
      if (epochs <= 10) return [...Array(epochs).keys()];
      const step = Math.ceil(epochs / 5);
      const ticks = [];
      for (let i = 0; i < epochs; i += step) {
        ticks.push(i);
      }
      if (ticks[ticks.length - 1] !== epochs - 1) {
        ticks.push(epochs - 1);
      }
      return ticks;
    };

    return (
      <div className="single-chart-container">
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="line-chart"
        >
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={padding.left}
                y1={yScale(tick)}
                x2={chartWidth - padding.right}
                y2={yScale(tick)}
                stroke="var(--border)"
                strokeDasharray="2,2"
              />
              <text
                x={padding.left - 8}
                y={yScale(tick)}
                className="axis-tick"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          {getEpochTicks().map((epochIdx) => (
            <text
              key={epochIdx}
              x={xScale(epochIdx)}
              y={chartHeight - padding.bottom + 15}
              className="axis-tick"
              textAnchor="middle"
            >
              {epochIdx + 1}
            </text>
          ))}
          <path
            d={createPath(data.train_accuracy)}
            fill="none"
            stroke="var(--success)"
            strokeWidth="2.5"
          />
          {data.val_accuracy && (
            <path
              d={createPath(data.val_accuracy)}
              fill="none"
              stroke="var(--info)"
              strokeWidth="2.5"
            />
          )}
          <text x={chartWidth / 2} y={chartHeight - 8} className="axis-label">
            Epoch
          </text>
        </svg>
        <div className="chart-legend">
          <span className="legend-item">
            <span
              className="legend-dot"
              style={{ background: "var(--success)" }}
            ></span>
            Train Acc
          </span>
          {data.val_accuracy && (
            <span className="legend-item">
              <span
                className="legend-dot"
                style={{ background: "var(--info)" }}
              ></span>
              Val Acc
            </span>
          )}
        </div>
      </div>
    );
  };

  // Normalized Confusion Matrix Component
  const ConfusionMatrixChart = ({ data }) => {
    if (!data || !data.matrix || !data.classes) {
      return (
        <div className="chart-placeholder">
          <p>No confusion matrix available</p>
        </div>
      );
    }

    // Filter out reject class
    const rejectIndex = data.classes.findIndex(
      (c) => c.toLowerCase() === "reject",
    );
    let classes = data.classes;
    let matrix = data.matrix;
    let normalized = data.normalized;

    if (rejectIndex !== -1) {
      classes = data.classes.filter((_, i) => i !== rejectIndex);
      matrix = data.matrix
        .filter((_, i) => i !== rejectIndex)
        .map((row) => row.filter((_, j) => j !== rejectIndex));
      if (data.normalized) {
        normalized = data.normalized
          .filter((_, i) => i !== rejectIndex)
          .map((row) => row.filter((_, j) => j !== rejectIndex));
      }
    }

    const displayMatrix = normalized || matrix;
    const isNormalized = !!normalized;

    const getColor = (value) => {
      const intensity = isNormalized
        ? value
        : value / Math.max(...matrix.flat());
      const r = Math.round(20 + (1 - intensity) * 30);
      const g = Math.round(80 + intensity * 100);
      const b = Math.round(120 + intensity * 80);
      return `rgb(${r}, ${g}, ${b})`;
    };

    return (
      <div className="confusion-matrix-container">
        <div className="matrix-grid">
          <div className="matrix-corner"></div>
          <div className="matrix-header-label">Predicted</div>
          <div className="matrix-side-label">Actual</div>
          <div className="matrix-content">
            {/* Header row */}
            <div className="matrix-row header-row">
              <div className="matrix-cell empty"></div>
              {classes.map((cls) => (
                <div key={cls} className="matrix-cell header">
                  {cls}
                </div>
              ))}
            </div>
            {/* Data rows */}
            {displayMatrix.map((row, i) => (
              <div key={i} className="matrix-row">
                <div className="matrix-cell row-header">{classes[i]}</div>
                {row.map((value, j) => (
                  <div
                    key={j}
                    className={`matrix-cell data ${i === j ? "diagonal" : ""}`}
                    style={{ backgroundColor: getColor(value) }}
                  >
                    {isNormalized ? `${(value * 100).toFixed(0)}%` : value}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        {data.metrics && (
          <div className="matrix-metrics">
            <div className="metric-item">
              <span className="metric-label">Accuracy</span>
              <span className="metric-value">
                {(data.metrics.accuracy * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}
      </div>
    );
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
          <p className="page-subtitle">
            Operational metrics and model performance
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      {kpis && (
        <div className="kpi-grid">
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
              <span className="kpi-label">Model Accuracy</span>
            </div>
            <div className="kpi-value">
              {confusionMatrix?.metrics?.accuracy
                ? `${(confusionMatrix.metrics.accuracy * 100).toFixed(1)}%`
                : "N/A"}
            </div>
          </div>
        </div>
      )}

      {/* Quality Distribution and Alerts */}
      <div className="charts-alerts-grid">
        <div className="charts-column">
          {confusionMatrix &&
            confusionMatrix.matrix &&
            confusionMatrix.matrix.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">Predicted Grade Distribution</h2>
                  <p className="card-subtitle">Model classification results</p>
                </div>
                <div className="pie-chart-container">
                  <svg viewBox="0 0 220 220" className="pie-chart">
                    {(() => {
                      const cx = 110;
                      const cy = 110;
                      const radius = 80;
                      const innerRadius = 50;

                      // Calculate predicted counts from confusion matrix columns
                      const classes = confusionMatrix.classes || [];
                      const matrix = confusionMatrix.matrix || [];

                      const predictedCounts = {};
                      classes.forEach((cls, colIdx) => {
                        let count = 0;
                        matrix.forEach((row) => {
                          count += row[colIdx] || 0;
                        });
                        predictedCounts[cls] = count;
                      });

                      const total = Object.values(predictedCounts).reduce(
                        (a, b) => a + b,
                        0,
                      );

                      const marketCount = predictedCounts["market"] || 0;
                      const standardCount = predictedCounts["standard"] || 0;
                      const premiumCount = predictedCounts["premium"] || 0;

                      const marketPct =
                        total > 0 ? (marketCount / total) * 100 : 0;
                      const standardPct =
                        total > 0 ? (standardCount / total) * 100 : 0;
                      const premiumPct =
                        total > 0 ? (premiumCount / total) * 100 : 0;

                      // Handle 100% case with full circle
                      if (marketPct === 100) {
                        return (
                          <>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={radius}
                              fill="var(--success)"
                            />
                            <circle
                              cx={cx}
                              cy={cy}
                              r={innerRadius}
                              fill="var(--bg-medium)"
                            />
                          </>
                        );
                      }
                      if (standardPct === 100) {
                        return (
                          <>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={radius}
                              fill="var(--info)"
                            />
                            <circle
                              cx={cx}
                              cy={cy}
                              r={innerRadius}
                              fill="var(--bg-medium)"
                            />
                          </>
                        );
                      }
                      if (premiumPct === 100) {
                        return (
                          <>
                            <circle cx={cx} cy={cy} r={radius} fill="#9b59b6" />
                            <circle
                              cx={cx}
                              cy={cy}
                              r={innerRadius}
                              fill="var(--bg-medium)"
                            />
                          </>
                        );
                      }

                      const createArcPath = (startAngle, endAngle, color) => {
                        if (endAngle - startAngle <= 0) return null;

                        const startRad = (startAngle - 90) * (Math.PI / 180);
                        const endRad = (endAngle - 90) * (Math.PI / 180);

                        const x1 = cx + radius * Math.cos(startRad);
                        const y1 = cy + radius * Math.sin(startRad);
                        const x2 = cx + radius * Math.cos(endRad);
                        const y2 = cy + radius * Math.sin(endRad);
                        const x3 = cx + innerRadius * Math.cos(endRad);
                        const y3 = cy + innerRadius * Math.sin(endRad);
                        const x4 = cx + innerRadius * Math.cos(startRad);
                        const y4 = cy + innerRadius * Math.sin(startRad);

                        const largeArc = endAngle - startAngle > 180 ? 1 : 0;

                        const d = `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x4} ${y4} Z`;

                        return <path key={color} d={d} fill={color} />;
                      };

                      let currentAngle = 0;
                      const segments = [];

                      if (marketPct > 0) {
                        const endAngle = currentAngle + (marketPct / 100) * 360;
                        segments.push(
                          createArcPath(
                            currentAngle,
                            endAngle,
                            "var(--success)",
                          ),
                        );
                        currentAngle = endAngle;
                      }
                      if (standardPct > 0) {
                        const endAngle =
                          currentAngle + (standardPct / 100) * 360;
                        segments.push(
                          createArcPath(currentAngle, endAngle, "var(--info)"),
                        );
                        currentAngle = endAngle;
                      }
                      if (premiumPct > 0) {
                        const endAngle =
                          currentAngle + (premiumPct / 100) * 360;
                        segments.push(
                          createArcPath(currentAngle, endAngle, "#9b59b6"),
                        );
                        currentAngle = endAngle;
                      }

                      return segments;
                    })()}
                  </svg>

                  <div className="pie-chart-legend">
                    {(() => {
                      const classes = confusionMatrix.classes || [];
                      const matrix = confusionMatrix.matrix || [];

                      const predictedCounts = {};
                      classes.forEach((cls, colIdx) => {
                        let count = 0;
                        matrix.forEach((row) => {
                          count += row[colIdx] || 0;
                        });
                        predictedCounts[cls] = count;
                      });

                      const total = Object.values(predictedCounts).reduce(
                        (a, b) => a + b,
                        0,
                      );

                      const marketCount = predictedCounts["market"] || 0;
                      const standardCount = predictedCounts["standard"] || 0;
                      const premiumCount = predictedCounts["premium"] || 0;

                      const marketPct =
                        total > 0
                          ? ((marketCount / total) * 100).toFixed(1)
                          : 0;
                      const standardPct =
                        total > 0
                          ? ((standardCount / total) * 100).toFixed(1)
                          : 0;
                      const premiumPct =
                        total > 0
                          ? ((premiumCount / total) * 100).toFixed(1)
                          : 0;

                      return (
                        <>
                          <div className="legend-item">
                            <div
                              className="legend-color"
                              style={{ background: "var(--success)" }}
                            ></div>
                            <div className="legend-info">
                              <span className="legend-label">Market Grade</span>
                              <span className="legend-value">
                                {marketCount} ({marketPct}%)
                              </span>
                            </div>
                          </div>
                          <div className="legend-item">
                            <div
                              className="legend-color"
                              style={{ background: "var(--info)" }}
                            ></div>
                            <div className="legend-info">
                              <span className="legend-label">
                                Standard Grade
                              </span>
                              <span className="legend-value">
                                {standardCount} ({standardPct}%)
                              </span>
                            </div>
                          </div>
                          <div className="legend-item">
                            <div
                              className="legend-color"
                              style={{ background: "#9b59b6" }}
                            ></div>
                            <div className="legend-info">
                              <span className="legend-label">
                                Premium Grade
                              </span>
                              <span className="legend-value">
                                {premiumCount} ({premiumPct}%)
                              </span>
                            </div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            )}
        </div>

        <div className="alerts-sidebar">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">System Status</h2>
              <p className="card-subtitle">Current alerts</p>
            </div>
            <div className="alert-list">
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`alert-item alert-${alert.type}`}
                  >
                    <div className="alert-icon">
                      {alert.type === "success" ? (
                        <FiCheckCircle />
                      ) : (
                        <FiAlertCircle />
                      )}
                    </div>
                    <div className="alert-content">
                      <div className="alert-title">{alert.title}</div>
                      <div className="alert-message">{alert.message}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <FiCheckCircle size={24} color="var(--success)" />
                  <p>All systems operational</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Training History Charts - Side by Side */}
      <div className="training-charts-grid">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Loss Over Epochs</h2>
            <p className="card-subtitle">Training and validation loss</p>
          </div>
          <LossChart data={trainingHistory} />
        </div>
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Accuracy Over Epochs</h2>
            <p className="card-subtitle">Training and validation accuracy</p>
          </div>
          <AccuracyChart data={trainingHistory} />
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Normalized Confusion Matrix</h2>
          <p className="card-subtitle">Classification performance by grade</p>
        </div>
        <ConfusionMatrixChart data={confusionMatrix} />
      </div>

      {/* Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            Classification Results ({results.length} items)
          </h2>
        </div>

        <div className="filters-container">
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
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Grades</option>
              <option value="market">Market</option>
              <option value="standard">Standard</option>
              <option value="premium">Premium</option>
            </select>
          </div>
        </div>

        {results.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Grade</th>
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

      {/* Export */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Export Data</h2>
        </div>
        <div className="export-options">
          <button
            className="btn btn-secondary"
            onClick={handleExportCSV}
            disabled={exporting}
          >
            <FiDownload />
            {exporting ? "Exporting..." : "Export CSV"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;
