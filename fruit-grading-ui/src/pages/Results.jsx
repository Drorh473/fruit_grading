import React, { useState, useEffect } from 'react';
import { FiDownload, FiFilter, FiSearch } from 'react-icons/fi';
import './Results.css';

const Results = () => {
  const [results, setResults] = useState([]);
  const [filteredResults, setFilteredResults] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [confusionMatrix, setConfusionMatrix] = useState(null);

  useEffect(() => {
    // Fetch results from API
    fetchResults();
    fetchConfusionMatrix();
  }, []);

  useEffect(() => {
    filterResults();
  }, [searchTerm, filterType, results]);

  const fetchResults = async () => {
    // Mock data - replace with actual API call
    const mockResults = [
      { id: 'obj0001', type: 'market', confidence: 0.95, timestamp: '2025-12-12 14:30:22', images: 12 },
      { id: 'obj0002', type: 'standard', confidence: 0.88, timestamp: '2025-12-12 14:28:15', images: 12 },
      { id: 'obj0003', type: 'reject', confidence: 0.92, timestamp: '2025-12-12 14:25:08', images: 12 },
      { id: 'obj0004', type: 'market', confidence: 0.89, timestamp: '2025-12-12 14:20:45', images: 12 },
      { id: 'obj0005', type: 'standard', confidence: 0.91, timestamp: '2025-12-12 14:18:33', images: 12 },
    ];
    setResults(mockResults);
  };

  const fetchConfusionMatrix = async () => {
    // Mock confusion matrix data
    setConfusionMatrix({
      classes: ['market', 'standard', 'reject'],
      matrix: [
        [8, 1, 0],  // market
        [2, 6, 1],  // standard
        [0, 2, 8]   // reject
      ],
      metrics: {
        market: { precision: 0.80, recall: 0.89, f1: 0.84 },
        standard: { precision: 0.67, recall: 0.67, f1: 0.67 },
        reject: { precision: 0.89, recall: 0.80, f1: 0.84 }
      }
    });
  };

  const filterResults = () => {
    let filtered = results;

    if (searchTerm) {
      filtered = filtered.filter(r => 
        r.id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType !== 'all') {
      filtered = filtered.filter(r => r.type === filterType);
    }

    setFilteredResults(filtered);
  };

  const exportResults = () => {
    const csv = [
      ['Object ID', 'Type', 'Confidence', 'Timestamp', 'Images'],
      ...filteredResults.map(r => [r.id, r.type, r.confidence, r.timestamp, r.images])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.csv';
    a.click();
  };

  return (
    <div className="results">
      <div className="page-header">
        <div>
          <h1>Classification Results</h1>
          <p className="page-subtitle">View and analyze fruit grading results</p>
        </div>
        <button className="btn btn-primary" onClick={exportResults}>
          <FiDownload />
          Export Results
        </button>
      </div>

      {/* Filters */}
      <div className="card">
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
              <option value="all">All Types</option>
              <option value="market">Market</option>
              <option value="standard">Standard</option>
              <option value="reject">Reject</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Results ({filteredResults.length})</h2>
        </div>
        {filteredResults.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Object ID</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Images</th>
                  <th>Timestamp</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((result) => (
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
                        <span className="confidence-text">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td>{result.images}</td>
                    <td className="timestamp">{result.timestamp}</td>
                    <td>
                      <button className="btn-link">View Details</button>
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

      {/* Confusion Matrix */}
      {confusionMatrix && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Confusion Matrix</h2>
            <span className="card-subtitle">Model classification performance</span>
          </div>
          <div className="confusion-matrix-container">
            <div className="matrix-grid">
              <div className="matrix-header"></div>
              {confusionMatrix.classes.map(cls => (
                <div key={cls} className="matrix-header">
                  <span className={`type-badge type-${cls}`}>{cls}</span>
                </div>
              ))}
              {confusionMatrix.matrix.map((row, i) => (
                <React.Fragment key={i}>
                  <div className="matrix-row-header">
                    <span className={`type-badge type-${confusionMatrix.classes[i]}`}>
                      {confusionMatrix.classes[i]}
                    </span>
                  </div>
                  {row.map((value, j) => (
                    <div 
                      key={j} 
                      className={`matrix-cell ${i === j ? 'diagonal' : ''}`}
                      style={{
                        background: i === j 
                          ? `rgba(39, 174, 96, ${value / 10})` 
                          : `rgba(231, 76, 60, ${value / 10})`
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
            {Object.entries(confusionMatrix.metrics).map(([className, metrics]) => (
              <div key={className} className="metric-card">
                <h4 className={`metric-title type-${className}`}>
                  {className.toUpperCase()}
                </h4>
                <div className="metric-values">
                  <div className="metric-item">
                    <span className="metric-label">Precision</span>
                    <span className="metric-value">{(metrics.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Recall</span>
                    <span className="metric-value">{(metrics.recall * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">F1-Score</span>
                    <span className="metric-value">{(metrics.f1 * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;
