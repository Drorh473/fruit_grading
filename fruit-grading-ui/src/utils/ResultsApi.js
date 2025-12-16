/**
 * API for Results Page
 * Handles results listing, filtering, KPIs, analytics, and export
 */

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000/api";

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.message || `HTTP error! status: ${response.status}`
      );
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      return await response.json();
    }

    return await response.text();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

/**
 * Get all results with filtering
 * @param {Object} filters - { search, type, batch, confidence, limit, offset }
 * @returns {Promise<Object>} { results: [...], total, limit, offset }
 */
export async function getResultsList(filters = {}) {
  const params = new URLSearchParams();

  if (filters.search) params.append("search", filters.search);
  if (filters.type && filters.type !== "all")
    params.append("type", filters.type);
  if (filters.batch && filters.batch !== "all")
    params.append("batch", filters.batch);
  if (filters.confidence && filters.confidence !== "all")
    params.append("confidence", filters.confidence);
  if (filters.limit) params.append("limit", filters.limit);
  if (filters.offset) params.append("offset", filters.offset);

  const queryString = params.toString();
  const endpoint = `/results/list${queryString ? `?${queryString}` : ""}`;

  return apiFetch(endpoint);
}

/**
 * Get KPI metrics
 * @returns {Promise<Object>} { totalProcessed, qualityRate, avgConfidence, processingSpeed, trends }
 */
export async function getKPIs() {
  return apiFetch("/results/kpis");
}

/**
 * Get quality distribution
 * @returns {Promise<Object>} { market, standard, premium } with counts and percentages
 */
export async function getQualityDistribution() {
  return apiFetch("/results/quality-distribution");
}

/**
 * Get quality alerts
 * @returns {Promise<Array>} Array of alerts with { id, type, title, message }
 */
export async function getQualityAlerts() {
  return apiFetch("/results/alerts");
}

/**
 * Get hourly processing trend
 * @param {number} hours - Number of hours to fetch (default 24)
 * @returns {Promise<Array>} Array of { hour, processed, qualityRate }
 */
export async function getHourlyTrend(hours = 24) {
  return apiFetch(`/results/hourly-trend?hours=${hours}`);
}

/**
 * Get batch performance comparison
 * @param {string} batchId - Specific batch to compare
 * @returns {Promise<Object>} { currentBatch, historicalAvg, metrics }
 */
export async function getBatchComparison(batchId) {
  const endpoint = batchId
    ? `/results/batch-comparison?batchId=${batchId}`
    : "/results/batch-comparison";
  return apiFetch(endpoint);
}

/**
 * Get list of unique batches
 * @returns {Promise<Array>} Array of batch IDs
 */
export async function getBatches() {
  return apiFetch("/results/batches");
}

/**
 * Get confusion matrix data
 * @returns {Promise<Object>} { classes: [...], matrix: [[...]], metrics: {...} }
 */
export async function getConfusionMatrix() {
  return apiFetch("/results/confusion-matrix");
}

/**
 * Export results as CSV
 * @param {Object} filters - { search, type, batch, confidence }
 * @returns {Promise<string>} CSV string
 */
export async function exportResultsCSV(filters = {}) {
  return apiFetch("/results/export/csv", {
    method: "POST",
    body: JSON.stringify({ filters }),
  });
}

/**
 * Export results as PDF report
 * @param {Object} options - Report options
 * @returns {Promise<Blob>} PDF blob
 */
export async function exportResultsPDF(options = {}) {
  const response = await fetch(`${API_BASE_URL}/results/export/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });

  if (!response.ok) {
    throw new Error(`PDF export failed: ${response.status}`);
  }

  return await response.blob();
}

/**
 * Export results as Excel with charts
 * @param {Object} options - Export options
 * @returns {Promise<Blob>} Excel blob
 */
export async function exportResultsExcel(options = {}) {
  const response = await fetch(`${API_BASE_URL}/results/export/excel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });

  if (!response.ok) {
    throw new Error(`Excel export failed: ${response.status}`);
  }

  return await response.blob();
}

/**
 * Schedule email report
 * @param {Object} config - { email, frequency, includeCharts }
 * @returns {Promise<Object>} Confirmation response
 */
export async function scheduleEmailReport(config) {
  return apiFetch("/results/schedule-report", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Download file blob
 * @param {Blob} blob - File blob
 * @param {string} filename - Filename for download
 */
export function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * Download CSV file
 * @param {string} csvContent - CSV string content
 * @param {string} filename - Filename for download
 */
export function downloadCSV(csvContent, filename = "results.csv") {
  const blob = new Blob([csvContent], { type: "text/csv" });
  downloadBlob(blob, filename);
}

/**
 * Check API health
 * @returns {Promise<Object>} { status, database }
 */
export async function checkHealth() {
  return apiFetch("/health");
}

// Export all functions as default object as well for flexibility
export default {
  getResultsList,
  getKPIs,
  getQualityDistribution,
  getQualityAlerts,
  getHourlyTrend,
  getBatchComparison,
  getBatches,
  getConfusionMatrix,
  exportResultsCSV,
  exportResultsPDF,
  exportResultsExcel,
  scheduleEmailReport,
  downloadBlob,
  downloadCSV,
  checkHealth,
};
