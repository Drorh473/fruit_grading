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
        error.message || `HTTP error! status: ${response.status}`,
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
 * @param {Object} filters - { search, type, batch, limit, offset }
 * @returns {Promise<Object>} { results: [...], total, limit, offset }
 */
export async function getResultsList(filters = {}) {
  const params = new URLSearchParams();

  if (filters.search) params.append("search", filters.search);
  if (filters.type && filters.type !== "all")
    params.append("type", filters.type);
  if (filters.batch && filters.batch !== "all")
    params.append("batch", filters.batch);
  if (filters.limit) params.append("limit", filters.limit);
  if (filters.offset) params.append("offset", filters.offset);

  const queryString = params.toString();
  const endpoint = `/results/list${queryString ? `?${queryString}` : ""}`;

  return apiFetch(endpoint);
}

/**
 * Get KPI metrics
 * @returns {Promise<Object>} { totalProcessed, qualityRate, processingSpeed, trends }
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
 * Get training history from model metadata
 * @returns {Promise<Object>} { train_loss, train_accuracy, val_loss, val_accuracy }
 */
export async function getTrainingHistory() {
  return apiFetch("/results/training-history");
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
 * @returns {Promise<Object>} { classes, matrix, normalized, metrics }
 */
export async function getConfusionMatrix() {
  return apiFetch("/results/confusion-matrix");
}

/**
 * Export results as CSV
 * @returns {Promise<string>} CSV string
 */
export async function exportResultsCSV() {
  return apiFetch("/results/export");
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

export default {
  getResultsList,
  getKPIs,
  getQualityDistribution,
  getQualityAlerts,
  getTrainingHistory,
  getBatches,
  getConfusionMatrix,
  exportResultsCSV,
  downloadBlob,
  downloadCSV,
  checkHealth,
};
