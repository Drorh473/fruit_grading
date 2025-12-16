/**
 * API for Admin Dashboard
 * Handles system status, processing stats, and recent results
 */

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000/api";

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
      throw new Error(error.error || `HTTP error! status: ${response.status}`);
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
 * Get system status (database, model, cameras)
 * @returns {Promise<Object>} { database, model, cameras: [] }
 */
export async function getSystemStatus() {
  return apiFetch("/admin/system-status");
}

/**
 * Get processing statistics
 * @returns {Promise<Object>} { totalProcessed, accuracy, lastUpdate }
 */
export async function getProcessingStats() {
  return apiFetch("/admin/processing-stats");
}

/**
 * Get recent classification results (last N)
 * @param {number} limit - Number of results to fetch (default: 10)
 * @returns {Promise<Array>} Array of results
 */
export async function getRecentResults(limit = 10) {
  return apiFetch(`/admin/recent-results?limit=${limit}`);
}

/**
 * Get dataset information
 * @returns {Promise<Object>} { trainingCount, testingCount, totalImages, featureDim }
 */
export async function getDatasetInfo() {
  return apiFetch("/admin/dataset-info");
}

/**
 * Get model performance metrics
 * @returns {Promise<Object>} { trainAccuracy, testAccuracy, architecture, classes }
 */
export async function getModelPerformance() {
  return apiFetch("/admin/model-performance");
}

export default {
  getSystemStatus,
  getProcessingStats,
  getRecentResults,
  getDatasetInfo,
  getModelPerformance,
};
