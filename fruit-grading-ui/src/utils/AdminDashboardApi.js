/**
 * API for Admin Dashboard
 * Handles system status, processing stats, and recent results
 */

import { apiFetch } from "./apiClient";

/**
 * Get system status (database, model, cameras)
 * @returns {Promise<Object>} { database, model, cameras: [] }
 */
export async function getSystemStatus() {
  return apiFetch("/admin/system-status");
}

/**
 * Get processing statistics
 * @returns {Promise<Object>} { totalProcessed, accuracy, marketCount, standardCount, premiumCount, rejectCount, totalImages, totalObjects }
 */
export async function getProcessingStats() {
  // FIXED: Changed from /admin/processing-stats to /admin/dashboard-stats
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
