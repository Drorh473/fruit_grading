/**
 * API for User Dashboard (Operator Role)
 * Handles dashboard statistics and recent results
 */

import { apiFetch } from "./apiClient";

/**
 * Get dashboard statistics for user role
 * @returns {Promise<Object>} { totalToday, marketCount, standardCount, rejectCount }
 */
export async function getUserDashboardStats() {
  return apiFetch("/user/dashboard-stats");
}

/**
 * Get recent classification results (last 10)
 * @returns {Promise<Array>} Array of { id, type, confidence, timestamp, images }
 */
export async function getRecentResults() {
  return apiFetch("/user/recent-results");
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
  getUserDashboardStats,
  getRecentResults,
  checkHealth,
};
