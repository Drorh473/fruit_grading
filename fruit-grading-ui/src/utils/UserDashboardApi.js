/**
 * API for User Dashboard (Operator Role)
 * Handles dashboard statistics and recent results
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
