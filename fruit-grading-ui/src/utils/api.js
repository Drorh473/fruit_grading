/**
 * API Utility for Fruit Grading System
 * Handles all communication between React frontend and Flask backend
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

// ============================================================================
// USER/OPERATOR ENDPOINTS
// ============================================================================

/**
 * Get dashboard statistics for user role
 * @returns {Promise<Object>} { totalToday, marketCount, standardCount, rejectCount }
 */
export async function getUserDashboardStats() {
  return apiFetch("/user/dashboard-stats");
}

/**
 * Get recent classification results (last 10)
 * @returns {Promise<Array>} Array of { id, type, timestamp, images }
 */
export async function getRecentResults() {
  return apiFetch("/user/recent-results");
}

/**
 * Get all results with filtering
 * @param {Object} filters - { search, type, limit, offset }
 * @returns {Promise<Object>} { results: [...], total, limit, offset }
 */
export async function getResultsList(filters = {}) {
  const params = new URLSearchParams();

  if (filters.search) params.append("search", filters.search);
  if (filters.type) params.append("type", filters.type);
  if (filters.limit) params.append("limit", filters.limit);
  if (filters.offset) params.append("offset", filters.offset);

  const queryString = params.toString();
  const endpoint = `/results/list${queryString ? `?${queryString}` : ""}`;

  return apiFetch(endpoint);
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
 * @param {Object} filters - { search, type }
 * @returns {Promise<string>} CSV string
 */
export async function exportResults(filters = {}) {
  return apiFetch("/results/export", {
    method: "POST",
    body: JSON.stringify({ filters }),
  });
}

/**
 * Check API health
 * @returns {Promise<Object>} { status, database }
 */
export async function checkHealth() {
  return apiFetch("/health");
}

/**
 * Download CSV file
 * @param {string} csvContent - CSV string content
 * @param {string} filename - Filename for download
 */
export function downloadCSV(csvContent, filename = "results.csv") {
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

// Export all functions as default object as well for flexibility
export default {
  getUserDashboardStats,
  getRecentResults,
  getResultsList,
  getConfusionMatrix,
  exportResults,
  checkHealth,
  downloadCSV,
};
