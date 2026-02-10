/**
 * Shared API Client
 * Centralized API configuration and fetch wrapper
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000/api";

/**
 * Generic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint (e.g., "/admin/system-status")
 * @param {Object} options - Fetch options
 * @param {boolean} options.skipContentType - Skip setting Content-Type header (for FormData)
 * @returns {Promise<any>} Response data
 */
export async function apiFetch(endpoint, options = {}) {
  try {
    const { skipContentType, ...fetchOptions } = options;

    const headers = skipContentType
      ? { ...fetchOptions.headers }
      : {
          "Content-Type": "application/json",
          ...fetchOptions.headers,
        };

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers,
      ...fetchOptions,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(
        error.message || error.error || `HTTP error! status: ${response.status}`
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
 * Check API health
 * @returns {Promise<Object>} { status, database }
 */
export async function checkHealth() {
  return apiFetch("/health");
}

export default {
  API_BASE_URL,
  apiFetch,
  checkHealth,
};
