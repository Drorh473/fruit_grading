// Centralized API fetch wrapper with error handling

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000/api";

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

export async function checkHealth() {
  return apiFetch("/health");
}

export default {
  API_BASE_URL,
  apiFetch,
  checkHealth,
};
