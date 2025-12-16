/**
 * API for Add Fruit
 * Handles fruit folder validation and processing
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
 * Validate folder structure
 * @param {string} folderPath - Path to fruit folder
 * @returns {Promise<Object>} { valid, message, details: { anglesFound, totalImages } }
 */
export async function validateFolder(folderPath) {
  return apiFetch("/fruit/validate", {
    method: "POST",
    body: JSON.stringify({ folderPath }),
  });
}

/**
 * Process new fruit folder (complete pipeline)
 * @param {string} folderPath - Path to fruit folder
 * @param {Object} options - { runTests: boolean }
 * @returns {Promise<Object>} { objectId, predictedType, confidence, imagesProcessed, processingTime }
 */
export async function processFruit(folderPath, options = {}) {
  return apiFetch("/fruit/process", {
    method: "POST",
    body: JSON.stringify({ folderPath, ...options }),
  });
}

/**
 * Get processing status for fruit
 * @param {string} objectId - Object ID being processed
 * @returns {Promise<Object>} { status, currentStep, progress, steps }
 */
export async function getFruitProcessingStatus(objectId) {
  return apiFetch(`/fruit/status/${objectId}`);
}

/**
 * Cancel fruit processing
 * @param {string} objectId - Object ID to cancel
 * @returns {Promise<Object>} Status response
 */
export async function cancelFruitProcessing(objectId) {
  return apiFetch(`/fruit/cancel/${objectId}`, {
    method: "POST",
  });
}

export default {
  validateFolder,
  processFruit,
  getFruitProcessingStatus,
  cancelFruitProcessing,
};
