/**
 * API for Camera Monitor
 * Handles camera status, feeds, and configuration
 */

import { apiFetch } from "./apiClient";

/**
 * Get all camera statuses
 * @returns {Promise<Array>} Array of camera objects with status
 */
export async function getCameraStatuses() {
  return apiFetch("/cameras/status");
}

/**
 * Get specific camera details
 * @param {number} cameraId - Camera ID (0-3)
 * @returns {Promise<Object>} Camera details
 */
export async function getCameraDetails(cameraId) {
  return apiFetch(`/cameras/${cameraId}`);
}

/**
 * Refresh specific camera feed
 * @param {number} cameraId - Camera ID to refresh
 * @returns {Promise<Object>} Status response
 */
export async function refreshCamera(cameraId) {
  return apiFetch(`/cameras/${cameraId}/refresh`, {
    method: "POST",
  });
}

/**
 * Refresh all camera feeds
 * @returns {Promise<Object>} Status response
 */
export async function refreshAllCameras() {
  return apiFetch("/cameras/refresh-all", {
    method: "POST",
  });
}

/**
 * Get camera configuration
 * @returns {Promise<Object>} { fps, numCameras, imageSize, preprocessing }
 */
export async function getCameraConfig() {
  return apiFetch("/cameras/config");
}

export default {
  getCameraStatuses,
  getCameraDetails,
  refreshCamera,
  refreshAllCameras,
  getCameraConfig,
};
