/**
 * API for Settings
 * Handles system configuration management
 */

import { apiFetch } from "./apiClient";

/**
 * Get all system settings
 * @returns {Promise<Object>} Complete settings object
 */
export async function getSettings() {
  return apiFetch("/settings");
}

/**
 * Update system settings
 * @param {Object} settings - Settings to update
 * @returns {Promise<Object>} Updated settings
 */
export async function updateSettings(settings) {
  return apiFetch("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

/**
 * Update partial settings (only specific fields)
 * @param {Object} partialSettings - Specific settings to update
 * @returns {Promise<Object>} Updated settings
 */
export async function updatePartialSettings(partialSettings) {
  return apiFetch("/settings/partial", {
    method: "PATCH",
    body: JSON.stringify(partialSettings),
  });
}

/**
 * Reset settings to defaults
 * @returns {Promise<Object>} Default settings
 */
export async function resetSettings() {
  return apiFetch("/settings/reset", {
    method: "POST",
  });
}

/**
 * Test database connection
 * @param {string} connectionString - MongoDB connection string
 * @param {string} dbName - Database name
 * @returns {Promise<Object>} { success, message, details }
 */
export async function testDatabaseConnection(connectionString, dbName) {
  return apiFetch("/settings/test-database", {
    method: "POST",
    body: JSON.stringify({ connectionString, dbName }),
  });
}

/**
 * Test path accessibility
 * @param {string} path - Path to test
 * @param {string} type - Path type (e.g., 'stored', 'original', 'processed')
 * @returns {Promise<Object>} { success, exists, writable, readable, message }
 */
export async function testPath(path, type) {
  return apiFetch("/settings/test-path", {
    method: "POST",
    body: JSON.stringify({ path, type }),
  });
}

/**
 * Validate all paths at once
 * @param {Object} paths - { storedDataset, originalDataset, processedDataset }
 * @returns {Promise<Object>} Validation results for all paths
 */
export async function validateAllPaths(paths) {
  return apiFetch("/settings/validate-paths", {
    method: "POST",
    body: JSON.stringify(paths),
  });
}

/**
 * Get system status for settings page
 * @returns {Promise<Object>} { database, model, cameras, diskSpace, memory }
 */
export async function getSystemStatus() {
  return apiFetch("/settings/status");
}

/**
 * Get detailed system diagnostics
 * @returns {Promise<Object>} Comprehensive system information
 */
export async function getSystemDiagnostics() {
  return apiFetch("/settings/diagnostics");
}

/**
 * Restart system service
 * @param {string} service - Service to restart (e.g., 'api', 'cameras', 'model')
 * @returns {Promise<Object>} Status response
 */
export async function restartService(service) {
  return apiFetch(`/settings/restart/${service}`, {
    method: "POST",
  });
}

/**
 * Get available model variants
 * @returns {Promise<Array>} List of available ShuffleNet variants
 */
export async function getModelVariants() {
  return apiFetch("/settings/model-variants");
}

/**
 * Validate model configuration
 * @param {Object} modelConfig - { batchSize, modelVariant }
 * @returns {Promise<Object>} { valid, warnings, recommendations }
 */
export async function validateModelConfig(modelConfig) {
  return apiFetch("/settings/validate-model", {
    method: "POST",
    body: JSON.stringify(modelConfig),
  });
}

/**
 * Export settings configuration
 * @returns {Promise<Object>} Settings configuration object
 */
export async function exportSettings() {
  return apiFetch("/settings/export");
}

/**
 * Import settings configuration
 * @param {Object} settings - Settings to import
 * @returns {Promise<Object>} Import result
 */
export async function importSettings(settings) {
  return apiFetch("/settings/import", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

/**
 * Get settings change history
 * @param {number} limit - Number of history entries to fetch
 * @returns {Promise<Array>} Settings change history
 */
export async function getSettingsHistory(limit = 10) {
  return apiFetch(`/settings/history?limit=${limit}`);
}

/**
 * Check API health
 * @returns {Promise<Object>} { status, timestamp }
 */
export async function checkHealth() {
  return apiFetch("/health");
}

export default {
  getSettings,
  updateSettings,
  updatePartialSettings,
  resetSettings,
  testDatabaseConnection,
  testPath,
  validateAllPaths,
  getSystemStatus,
  getSystemDiagnostics,
  restartService,
  getModelVariants,
  validateModelConfig,
  exportSettings,
  importSettings,
  getSettingsHistory,
  checkHealth,
};
