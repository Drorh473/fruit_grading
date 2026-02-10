/**
 * API for Processing Pipeline
 * Handles pipeline execution, status, and logs
 */

import { apiFetch } from "./apiClient";

/**
 * Start the processing pipeline
 * @param {Object} config - Pipeline configuration { skipTests, hiddenDim, epochs, learningRate, lambdaReg }
 * @returns {Promise<Object>} { pipelineId, status }
 */
export async function startPipeline(config = {}) {
  return apiFetch("/pipeline/start", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Stop the running pipeline
 * @returns {Promise<Object>} Status response
 */
export async function stopPipeline() {
  return apiFetch("/pipeline/stop", {
    method: "POST",
  });
}

/**
 * Get current pipeline status
 * @returns {Promise<Object>} { status, currentStep, progress, steps }
 */
export async function getPipelineStatus() {
  return apiFetch("/pipeline/status");
}

/**
 * Get pipeline logs
 * @param {number} limit - Number of log entries to fetch
 * @returns {Promise<Array>} Array of log entries
 */
export async function getPipelineLogs(limit = 100) {
  return apiFetch(`/pipeline/logs?limit=${limit}`);
}

/**
 * Get pipeline configuration
 * @returns {Promise<Object>} { hiddenDim, epochs, learningRate, batchSize, etc. }
 */
export async function getPipelineConfig() {
  return apiFetch("/pipeline/config");
}

/**
 * Update pipeline configuration
 * @param {Object} config - Configuration to update
 * @returns {Promise<Object>} Updated configuration
 */
export async function updatePipelineConfig(config) {
  return apiFetch("/pipeline/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

/**
 * Get pipeline history
 * @param {number} limit - Number of history entries to fetch
 * @returns {Promise<Array>} Array of past pipeline runs
 */
export async function getPipelineHistory(limit = 10) {
  return apiFetch(`/pipeline/history?limit=${limit}`);
}

export default {
  startPipeline,
  stopPipeline,
  getPipelineStatus,
  getPipelineLogs,
  getPipelineConfig,
  updatePipelineConfig,
  getPipelineHistory,
};
