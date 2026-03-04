// Admin dashboard API endpoints

import { apiFetch } from "./apiClient";

export async function getSystemStatus() {
  return apiFetch("/admin/system-status");
}

export async function getProcessingStats() {
  return apiFetch("/admin/processing-stats");
}

export async function getRecentResults(limit = 10) {
  return apiFetch(`/admin/recent-results?limit=${limit}`);
}

export async function getDatasetInfo() {
  return apiFetch("/admin/dataset-info");
}

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
