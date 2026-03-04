// Processing pipeline API endpoints

import { apiFetch } from "./apiClient";

export async function startPipeline(config = {}) {
  return apiFetch("/pipeline/start", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function stopPipeline() {
  return apiFetch("/pipeline/stop", {
    method: "POST",
  });
}

export async function getPipelineStatus() {
  return apiFetch("/pipeline/status");
}

export async function getPipelineLogs(limit = 100) {
  return apiFetch(`/pipeline/logs?limit=${limit}`);
}

export async function getPipelineConfig() {
  return apiFetch("/pipeline/config");
}

export async function updatePipelineConfig(config) {
  return apiFetch("/pipeline/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

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
