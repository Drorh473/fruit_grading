// System settings API endpoints

import { apiFetch } from "./apiClient";

export async function getSettings() {
  return apiFetch("/settings");
}

export async function updateSettings(settings) {
  return apiFetch("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export async function updatePartialSettings(partialSettings) {
  return apiFetch("/settings/partial", {
    method: "PATCH",
    body: JSON.stringify(partialSettings),
  });
}

export async function resetSettings() {
  return apiFetch("/settings/reset", {
    method: "POST",
  });
}

export async function testDatabaseConnection(connectionString, dbName) {
  return apiFetch("/settings/test-database", {
    method: "POST",
    body: JSON.stringify({ connectionString, dbName }),
  });
}

export async function testPath(path, type) {
  return apiFetch("/settings/test-path", {
    method: "POST",
    body: JSON.stringify({ path, type }),
  });
}

export async function validateAllPaths(paths) {
  return apiFetch("/settings/validate-paths", {
    method: "POST",
    body: JSON.stringify(paths),
  });
}

export async function getSystemStatus() {
  return apiFetch("/settings/status");
}

export async function getSystemDiagnostics() {
  return apiFetch("/settings/diagnostics");
}

export async function restartService(service) {
  return apiFetch(`/settings/restart/${service}`, {
    method: "POST",
  });
}

export async function getModelVariants() {
  return apiFetch("/settings/model-variants");
}

export async function validateModelConfig(modelConfig) {
  return apiFetch("/settings/validate-model", {
    method: "POST",
    body: JSON.stringify(modelConfig),
  });
}

export async function exportSettings() {
  return apiFetch("/settings/export");
}

export async function importSettings(settings) {
  return apiFetch("/settings/import", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

export async function getSettingsHistory(limit = 10) {
  return apiFetch(`/settings/history?limit=${limit}`);
}

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
