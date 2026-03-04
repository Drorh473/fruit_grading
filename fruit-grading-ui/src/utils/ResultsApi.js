// Results and analytics API endpoints

import { apiFetch } from "./apiClient";

export async function getTestPredictions(filters = {}) {
  const params = new URLSearchParams();

  if (filters.search) params.append("search", filters.search);
  if (filters.actual && filters.actual !== "all")
    params.append("actual", filters.actual);
  if (filters.predicted && filters.predicted !== "all")
    params.append("predicted", filters.predicted);
  if (filters.correct && filters.correct !== "all")
    params.append("correct", filters.correct);

  const queryString = params.toString();
  const endpoint = `/results/test-predictions${queryString ? `?${queryString}` : ""}`;

  return apiFetch(endpoint);
}

export async function getResultsList(filters = {}) {
  const params = new URLSearchParams();

  if (filters.search) params.append("search", filters.search);
  if (filters.type && filters.type !== "all")
    params.append("type", filters.type);
  if (filters.batch && filters.batch !== "all")
    params.append("batch", filters.batch);
  if (filters.limit) params.append("limit", filters.limit);
  if (filters.offset) params.append("offset", filters.offset);

  const queryString = params.toString();
  const endpoint = `/results/list${queryString ? `?${queryString}` : ""}`;

  return apiFetch(endpoint);
}

export async function getKPIs() {
  return apiFetch("/results/kpis");
}

export async function getQualityDistribution() {
  return apiFetch("/results/quality-distribution");
}

export async function getQualityAlerts() {
  return apiFetch("/results/alerts");
}

export async function getTrainingHistory() {
  return apiFetch("/results/training-history");
}

export async function getBatches() {
  return apiFetch("/results/batches");
}

export async function getConfusionMatrix() {
  return apiFetch("/results/confusion-matrix");
}

export async function exportResultsCSV() {
  return apiFetch("/results/export");
}

export function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export function downloadCSV(csvContent, filename = "results.csv") {
  const blob = new Blob([csvContent], { type: "text/csv" });
  downloadBlob(blob, filename);
}

export async function checkHealth() {
  return apiFetch("/health");
}

export default {
  getTestPredictions,
  getResultsList,
  getKPIs,
  getQualityDistribution,
  getQualityAlerts,
  getTrainingHistory,
  getBatches,
  getConfusionMatrix,
  exportResultsCSV,
  downloadBlob,
  downloadCSV,
  checkHealth,
};
