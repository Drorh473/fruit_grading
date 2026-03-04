// User dashboard API endpoints

import { apiFetch } from "./apiClient";

export async function getUserDashboardStats() {
  return apiFetch("/user/dashboard-stats");
}

export async function getRecentResults() {
  return apiFetch("/user/recent-results");
}

export async function checkHealth() {
  return apiFetch("/health");
}

export default {
  getUserDashboardStats,
  getRecentResults,
  checkHealth,
};
