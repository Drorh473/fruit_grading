// Camera monitor API endpoints

import { apiFetch } from "./apiClient";

export async function getCameraStatuses() {
  return apiFetch("/cameras/status");
}

export async function getCameraDetails(cameraId) {
  return apiFetch(`/cameras/${cameraId}`);
}

export async function refreshCamera(cameraId) {
  return apiFetch(`/cameras/${cameraId}/refresh`, {
    method: "POST",
  });
}

export async function refreshAllCameras() {
  return apiFetch("/cameras/refresh-all", {
    method: "POST",
  });
}

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
