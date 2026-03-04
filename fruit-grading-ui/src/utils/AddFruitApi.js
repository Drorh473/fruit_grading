// Fruit dataset upload and processing API

import { apiFetch } from "./apiClient";

export async function validateFolder(folderPath) {
  return apiFetch("/fruit/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folderPath }),
  });
}

export async function uploadAndProcessFruit(files) {
  const formData = new FormData();

  Array.from(files).forEach((file) => {
    // Preserve relative path to reconstruct folder structure on the backend
    const path = file.webkitRelativePath || file.path || file.name;
    formData.append("dataset", file, path);
  });

  return apiFetch("/fruit/upload-process", {
    method: "POST",
    body: formData,
    skipContentType: true,
  });
}

export async function getFruitProcessingStatus(objectId) {
  return apiFetch(`/fruit/status/${objectId}`);
}

export async function cancelFruitProcessing(objectId) {
  return apiFetch(`/fruit/cancel/${objectId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

export default {
  validateFolder,
  uploadAndProcessFruit,
  getFruitProcessingStatus,
  cancelFruitProcessing,
};
