/**
 * API for Add Fruit
 * Handles fruit folder upload and processing
 */

import { apiFetch } from "./apiClient";

/**
 * Validate folder structure (Now strictly Client-Side helper, or optional server check)
 * Since we are uploading, we usually validate in the browser first.
 */
export async function validateFolder(folderPath) {
  // Legacy support if needed, otherwise unused in new upload flow
  return apiFetch("/fruit/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folderPath }),
  });
}

/**
 * Upload and Process fruit dataset
 * Sends actual files to the backend
 * @param {Array<File>} files - List of file objects
 * @returns {Promise<Object>} Result data
 */
export async function uploadAndProcessFruit(files) {
  const formData = new FormData();

  // Append all files to the form data
  Array.from(files).forEach((file) => {
    // We send the relative path so backend can reconstruct folders (angle_0, etc.)
    // For input[type=file], use webkitRelativePath.
    // For DnD, we ensure the 'path' property is set on the file object manually.
    const path = file.webkitRelativePath || file.path || file.name;
    formData.append("dataset", file, path);
  });

  return apiFetch("/fruit/upload-process", {
    method: "POST",
    body: formData,
    skipContentType: true, // Let browser set Content-Type for FormData
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
