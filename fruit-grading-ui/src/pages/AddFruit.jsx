import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiUpload,
  FiFolder,
  FiCheckCircle,
  FiAlertCircle,
  FiArrowRight,
  FiPlus,
} from "react-icons/fi";
import { uploadAndProcessFruit } from "../utils/AddFruitApi";
import PageHeader from "../components/PageHeader";
import "./AddFruit.css";

const AddFruit = () => {
  const navigate = useNavigate();
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [folderName, setFolderName] = useState("");
  const [validation, setValidation] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  // Client-Side Validation Helper
  const validateFilesClientSide = (files) => {
    const requiredAngles = ["an_0", "an_1", "an_2", "an_3"];
    // Track count of images per angle folder
    const counts = Object.fromEntries(requiredAngles.map((a) => [a, 0]));

    let totalImages = 0;
    const fileList = Array.from(files);

    fileList.forEach((file) => {
      // Path format from input: "RootFolder/SubFolder/Image.jpg"
      const path = file.webkitRelativePath || file.path || "";
      const parts = path.split("/");
      const isImage = file.type && file.type.startsWith("image/");

      // We expect at least: RootName/AngleName/File.jpg
      if (parts.length >= 2 && isImage) {
        totalImages++;

        // Check if any part of the path matches our required angle folders
        requiredAngles.forEach((angle) => {
          if (parts.includes(angle)) {
            counts[angle]++;
          }
        });
      }
    });

    const missing = requiredAngles.filter((a) => counts[a] === 0);
    const isValid = missing.length === 0 && totalImages > 0;

    return {
      valid: isValid,
      message: isValid
        ? "Structure looks good! Ready to upload."
        : `Missing or empty folders: ${missing.join(", ")}`,
      details: {
        anglesFound: requiredAngles.filter((a) => counts[a] > 0).length,
        totalImages: totalImages,
      },
    };
  };

  // Main Action
  const handleUploadAndProcess = async () => {
    if (selectedFiles.length === 0) {
      setError("No files selected");
      return;
    }

    try {
      setIsProcessing(true);
      setError(null);

      // Upload and Process
      const data = await uploadAndProcessFruit(selectedFiles);
      setResult(data);
    } catch (err) {
      console.error("Upload failed:", err);
      setError("Upload failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setFolderName("");
    setValidation(null);
    setResult(null);
    setError(null);
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Input Change Handler
  const handleDirectorySelect = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      processFiles(files);
    }
  };

  const processFiles = (files) => {
    const fileArray = Array.from(files);

    // 1. Set Files
    setSelectedFiles(fileArray);

    // 2. Set Folder Name (Visual only)
    const firstPath =
      fileArray[0].webkitRelativePath || fileArray[0].path || "";
    const name = firstPath.split("/")[0] || "Selected Folder";
    setFolderName(`${name} (${fileArray.length} files)`);

    // 3. Instant Validation
    const validResult = validateFilesClientSide(fileArray);
    setValidation(validResult);

    if (!validResult.valid) {
      setError(validResult.message);
    } else {
      setError(null);
    }
  };

  // Drag and Drop Logic 
  // Helper to read entries recursively with batch reading support
  const traverseFileTree = async (item, path = "") => {
    if (item.isFile) {
      return new Promise((resolve) => {
        item.file((file) => {
          Object.defineProperty(file, "webkitRelativePath", {
            value: path + file.name,
          });
          resolve([file]);
        });
      });
    } else if (item.isDirectory) {
      const dirReader = item.createReader();
      const entries = [];

      // Read all entries in the directory (handling browser batch limits)
      const readEntriesBatch = async () => {
        const batch = await new Promise((resolve, reject) => {
          dirReader.readEntries(resolve, reject);
        });

        if (batch.length > 0) {
          entries.push(...batch);
          await readEntriesBatch(); // Recursive call to get next batch
        }
      };

      await readEntriesBatch();

      const entriesPromises = entries.map((entry) =>
        traverseFileTree(entry, path + item.name + "/"),
      );
      const results = await Promise.all(entriesPromises);
      return results.flat();
    }
    return [];
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isProcessing) return;

    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const item = items[0].webkitGetAsEntry();
      if (item) {
        try {
          // Recursively get all files
          const files = await traverseFileTree(item);
          if (files.length > 0) {
            processFiles(files);
          } else {
            setError("Folder is empty");
          }
        } catch (err) {
          console.error("Error reading folder:", err);
          setError(
            "Failed to read folder contents. Please try using the Browse button.",
          );
        }
      }
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isProcessing) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget)) return;
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div
      className="add-fruit"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      {isDragging && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(52, 152, 219, 0.9)",
            zIndex: 1000,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "8px",
            color: "white",
          }}
        >
          <FiFolder size={64} style={{ marginBottom: "1rem" }} />
          <h2 style={{ color: "white" }}>Drop Folder Here</h2>
        </div>
      )}

      <PageHeader
        title="Add New Fruit"
        subtitle="Add and classify a new fruit object"
      />

      {error && (
        <div
          className="card"
          style={{
            background: "rgba(231, 76, 60, 0.1)",
            border: "1px solid var(--error)",
            marginBottom: "var(--spacing-lg)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--spacing-md)",
            }}
          >
            <FiAlertCircle
              style={{ color: "var(--error)", fontSize: "1.5rem" }}
            />
            <p style={{ color: "var(--error)", margin: 0 }}>{error}</p>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="card instruction-card">
        <div className="card-header">
          <h2 className="card-title">📋 Instructions</h2>
        </div>
        <div className="instructions">
          <div className="instruction-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Prepare Your Images</h4>
              <p>
                Organize images in a folder with subdirectories:{" "}
                <code>an_0</code>, <code>an_1</code>, <code>an_2</code>,{" "}
                <code>an_3</code>
              </p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Select Folder</h4>
              <p>
                Drag and drop your folder here, or click "Browse" to select it.
              </p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Upload & Process</h4>
              <p>
                The system will upload the files and start classification
                automatically.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* File Selection Area */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Folder Selection</h2>
        </div>
        <div className="folder-input-container">
          <div className="input-group-inline">
            <FiFolder size={20} className="input-icon" />

            {/* Read-Only Input for Display */}
            <input
              type="text"
              className="input-field"
              placeholder="No folder selected"
              value={folderName}
              readOnly
              disabled={isProcessing}
            />

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              webkitdirectory="true"
              directory="true"
              multiple
              style={{ display: "none" }}
              onChange={handleDirectorySelect}
            />

            <button
              className="btn btn-browse"
              onClick={handleBrowseClick}
              disabled={isProcessing}
            >
              <FiFolder /> Browse
            </button>
          </div>

          {validation && (
            <div
              className={`validation-result ${validation.valid ? "valid" : "invalid"}`}
            >
              {validation.valid ? (
                <FiCheckCircle className="validation-icon" />
              ) : (
                <FiAlertCircle className="validation-icon" />
              )}
              <div className="validation-content">
                <p className="validation-message">{validation.message}</p>
                {validation.valid && validation.details && (
                  <div className="validation-details">
                    <span>Angles: {validation.details.anglesFound}</span>
                    <span>Files: {validation.details.totalImages}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Processing Steps */}
      {isProcessing && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Processing Pipeline</h2>
          </div>
          <div className="pipeline-steps">
            <div className="pipeline-step active">
              <div className="step-indicator">
                <div className="spinner-small" />
              </div>
              <span>Uploading and Processing...</span>
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card result-card-simple">
          <div className="result-header">
            <FiCheckCircle className="result-success-icon" />
            <div>
              <h2>Classification Complete</h2>
              <p className="result-object-id">{result.objectId}</p>
            </div>
          </div>

          <div className="result-details">
            <div className="result-row">
              <span className="result-label">Classification</span>
              <span className={`result-type type-${result.predictedType}`}>
                {result.predictedType}
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Confidence</span>
              <span className="result-value">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Images Processed</span>
              <span className="result-value">{result.imagesProcessed}</span>
            </div>
            <div className="result-row">
              <span className="result-label">Processing Time</span>
              <span className="result-value">
                {result.processingTime
                  ? result.processingTime.toFixed(1) + "s"
                  : "-"}
              </span>
            </div>
          </div>

          <div className="result-actions-simple">
            <button
              className="btn btn-primary"
              onClick={() => navigate("/results", { state: { refresh: true, newObjectId: result.objectId } })}
            >
              View in Results <FiArrowRight />
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              <FiPlus /> Add Another
            </button>
          </div>
        </div>
      )}

      {/* Centered Add Fruit Button */}
      {!result && (
        <div className="add-fruit-action-container">
          <button
            className="btn btn-primary btn-large btn-add-fruit"
            onClick={handleUploadAndProcess}
            disabled={!validation?.valid || isProcessing}
          >
            {isProcessing ? (
              <>
                <div className="spinner-small" /> Processing...
              </>
            ) : (
              <>
                <FiUpload /> Add Fruit
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default AddFruit;
