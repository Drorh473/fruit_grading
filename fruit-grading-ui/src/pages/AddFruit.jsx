import React, { useState } from "react";
import {
  FiUpload,
  FiFolder,
  FiCheckCircle,
  FiAlertCircle,
} from "react-icons/fi";
import { validateFolder, processFruit } from "../utils/AddFruitApi";
import "./AddFruit.css";

const AddFruit = () => {
  const [folderPath, setFolderPath] = useState("");
  const [validation, setValidation] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [validating, setValidating] = useState(false);

  const handleValidate = async () => {
    if (!folderPath) {
      setError("Please enter a folder path");
      return;
    }

    try {
      setValidating(true);
      setError(null);

      const data = await validateFolder(folderPath);
      setValidation(data);

      if (!data.valid) {
        setError(data.message);
      }
    } catch (err) {
      console.error("Validation failed:", err);
      setValidation({
        valid: false,
        message: "Validation failed. Please check the folder structure.",
      });
      setError("Validation failed: " + err.message);
    } finally {
      setValidating(false);
    }
  };

  const handleProcess = async () => {
    try {
      setIsProcessing(true);
      setResult(null);
      setError(null);

      const data = await processFruit(folderPath, { runTests: false });
      setResult(data);
    } catch (err) {
      console.error("Processing failed:", err);
      setError("Processing failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFolderPath("");
    setValidation(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="add-fruit">
      <div className="page-header">
        <div>
          <h1>Add New Fruit</h1>
          <p className="page-subtitle">Add and classify a new fruit object</p>
        </div>
      </div>

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
                <code>angle_0</code>, <code>angle_1</code>, <code>angle_2</code>
                , <code>angle_3</code>
              </p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Enter Folder Path</h4>
              <p>
                Provide the path to the folder containing your angle directories
              </p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Validate and Process</h4>
              <p>
                System will validate structure, preprocess images, extract
                features, and classify
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Folder Input */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Folder Selection</h2>
        </div>
        <div className="folder-input-container">
          <div className="input-group-inline">
            <FiFolder size={20} className="input-icon" />
            <input
              type="text"
              className="input-field"
              placeholder="Enter folder path (e.g., C:\Fruits\NewFruit)"
              value={folderPath}
              onChange={(e) => {
                setFolderPath(e.target.value);
                setValidation(null);
                setError(null);
              }}
              disabled={isProcessing}
            />
            <button
              className="btn btn-secondary"
              onClick={handleValidate}
              disabled={!folderPath || validating || isProcessing}
            >
              {validating ? "Validating..." : "Validate"}
            </button>
          </div>

          {validation && (
            <div
              className={`validation-result ${
                validation.valid ? "valid" : "invalid"
              }`}
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
                    <span>Angles found: {validation.details.anglesFound}</span>
                    <span>Total images: {validation.details.totalImages}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Process Button */}
      {validation?.valid && !result && (
        <div className="process-section">
          <button
            className="btn btn-primary btn-large"
            onClick={handleProcess}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <div className="spinner-small" />
                Processing...
              </>
            ) : (
              <>
                <FiUpload />
                Process Fruit
              </>
            )}
          </button>
          {isProcessing && (
            <p className="processing-note">
              This may take a few minutes. The system will preprocess images,
              extract features, and classify the fruit.
            </p>
          )}
        </div>
      )}

      {/* Processing Steps (shown while processing) */}
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
              <span>Processing in progress...</span>
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card result-card">
          <div className="card-header">
            <h2 className="card-title">✅ Classification Complete</h2>
          </div>
          <div className="result-content">
            <div className="result-main">
              <div className="result-icon">🎯</div>
              <div className="result-info">
                <h3>
                  Object ID: <code>{result.objectId}</code>
                </h3>
                <div className="result-classification">
                  <span>Classification:</span>
                  <span
                    className={`type-badge-large type-${result.predictedType}`}
                  >
                    {result.predictedType}
                  </span>
                </div>
                <div className="result-confidence">
                  <span>Confidence:</span>
                  <div className="confidence-bar-large">
                    <div
                      className="confidence-fill"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                    <span className="confidence-text">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="result-stats">
              <div className="result-stat">
                <span className="stat-label">Images Processed</span>
                <span className="stat-value">{result.imagesProcessed}</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Processing Time</span>
                <span className="stat-value">
                  {result.processingTime.toFixed(1)}s
                </span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Status</span>
                <span className="status-badge status-success">Complete</span>
              </div>
            </div>

            <div className="result-actions">
              <button className="btn btn-primary">View in Results</button>
              <button className="btn btn-secondary" onClick={handleReset}>
                Add Another Fruit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AddFruit;
