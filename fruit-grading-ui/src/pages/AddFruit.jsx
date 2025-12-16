import React, { useState } from "react";
import {
  FiUpload,
  FiFolder,
  FiCheckCircle,
  FiAlertCircle,
} from "react-icons/fi";
import "./AddFruit.css";

const AddFruit = () => {
  const [folderPath, setFolderPath] = useState("");
  const [validation, setValidation] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  const validateFolder = async () => {
    if (!folderPath) {
      setValidation({ valid: false, message: "Please enter a folder path" });
      return;
    }

    // Simulate validation API call
    setTimeout(() => {
      const mockValidation = {
        valid: true,
        message: "Folder structure is valid",
        details: {
          anglesFound: 4,
          imagesPerAngle: [15, 15, 15, 15],
          totalImages: 60,
        },
      };
      setValidation(mockValidation);
    }, 1000);
  };

  const processFruit = async () => {
    setIsProcessing(true);
    setResult(null);

    // Simulate processing pipeline
    setTimeout(() => {
      const mockResult = {
        objectId: "obj0015",
        predictedType: "market",
        confidence: 0.94,
        imagesProcessed: 60,
        processingTime: 45.3,
      };
      setResult(mockResult);
      setIsProcessing(false);
    }, 5000);
  };

  return (
    <div className="add-fruit">
      <div className="page-header">
        <div>
          <h1>Add New Fruit</h1>
          <p className="page-subtitle">Add and classify a new fruit object</p>
        </div>
      </div>

      {/* Instructions */}
      <div className="card instruction-card">
        <div className="card-header">
          <h2 className="card-title"> Instructions</h2>
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
              }}
            />
            <button
              className="btn btn-secondary"
              onClick={validateFolder}
              disabled={!folderPath}
            >
              Validate
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
            onClick={processFruit}
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
              <span>Validating folder structure...</span>
            </div>
            <div className="pipeline-step">
              <div className="step-indicator" />
              <span>Inserting into database...</span>
            </div>
            <div className="pipeline-step">
              <div className="step-indicator" />
              <span>Copying to stored dataset...</span>
            </div>
            <div className="pipeline-step">
              <div className="step-indicator" />
              <span>Preprocessing images...</span>
            </div>
            <div className="pipeline-step">
              <div className="step-indicator" />
              <span>Extracting features...</span>
            </div>
            <div className="pipeline-step">
              <div className="step-indicator" />
              <span>Running classification...</span>
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card result-card">
          <div className="card-header">
            <h2 className="card-title">âœ“ Classification Complete</h2>
          </div>
          <div className="result-content">
            <div className="result-main">
              <div className="result-icon">ðŸŽ¯</div>
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
                <span className="stat-value">{result.processingTime}s</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Status</span>
                <span className="status-badge status-success">Complete</span>
              </div>
            </div>

            <div className="result-actions">
              <button className="btn btn-primary">View in Results</button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setFolderPath("");
                  setValidation(null);
                  setResult(null);
                }}
              >
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
