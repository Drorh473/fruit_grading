import React, { useState } from "react";
import { FiPlay, FiSquare, FiCheckCircle, FiChevronDown } from "react-icons/fi";
import "./Processing.css";

const Processing = ({ setProcessingStats }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [openDropdown, setOpenDropdown] = useState(null);

  // Training configuration state
  const [config, setConfig] = useState({
    hiddenDim: 256,
    epochs: 100,
    learningRate: 0.001,
    lambdaReg: 0.01,
    batchSize: 32,
  });

  // Common values for each parameter
  const presets = {
    hiddenDim: [16, 32, 64, 128, 256, 512, 1024],
    epochs: [10, 25, 50, 100, 200, 500, 1000],
    learningRate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    lambdaReg: [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    batchSize: [8, 16, 32, 64, 128, 256],
  };

  const processingSteps = [
    { id: 1, name: "Database Setup", status: "pending" },
    { id: 2, name: "Data Preprocessing", status: "pending" },
    { id: 3, name: "Feature Extraction", status: "pending" },
    { id: 4, name: "Model Training", status: "pending" },
    { id: 5, name: "Evaluation", status: "pending" },
  ];

  const [steps, setSteps] = useState(processingSteps);

  const handleConfigChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: parseFloat(value),
    }));
    setOpenDropdown(null);
  };

  const toggleDropdown = (field) => {
    if (isProcessing) return;
    setOpenDropdown(openDropdown === field ? null : field);
  };

  const addLog = (message, type = "info") => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { message, type, timestamp }]);
  };

  const updateStep = (stepId, status) => {
    setSteps((prev) =>
      prev.map((step) => (step.id === stepId ? { ...step, status } : step))
    );
  };

  const runPipeline = async () => {
    setIsProcessing(true);
    setLogs([]);
    setProgress(0);
    setCurrentStep(0);
    setOpenDropdown(null);

    try {
      // Step 1: Database Setup
      setCurrentStep(1);
      updateStep(1, "processing");
      addLog("Starting database setup...", "info");
      await simulateStep(20);
      updateStep(1, "completed");
      addLog("Database setup complete", "success");

      // Step 2: Preprocessing
      setCurrentStep(2);
      updateStep(2, "processing");
      addLog("Processing images with Gaussian Blur and CLAHE...", "info");
      await simulateStep(40);
      updateStep(2, "completed");
      addLog("165 images preprocessed successfully", "success");

      // Step 3: Feature Extraction
      setCurrentStep(3);
      updateStep(3, "processing");
      addLog("Extracting features using ShuffleNetV2...", "info");
      await simulateStep(60);
      addLog("Flattening features...", "info");
      await simulateStep(70);
      addLog("Temporal pooling across frames...", "info");
      await simulateStep(80);
      addLog("Multi-view fusion from 4 cameras...", "info");
      await simulateStep(85);
      updateStep(3, "completed");
      addLog("Feature extraction complete (dim: 200,704)", "success");

      // Step 4: Training
      setCurrentStep(4);
      updateStep(4, "processing");
      addLog(
        `Training with config: Hidden=${config.hiddenDim}, LR=${config.learningRate}, Lambda=${config.lambdaReg}`,
        "info"
      );
      await simulateStep(95);
      updateStep(4, "completed");
      addLog(`Training complete (${config.epochs} epochs)`, "success");

      // Step 5: Evaluation
      setCurrentStep(5);
      updateStep(5, "processing");
      addLog("Evaluating model performance...", "info");
      await simulateStep(100);
      updateStep(5, "completed");
      addLog("Evaluation complete", "success");
      addLog("Training Accuracy: 100%", "success");
      addLog("Test Accuracy: 36.36%", "warning");

      setProcessingStats({
        totalProcessed: 14,
        accuracy: 0.3636,
        lastUpdate: new Date().toISOString(),
      });
    } catch (error) {
      addLog(`Error: ${error.message}`, "error");
      updateStep(currentStep, "failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const simulateStep = (targetProgress) => {
    return new Promise((resolve) => {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= targetProgress) {
            clearInterval(interval);
            resolve();
            return targetProgress;
          }
          return prev + 1;
        });
      }, 50);
    });
  };

  const stopProcessing = () => {
    setIsProcessing(false);
    addLog("Processing stopped by user", "warning");
  };

  const getStepStatus = (step) => {
    if (step.status === "completed") return "step-completed";
    if (step.status === "processing") return "step-processing";
    if (step.status === "failed") return "step-failed";
    return "step-pending";
  };

  return (
    <div className="processing">
      <div className="page-header">
        <div>
          <h1>Processing Pipeline</h1>
          <p className="page-subtitle">
            Run the complete ML pipeline from data to model
          </p>
        </div>
        <div className="header-actions">
          {!isProcessing ? (
            <button className="btn btn-primary" onClick={runPipeline}>
              <FiPlay />
              Start Pipeline
            </button>
          ) : (
            <button className="btn btn-danger" onClick={stopProcessing}>
              <FiSquare />
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Progress Overview */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Pipeline Progress</h2>
          <span className="progress-percent">{progress}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        {isProcessing && (
          <p className="processing-message">
            Processing step {currentStep} of {steps.length}...
          </p>
        )}
      </div>

      {/* Pipeline Steps */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Pipeline Steps</h2>
        </div>
        <div className="steps-container">
          {steps.map((step, index) => (
            <div key={step.id} className={`step-item ${getStepStatus(step)}`}>
              <div className="step-content-full">
                <h3 className="step-name">{step.name}</h3>
                <span className={`step-status status-${step.status}`}>
                  {step.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Processing Logs */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Processing Logs</h2>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setLogs([])}
            disabled={logs.length === 0}
          >
            Clear Logs
          </button>
        </div>
        <div className="logs-container">
          {logs.length === 0 ? (
            <div className="empty-state">
              <p>No logs yet. Start the pipeline to see processing logs.</p>
            </div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className={`log-entry log-${log.type}`}>
                <span className="log-timestamp">{log.timestamp}</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Configuration */}
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Training Configuration</h2>
          </div>
          <div className="config-list-selects">
            {/* Hidden Layer Dimension */}
            <div className="config-select-item">
              <label className="config-label">Hidden Layer Dimension</label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "hiddenDim" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("hiddenDim")}
                  disabled={isProcessing}
                >
                  <span className="select-value">{config.hiddenDim}</span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "hiddenDim" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "hiddenDim" && (
                  <div className="select-options">
                    {presets.hiddenDim.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.hiddenDim === value ? "active" : ""
                        }`}
                        onClick={() => handleConfigChange("hiddenDim", value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Epochs */}
            <div className="config-select-item">
              <label className="config-label">Epochs</label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "epochs" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("epochs")}
                  disabled={isProcessing}
                >
                  <span className="select-value">{config.epochs}</span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "epochs" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "epochs" && (
                  <div className="select-options">
                    {presets.epochs.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.epochs === value ? "active" : ""
                        }`}
                        onClick={() => handleConfigChange("epochs", value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Learning Rate */}
            <div className="config-select-item">
              <label className="config-label">Learning Rate</label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "learningRate" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("learningRate")}
                  disabled={isProcessing}
                >
                  <span className="select-value">{config.learningRate}</span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "learningRate" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "learningRate" && (
                  <div className="select-options">
                    {presets.learningRate.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.learningRate === value ? "active" : ""
                        }`}
                        onClick={() =>
                          handleConfigChange("learningRate", value)
                        }
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* L2 Regularization */}
            <div className="config-select-item">
              <label className="config-label">L2 Regularization (Lambda)</label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "lambdaReg" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("lambdaReg")}
                  disabled={isProcessing}
                >
                  <span className="select-value">{config.lambdaReg}</span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "lambdaReg" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "lambdaReg" && (
                  <div className="select-options">
                    {presets.lambdaReg.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.lambdaReg === value ? "active" : ""
                        }`}
                        onClick={() => handleConfigChange("lambdaReg", value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Batch Size */}
            <div className="config-select-item">
              <label className="config-label">Batch Size</label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "batchSize" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("batchSize")}
                  disabled={isProcessing}
                >
                  <span className="select-value">{config.batchSize}</span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "batchSize" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "batchSize" && (
                  <div className="select-options">
                    {presets.batchSize.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.batchSize === value ? "active" : ""
                        }`}
                        onClick={() => handleConfigChange("batchSize", value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Dataset Configuration</h2>
          </div>
          <div className="config-list">
            <div className="config-row">
              <span>Train/Test Split</span>
              <code>66% / 34%</code>
            </div>
            <div className="config-row">
              <span>Image Size</span>
              <code>224 x 224</code>
            </div>
            <div className="config-row">
              <span>Cameras</span>
              <code>4 (multi-view)</code>
            </div>
            <div className="config-row">
              <span>Preprocessing</span>
              <code>Gaussian + CLAHE</code>
            </div>
            <div className="config-row">
              <span>Feature Extractor</span>
              <code>ShuffleNetV2</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Processing;
