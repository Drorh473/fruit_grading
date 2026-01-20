import React, { useState, useEffect } from "react";
import {
  FiPlay,
  FiSquare,
  FiAlertCircle,
  FiRefreshCw,
  FiChevronDown,
} from "react-icons/fi";
import {
  startPipeline,
  stopPipeline,
  getPipelineStatus,
  getPipelineLogs,
  getPipelineConfig,
} from "../utils/processingApi";
import "./Processing.css";

const Processing = ({ setProcessingStats }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [openDropdown, setOpenDropdown] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  // Training configuration state
  const [config, setConfig] = useState({
    hiddenDim: 256,
    epochs: 100,
    learningRate: 0.001,
    lambdaReg: 0.01,
    batchSize: 32,
    pcaComponents: 0,
  });

  // Common values for each parameter
  const presets = {
    hiddenDim: [4, 8, 16, 32, 64, 128, 256],
    epochs: [10, 25, 50, 100, 200, 500, 1000],
    learningRate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    lambdaReg: [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    batchSize: [8, 16, 32, 64, 128, 256],
    pcaComponents: [0, 8, 16, 32, 64, 128, 256, 512],
  };

  // Updated steps - Testing is now Step 1
  const defaultSteps = [
    { id: 1, name: "Testing", status: "pending" },
    { id: 2, name: "Database Setup", status: "pending" },
    { id: 3, name: "Data Preprocessing", status: "pending" },
    { id: 4, name: "Feature Extraction", status: "pending" },
    { id: 5, name: "Model Training", status: "pending" },
    { id: 6, name: "Evaluation", status: "pending" },
  ];

  const [steps, setSteps] = useState(defaultSteps);

  // Calculate progress based on completed steps only
  const calculateProgress = (stepsArray) => {
    const completedCount = stepsArray.filter(
      (step) => step.status === "completed",
    ).length;
    return Math.round((completedCount / stepsArray.length) * 100);
  };

  useEffect(() => {
    loadConfig();
    checkStatus();
  }, []);

  useEffect(() => {
    let interval;
    if (isProcessing) {
      interval = setInterval(updateStatus, 10000);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const loadConfig = async () => {
    try {
      const data = await getPipelineConfig();
      setConfig({
        hiddenDim: data.hiddenDim || 256,
        epochs: data.epochs || 100,
        learningRate: data.learningRate || 0.001,
        lambdaReg: data.lambdaReg || 0.01,
        batchSize: data.batchSize || 32,
        pcaComponents: data.pcaComponents || 0,
      });
    } catch (err) {
      console.error("Failed to load config:", err);
      setError("Failed to load configuration");
    } finally {
      setLoading(false);
    }
  };

  const checkStatus = async () => {
    try {
      const statusData = await getPipelineStatus();
      if (statusData.running) {
        setIsProcessing(true);
        setStatus(statusData);
        updateStepsFromStatus(statusData);
      }
    } catch (err) {
      console.error("Failed to check status:", err);
    }
  };

  const updateStatus = async () => {
    try {
      const [statusData, logsData] = await Promise.all([
        getPipelineStatus(),
        getPipelineLogs(50),
      ]);

      setStatus(statusData);
      setLogs(logsData);
      updateStepsFromStatus(statusData);

      if (statusData.status === "completed" || statusData.status === "failed") {
        setIsProcessing(false);

        if (statusData.status === "completed" && setProcessingStats) {
          setProcessingStats({
            totalProcessed: statusData.totalProcessed || 0,
            accuracy: statusData.accuracy || 0,
            lastUpdate: new Date().toISOString(),
          });
        }

        if (statusData.status === "failed") {
          setError("Pipeline failed. Check logs for details.");
        }
      }
    } catch (err) {
      console.error("Failed to update status:", err);
      setError("Failed to update pipeline status");
    }
  };

  const updateStepsFromStatus = (statusData) => {
    if (statusData.steps) {
      setSteps(statusData.steps);
    } else if (statusData.currentStep) {
      const updatedSteps = defaultSteps.map((step, index) => {
        if (index < statusData.currentStep - 1) {
          return { ...step, status: "completed" };
        } else if (index === statusData.currentStep - 1) {
          return { ...step, status: "processing" };
        }
        return step;
      });
      setSteps(updatedSteps);
    }
  };

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

  const handleStart = async () => {
    try {
      setError(null);
      setLogs([]);
      setSteps(defaultSteps);
      setOpenDropdown(null);

      const response = await startPipeline({
        skipTests: false,
        hiddenDim: config.hiddenDim,
        epochs: config.epochs,
        learningRate: config.learningRate,
        lambdaReg: config.lambdaReg,
        batchSize: config.batchSize,
        pcaComponents: config.pcaComponents,
      });

      if (response.success) {
        setIsProcessing(true);
      } else {
        setError(response.message || "Failed to start pipeline");
      }
    } catch (err) {
      console.error("Failed to start pipeline:", err);
      setError("Failed to start pipeline: " + err.message);
    }
  };

  const handleStop = async () => {
    try {
      await stopPipeline();
      setIsProcessing(false);
      setError(null);
      setSteps(defaultSteps);
      setLogs((prev) => [
        ...prev,
        {
          message: "Processing stopped by user",
          type: "warning",
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (err) {
      console.error("Failed to stop pipeline:", err);
      setError("Failed to stop pipeline: " + err.message);
    }
  };

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    setSteps(defaultSteps);
    setLogs([]);
    await Promise.all([loadConfig(), checkStatus()]);
    setLoading(false);
  };

  const getStepStatus = (step) => {
    if (step.status === "completed") return "step-completed";
    if (step.status === "processing") return "step-processing";
    if (step.status === "failed") return "step-failed";
    return "step-pending";
  };

  const displayProgress = calculateProgress(steps);

  if (loading) {
    return (
      <div className="processing">
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "400px",
          }}
        >
          <div className="spinner" />
        </div>
      </div>
    );
  }

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
          <button
            className="btn btn-secondary"
            onClick={handleRefresh}
            disabled={isProcessing}
          >
            <FiRefreshCw />
            Refresh
          </button>
          {!isProcessing ? (
            <button className="btn btn-primary" onClick={handleStart}>
              <FiPlay />
              Start Pipeline
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleStop}>
              <FiSquare />
              Stop
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="alert alert-error">
          <FiAlertCircle />
          <span>{error}</span>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Pipeline Progress</h2>
          <span className="progress-percent">{displayProgress}%</span>
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${displayProgress}%` }}
          />
        </div>
        {isProcessing && (
          <p className="processing-message">
            Processing step {status?.currentStep || 0} of {steps.length}...
          </p>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Pipeline Steps</h2>
        </div>
        <div className="steps-container">
          {steps.map((step) => (
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
                <span className="log-timestamp">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="log-message">{log.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card config-card">
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

            {/* PCA Components */}
            <div className="config-select-item">
              <label className="config-label">
                PCA Components (0 = disabled)
              </label>
              <div className="select-dropdown">
                <button
                  className={`select-button ${
                    openDropdown === "pcaComponents" ? "open" : ""
                  }`}
                  onClick={() => toggleDropdown("pcaComponents")}
                  disabled={isProcessing}
                >
                  <span className="select-value">
                    {config.pcaComponents === 0
                      ? "Disabled"
                      : config.pcaComponents}
                  </span>
                  <FiChevronDown
                    className={`chevron ${
                      openDropdown === "pcaComponents" ? "rotate" : ""
                    }`}
                  />
                </button>
                {openDropdown === "pcaComponents" && (
                  <div className="select-options">
                    {presets.pcaComponents.map((value) => (
                      <button
                        key={value}
                        className={`select-option ${
                          config.pcaComponents === value ? "active" : ""
                        }`}
                        onClick={() =>
                          handleConfigChange("pcaComponents", value)
                        }
                      >
                        {value === 0 ? "Disabled" : value}
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
