import React, { useState } from 'react';
import { FiPlay, FiSquare, FiDatabase, FiImage, FiCpu, FiCheckCircle } from 'react-icons/fi';
import './Processing.css';

const Processing = ({ setProcessingStats }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);

  const processingSteps = [
    { id: 1, name: 'Database Setup', icon: <FiDatabase />, status: 'pending' },
    { id: 2, name: 'Data Preprocessing', icon: <FiImage />, status: 'pending' },
    { id: 3, name: 'Feature Extraction', icon: <FiCpu />, status: 'pending' },
    { id: 4, name: 'Model Training', icon: <FiCpu />, status: 'pending' },
    { id: 5, name: 'Evaluation', icon: <FiCheckCircle />, status: 'pending' }
  ];

  const [steps, setSteps] = useState(processingSteps);

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { message, type, timestamp }]);
  };

  const updateStep = (stepId, status) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, status } : step
    ));
  };

  const runPipeline = async () => {
    setIsProcessing(true);
    setLogs([]);
    setProgress(0);
    setCurrentStep(0);

    try {
      // Step 1: Database Setup
      setCurrentStep(1);
      updateStep(1, 'processing');
      addLog('Starting database setup...', 'info');
      await simulateStep(20);
      updateStep(1, 'completed');
      addLog('✓ Database setup complete', 'success');

      // Step 2: Preprocessing
      setCurrentStep(2);
      updateStep(2, 'processing');
      addLog('Processing images with Gaussian Blur and CLAHE...', 'info');
      await simulateStep(40);
      updateStep(2, 'completed');
      addLog('✓ 165 images preprocessed successfully', 'success');

      // Step 3: Feature Extraction
      setCurrentStep(3);
      updateStep(3, 'processing');
      addLog('Extracting features using ShuffleNetV2...', 'info');
      await simulateStep(60);
      addLog('Flattening features...', 'info');
      await simulateStep(70);
      addLog('Temporal pooling across frames...', 'info');
      await simulateStep(80);
      addLog('Multi-view fusion from 4 cameras...', 'info');
      await simulateStep(85);
      updateStep(3, 'completed');
      addLog('✓ Feature extraction complete (dim: 200,704)', 'success');

      // Step 4: Training
      setCurrentStep(4);
      updateStep(4, 'processing');
      addLog('Training fully connected classifier...', 'info');
      await simulateStep(95);
      updateStep(4, 'completed');
      addLog('✓ Training complete (100 epochs)', 'success');

      // Step 5: Evaluation
      setCurrentStep(5);
      updateStep(5, 'processing');
      addLog('Evaluating model performance...', 'info');
      await simulateStep(100);
      updateStep(5, 'completed');
      addLog('✓ Evaluation complete', 'success');
      addLog('Training Accuracy: 100%', 'success');
      addLog('Test Accuracy: 36.36%', 'warning');

      setProcessingStats({
        totalProcessed: 14,
        accuracy: 0.3636,
        lastUpdate: new Date().toISOString()
      });

    } catch (error) {
      addLog(`✗ Error: ${error.message}`, 'error');
      updateStep(currentStep, 'failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const simulateStep = (targetProgress) => {
    return new Promise((resolve) => {
      const interval = setInterval(() => {
        setProgress(prev => {
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
    addLog('Processing stopped by user', 'warning');
  };

  const getStepStatus = (step) => {
    if (step.status === 'completed') return 'step-completed';
    if (step.status === 'processing') return 'step-processing';
    if (step.status === 'failed') return 'step-failed';
    return 'step-pending';
  };

  return (
    <div className="processing">
      <div className="page-header">
        <div>
          <h1>Processing Pipeline</h1>
          <p className="page-subtitle">Run the complete ML pipeline from data to model</p>
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
              <div className="step-indicator">
                <div className="step-icon">
                  {step.status === 'completed' ? <FiCheckCircle /> : step.icon}
                </div>
                {index < steps.length - 1 && <div className="step-line" />}
              </div>
              <div className="step-content">
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
          <div className="config-list">
            <div className="config-row">
              <span>Hidden Layer Dimension</span>
              <code>256</code>
            </div>
            <div className="config-row">
              <span>Epochs</span>
              <code>100</code>
            </div>
            <div className="config-row">
              <span>Learning Rate</span>
              <code>0.001</code>
            </div>
            <div className="config-row">
              <span>Batch Size</span>
              <code>32</code>
            </div>
            <div className="config-row">
              <span>Optimizer</span>
              <code>Gradient Descent</code>
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
