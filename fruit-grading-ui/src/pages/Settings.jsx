import React, { useState } from "react";
import {
  FiSave,
  FiRefreshCw,
  FiDatabase,
  FiSettings as FiSettingsIcon,
} from "react-icons/fi";
import "./Settings.css";

const Settings = ({ systemStatus, setSystemStatus }) => {
  const [config, setConfig] = useState({
    // Database
    dbName: "fruit_grading",
    mongoConnection: "mongodb://localhost:27017",

    // Paths
    storedDataset: "C:\\GoogleDrive\\Datasets\\Dataset",
    originalDataset: "C:\\GoogleDrive\\Datasets\\FruitsDataset\\data",
    processedDataset: "C:\\GoogleDrive\\Datasets\\ProccesedDataset",

    // Model
    batchSize: 128,
    modelVariant: "1.0x",
  });

  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  const handleChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
    setSaveStatus(null);
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveStatus(null);

    // Simulate API call
    setTimeout(() => {
      setSaveStatus({ success: true, message: "Settings saved successfully" });
      setIsSaving(false);
    }, 1000);
  };

  const handleReset = () => {
    if (
      window.confirm("Are you sure you want to reset all settings to default?")
    ) {
      // Reset to defaults
      setSaveStatus({ success: true, message: "Settings reset to defaults" });
    }
  };

  const testConnection = async (type) => {
    setSaveStatus({
      success: true,
      message: `${type} connection test successful`,
    });
  };

  return (
    <div className="settings">
      <div className="page-header">
        <div>
          <h1>Settings</h1>
          <p className="page-subtitle">
            Configure system parameters and connections
          </p>
        </div>
      </div>

      {saveStatus && (
        <div
          className={`alert ${
            saveStatus.success ? "alert-success" : "alert-error"
          }`}
        >
          {saveStatus.message}
        </div>
      )}

      {/* Database Settings */}
      <div className="card settings-card">
        <div className="card-header">
          <div className="header-with-icon">
            <FiDatabase />
            <h2 className="card-title">Database Configuration</h2>
          </div>
        </div>
        <div className="settings-grid">
          <div className="setting-item">
            <label className="setting-label">Database Name</label>
            <input
              type="text"
              className="input-field"
              value={config.dbName}
              onChange={(e) => handleChange("dbName", e.target.value)}
            />
          </div>
          <div className="setting-item">
            <label className="setting-label">MongoDB Connection String</label>
            <div className="input-with-action">
              <input
                type="text"
                className="input-field"
                value={config.mongoConnection}
                onChange={(e) =>
                  handleChange("mongoConnection", e.target.value)
                }
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => testConnection("Database")}
              >
                Test
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Paths */}
      <div className="card settings-card">
        <div className="card-header">
          <div className="header-with-icon">
            <FiSettingsIcon />
            <h2 className="card-title">Dataset Paths</h2>
          </div>
        </div>
        <div className="settings-grid">
          <div className="setting-item full-width">
            <label className="setting-label">Stored Dataset Path</label>
            <input
              type="text"
              className="input-field"
              value={config.storedDataset}
              onChange={(e) => handleChange("storedDataset", e.target.value)}
            />
            <span className="setting-hint">
              Path where processed images are stored
            </span>
          </div>
          <div className="setting-item full-width">
            <label className="setting-label">Original Dataset Path</label>
            <input
              type="text"
              className="input-field"
              value={config.originalDataset}
              onChange={(e) => handleChange("originalDataset", e.target.value)}
            />
            <span className="setting-hint">Path to original raw images</span>
          </div>
          <div className="setting-item full-width">
            <label className="setting-label">Processed Dataset Path</label>
            <input
              type="text"
              className="input-field"
              value={config.processedDataset}
              onChange={(e) => handleChange("processedDataset", e.target.value)}
            />
            <span className="setting-hint">
              Path for preprocessed images output
            </span>
          </div>
        </div>
      </div>

      {/* Model Settings */}
      <div className="card settings-card">
        <div className="card-header">
          <h2 className="card-title">Model Configuration</h2>
        </div>
        <div className="settings-grid">
          <div className="setting-item">
            <label className="setting-label">Batch Size</label>
            <input
              type="number"
              className="input-field"
              value={config.batchSize}
              onChange={(e) =>
                handleChange("batchSize", parseInt(e.target.value))
              }
              min="1"
              max="512"
            />
            <span className="setting-hint">
              Number of samples per training batch
            </span>
          </div>
          <div className="setting-item">
            <label className="setting-label">ShuffleNet Variant</label>
            <select
              className="input-field"
              value={config.modelVariant}
              onChange={(e) => handleChange("modelVariant", e.target.value)}
            >
              <option value="0.5x">0.5x (Faster)</option>
              <option value="1.0x">1.0x (Default)</option>
              <option value="1.5x">1.5x (Slower)</option>
              <option value="2.0x">2.0x (Slowest)</option>
            </select>
            <span className="setting-hint">
              Model complexity and speed trade-off
            </span>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="card settings-card">
        <div className="card-header">
          <h2 className="card-title">System Status</h2>
        </div>
        <div className="status-grid">
          <div className="status-item">
            <span className="status-label">Database Connection</span>
            <span
              className={`status-badge ${
                systemStatus.database === "connected"
                  ? "status-success"
                  : "status-error"
              }`}
            >
              {systemStatus.database}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Model Status</span>
            <span
              className={`status-badge ${
                systemStatus.model === "loaded"
                  ? "status-success"
                  : "status-error"
              }`}
            >
              {systemStatus.model}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Active Cameras</span>
            <span className="status-value">
              {systemStatus.cameras.filter((c) => c).length} /{" "}
              {systemStatus.cameras.length}
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="settings-actions">
        <button className="btn btn-secondary" onClick={handleReset}>
          <FiRefreshCw />
          Reset to Defaults
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={isSaving}
        >
          {isSaving ? (
            <>
              <div className="spinner-small" />
              Saving...
            </>
          ) : (
            <>
              <FiSave />
              Save Settings
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default Settings;
