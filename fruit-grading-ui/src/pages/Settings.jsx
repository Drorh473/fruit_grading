import React, { useState, useEffect } from "react";
import {
  FiSave,
  FiRefreshCw,
  FiDatabase,
  FiSettings as FiSettingsIcon,
  FiCheckCircle,
  FiXCircle,
  FiAlertCircle,
  FiLoader,
} from "react-icons/fi";
import "./Settings.css";
import {
  getSettings,
  updateSettings,
  resetSettings,
  testDatabaseConnection,
  testPath,
  validateAllPaths,
  getSystemStatus,
} from "../utils/SettingsApi";

const Settings = () => {
  // Configuration state
  const [config, setConfig] = useState({
    // Database
    dbName: "",
    mongoConnection: "",

    // Paths
    storedDataset: "",
    originalDataset: "",
    processedDataset: "",
  });

  // System status state
  const [systemStatus, setSystemStatus] = useState({
    database: "unknown",
    cameras: [false, false, false, false],
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [originalConfig, setOriginalConfig] = useState(null);

  // Testing states
  const [testingConnection, setTestingConnection] = useState(false);
  const [testingPaths, setTestingPaths] = useState({
    stored: false,
    original: false,
    processed: false,
  });
  const [pathValidation, setPathValidation] = useState({
    stored: null,
    original: null,
    processed: null,
  });

  // Fetch initial data
  useEffect(() => {
    fetchAllData();
  }, []);

  // Track changes
  useEffect(() => {
    if (originalConfig) {
      const changed = JSON.stringify(config) !== JSON.stringify(originalConfig);
      setHasChanges(changed);
    }
  }, [config, originalConfig]);

  // Fetch all settings and system status
  const fetchAllData = async () => {
    setLoading(true);
    setSaveStatus(null);

    try {
      const [settingsData, statusData] = await Promise.all([
        getSettings().catch((err) => {
          console.error("Failed to fetch settings:", err);
          return null;
        }),
        getSystemStatus().catch((err) => {
          console.error("Failed to fetch system status:", err);
          return {
            database: "error",
            cameras: [false, false, false, false],
          };
        }),
      ]);

      if (settingsData) {
        setConfig(settingsData);
        setOriginalConfig(settingsData);
      }

      if (statusData) {
        setSystemStatus(statusData);
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      setSaveStatus({
        success: false,
        message: "Failed to load settings. Please refresh the page.",
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle field change
  const handleChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
    setSaveStatus(null);
  };

  // Save settings to backend
  const handleSave = async () => {
    setIsSaving(true);
    setSaveStatus(null);

    try {
      const updatedSettings = await updateSettings(config);
      setConfig(updatedSettings);
      setOriginalConfig(updatedSettings);
      setHasChanges(false);
      setSaveStatus({
        success: true,
        message: "Settings saved successfully",
      });

      // Refresh system status after save
      const statusData = await getSystemStatus();
      setSystemStatus(statusData);
    } catch (err) {
      console.error("Failed to save settings:", err);
      setSaveStatus({
        success: false,
        message: err.message || "Failed to save settings. Please try again.",
      });
    } finally {
      setIsSaving(false);
    }
  };

  // Reset settings to defaults
  const handleReset = async () => {
    if (
      !window.confirm("Are you sure you want to reset all settings to default?")
    ) {
      return;
    }

    setIsSaving(true);
    setSaveStatus(null);

    try {
      const defaultSettings = await resetSettings();
      setConfig(defaultSettings);
      setOriginalConfig(defaultSettings);
      setHasChanges(false);
      setSaveStatus({
        success: true,
        message: "Settings reset to defaults",
      });

      // Clear path validations
      setPathValidation({
        stored: null,
        original: null,
        processed: null,
      });
    } catch (err) {
      console.error("Failed to reset settings:", err);
      setSaveStatus({
        success: false,
        message: err.message || "Failed to reset settings. Please try again.",
      });
    } finally {
      setIsSaving(false);
    }
  };

  // Test database connection
  const testConnection = async () => {
    setTestingConnection(true);
    setSaveStatus(null);

    try {
      const result = await testDatabaseConnection(
        config.mongoConnection,
        config.dbName
      );

      setSaveStatus({
        success: result.success,
        message:
          result.message ||
          (result.success
            ? "Database connection successful"
            : "Database connection failed"),
      });

      // Update system status
      if (result.success) {
        setSystemStatus((prev) => ({
          ...prev,
          database: "connected",
        }));
      }
    } catch (err) {
      console.error("Database test failed:", err);
      setSaveStatus({
        success: false,
        message: err.message || "Failed to test database connection",
      });
    } finally {
      setTestingConnection(false);
    }
  };

  // Test individual path
  const testSinglePath = async (pathType, pathValue) => {
    setTestingPaths((prev) => ({ ...prev, [pathType]: true }));

    try {
      const result = await testPath(pathValue, pathType);

      setPathValidation((prev) => ({
        ...prev,
        [pathType]: result,
      }));

      if (!result.success) {
        setSaveStatus({
          success: false,
          message: result.message || `Path validation failed for ${pathType}`,
        });
      }
    } catch (err) {
      console.error(`Path test failed for ${pathType}:`, err);
      setPathValidation((prev) => ({
        ...prev,
        [pathType]: {
          success: false,
          message: err.message || "Path test failed",
        },
      }));
    } finally {
      setTestingPaths((prev) => ({ ...prev, [pathType]: false }));
    }
  };

  // Validate all paths at once
  const validatePaths = async () => {
    setTestingPaths({
      stored: true,
      original: true,
      processed: true,
    });
    setSaveStatus(null);

    try {
      const result = await validateAllPaths({
        storedDataset: config.storedDataset,
        originalDataset: config.originalDataset,
        processedDataset: config.processedDataset,
      });

      setPathValidation(result);

      const allValid = Object.values(result).every((r) => r.success);
      setSaveStatus({
        success: allValid,
        message: allValid
          ? "All paths validated successfully"
          : "Some paths failed validation. Check details below.",
      });
    } catch (err) {
      console.error("Path validation failed:", err);
      setSaveStatus({
        success: false,
        message: err.message || "Failed to validate paths",
      });
    } finally {
      setTestingPaths({
        stored: false,
        original: false,
        processed: false,
      });
    }
  };

  // Get validation icon for path
  const getValidationIcon = (pathType) => {
    const validation = pathValidation[pathType];
    const testing = testingPaths[pathType];

    if (testing) {
      return <FiLoader className="spinning" style={{ color: "var(--info)" }} />;
    }
    if (!validation) return null;
    if (validation.success) {
      return <FiCheckCircle style={{ color: "var(--success)" }} />;
    }
    return <FiXCircle style={{ color: "var(--error)" }} />;
  };

  // Loading state
  if (loading) {
    return (
      <div className="settings">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading settings...</p>
        </div>
      </div>
    );
  }

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
          {saveStatus.success ? <FiCheckCircle /> : <FiAlertCircle />}
          {saveStatus.message}
        </div>
      )}

      {hasChanges && (
        <div className="alert alert-info">
          <FiAlertCircle />
          You have unsaved changes. Click "Save Settings" to apply them.
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
              placeholder="fruit_grading"
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
                placeholder="mongodb://localhost:27017"
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={testConnection}
                disabled={testingConnection}
              >
                {testingConnection ? (
                  <>
                    <FiLoader className="spinning" />
                    Testing...
                  </>
                ) : (
                  "Test"
                )}
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
          <button
            className="btn btn-secondary btn-sm"
            onClick={validatePaths}
            disabled={Object.values(testingPaths).some((t) => t)}
          >
            Validate All Paths
          </button>
        </div>
        <div className="settings-grid">
          <div className="setting-item full-width">
            <label className="setting-label">
              Stored Dataset Path
              {getValidationIcon("stored")}
            </label>
            <div className="input-with-action">
              <input
                type="text"
                className="input-field"
                value={config.storedDataset}
                onChange={(e) => handleChange("storedDataset", e.target.value)}
                placeholder="C:\GoogleDrive\Datasets\Dataset"
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => testSinglePath("stored", config.storedDataset)}
                disabled={testingPaths.stored}
              >
                {testingPaths.stored ? (
                  <FiLoader className="spinning" />
                ) : (
                  "Test"
                )}
              </button>
            </div>
            <span className="setting-hint">
              Path where processed images are stored
            </span>
            {pathValidation.stored && !pathValidation.stored.success && (
              <span className="setting-error">
                {pathValidation.stored.message}
              </span>
            )}
          </div>

          <div className="setting-item full-width">
            <label className="setting-label">
              Original Dataset Path
              {getValidationIcon("original")}
            </label>
            <div className="input-with-action">
              <input
                type="text"
                className="input-field"
                value={config.originalDataset}
                onChange={(e) =>
                  handleChange("originalDataset", e.target.value)
                }
                placeholder="C:\GoogleDrive\Datasets\FruitsDataset\data"
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={() =>
                  testSinglePath("original", config.originalDataset)
                }
                disabled={testingPaths.original}
              >
                {testingPaths.original ? (
                  <FiLoader className="spinning" />
                ) : (
                  "Test"
                )}
              </button>
            </div>
            <span className="setting-hint">Path to original raw images</span>
            {pathValidation.original && !pathValidation.original.success && (
              <span className="setting-error">
                {pathValidation.original.message}
              </span>
            )}
          </div>

          <div className="setting-item full-width">
            <label className="setting-label">
              Processed Dataset Path
              {getValidationIcon("processed")}
            </label>
            <div className="input-with-action">
              <input
                type="text"
                className="input-field"
                value={config.processedDataset}
                onChange={(e) =>
                  handleChange("processedDataset", e.target.value)
                }
                placeholder="C:\GoogleDrive\Datasets\ProccesedDataset"
              />
              <button
                className="btn btn-secondary btn-sm"
                onClick={() =>
                  testSinglePath("processed", config.processedDataset)
                }
                disabled={testingPaths.processed}
              >
                {testingPaths.processed ? (
                  <FiLoader className="spinning" />
                ) : (
                  "Test"
                )}
              </button>
            </div>
            <span className="setting-hint">
              Path for preprocessed images output
            </span>
            {pathValidation.processed && !pathValidation.processed.success && (
              <span className="setting-error">
                {pathValidation.processed.message}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="card settings-card">
        <div className="card-header">
          <h2 className="card-title">System Status</h2>
          <button className="btn btn-secondary btn-sm" onClick={fetchAllData}>
            <FiRefreshCw />
            Refresh
          </button>
        </div>
        <div className="status-grid">
          <div className="status-item">
            <span className="status-label">Database Connection</span>
            <span
              className={`status-badge ${
                systemStatus.database === "connected"
                  ? "status-success"
                  : systemStatus.database === "disconnected"
                  ? "status-error"
                  : "status-warning"
              }`}
            >
              {systemStatus.database}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Active Cameras</span>
            <span className="status-value">
              {(() => {
                // Handle both array and object formats for cameras
                if (!systemStatus.cameras) return "0 / 4";
                const camerasArray = Array.isArray(systemStatus.cameras)
                  ? systemStatus.cameras
                  : Object.values(systemStatus.cameras);
                const activeCount = camerasArray.filter((c) => c).length;
                return `${activeCount} / ${camerasArray.length}`;
              })()}
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="settings-actions">
        <button
          className="btn btn-secondary"
          onClick={handleReset}
          disabled={isSaving}
        >
          <FiRefreshCw />
          Reset to Defaults
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={isSaving || !hasChanges}
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
