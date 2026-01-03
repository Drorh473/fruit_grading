/**
 * Test Utilities
 * Reusable helpers for frontend testing
 */

import { render } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../../src/context/AuthContext';

/**
 * Custom render function that wraps components with required providers
 */
export const renderWithProviders = (
  ui,
  {
    initialAuthState = null,
    route = '/',
    ...renderOptions
  } = {}
) => {
  // Set initial route
  window.history.pushState({}, 'Test page', route);

  // Create wrapper with all providers
  const Wrapper = ({ children }) => (
    <BrowserRouter>
      <AuthProvider initialState={initialAuthState}>
        {children}
      </AuthProvider>
    </BrowserRouter>
  );

  return render(ui, { wrapper: Wrapper, ...renderOptions });
};

/**
 * Create mock user for authentication tests
 */
export const createMockUser = (role = 'user') => ({
  username: role === 'admin' ? 'admin' : 'testuser',
  role,
  loginTime: new Date().toISOString(),
});

/**
 * Mock localStorage
 */
export const mockLocalStorage = () => {
  const store = {};
  return {
    getItem: (key) => store[key] || null,
    setItem: (key, value) => {
      store[key] = value.toString();
    },
    removeItem: (key) => {
      delete store[key];
    },
    clear: () => {
      Object.keys(store).forEach(key => delete store[key]);
    },
  };
};

/**
 * Mock API responses
 */
export const mockApiResponse = (data, options = {}) => {
  const {
    status = 200,
    ok = true,
    delay = 0,
  } = options;

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        ok,
        status,
        json: async () => data,
        text: async () => JSON.stringify(data),
        headers: new Headers(),
      });
    }, delay);
  });
};

/**
 * Mock API error
 */
export const mockApiError = (message = 'Network error', status = 500) => {
  return Promise.reject({
    message,
    status,
    response: {
      status,
      data: { error: message },
    },
  });
};

/**
 * Wait for element with timeout
 */
export const waitForElement = async (callback, timeout = 3000) => {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      const element = callback();
      if (element) return element;
    } catch (error) {
      // Element not found yet
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Element not found within timeout');
};

/**
 * Create mock system status
 */
export const createMockSystemStatus = (overrides = {}) => ({
  database: 'connected',
  model: 'loaded',
  cameras: [true, true, true, true],
  ...overrides,
});

/**
 * Create mock processing stats
 */
export const createMockProcessingStats = (overrides = {}) => ({
  totalProcessed: 150,
  accuracy: 0.92,
  lastUpdate: new Date().toISOString(),
  ...overrides,
});

/**
 * Create mock dashboard data
 */
export const createMockDashboardData = (overrides = {}) => ({
  totalProcessed: 1250,
  accuracy: 0.94,
  systemUptime: 48.5,
  cameraStatus: {
    camera1: true,
    camera2: true,
    camera3: true,
    camera4: false,
  },
  recentActivity: [
    {
      id: 1,
      type: 'classification',
      message: 'Apple classified as Premium',
      timestamp: new Date(Date.now() - 300000).toISOString(),
    },
    {
      id: 2,
      type: 'system',
      message: 'Camera 4 reconnected',
      timestamp: new Date(Date.now() - 600000).toISOString(),
    },
  ],
  ...overrides,
});

/**
 * Create mock results data
 */
export const createMockResults = (count = 10) => {
  const grades = ['market', 'standard', 'premium'];
  const fruitTypes = ['apple', 'orange', 'banana', 'mango'];
  
  return Array.from({ length: count }, (_, i) => ({
    id: `result_${i + 1}`,
    fruitType: fruitTypes[i % fruitTypes.length],
    grade: grades[i % grades.length],
    confidence: 0.85 + Math.random() * 0.14,
    timestamp: new Date(Date.now() - i * 3600000).toISOString(),
    objectId: `obj_${i + 1}`,
    images: [`img_${i}_1.jpg`, `img_${i}_2.jpg`, `img_${i}_3.jpg`, `img_${i}_4.jpg`],
  }));
};

/**
 * Create mock pipeline status
 */
export const createMockPipelineStatus = (overrides = {}) => ({
  running: false,
  status: 'idle',
  currentStep: 0,
  progress: 0,
  steps: [
    { id: 1, name: 'Database Setup', status: 'pending' },
    { id: 2, name: 'Data Preprocessing', status: 'pending' },
    { id: 3, name: 'Feature Extraction', status: 'pending' },
    { id: 4, name: 'Model Training', status: 'pending' },
    { id: 5, name: 'Evaluation', status: 'pending' },
  ],
  ...overrides,
});

/**
 * Create mock camera data
 */
export const createMockCameraData = () => [
  {
    id: 0,
    name: 'Camera 0',
    status: true,
    angle: 'Front View',
    fps: 30,
    resolution: '224x224',
    captureSuccess: 99.8,
    quality: 92,
    uptime: 24.5,
    framesProcessed: 65432,
    errorCount: 2,
  },
  {
    id: 1,
    name: 'Camera 1',
    status: true,
    angle: 'Right View',
    fps: 30,
    resolution: '224x224',
    captureSuccess: 99.3,
    quality: 89,
    uptime: 24.5,
    framesProcessed: 65128,
    errorCount: 5,
  },
  {
    id: 2,
    name: 'Camera 2',
    status: true,
    angle: 'Back View',
    fps: 30,
    resolution: '224x224',
    captureSuccess: 99.5,
    quality: 91,
    uptime: 24.5,
    framesProcessed: 65298,
    errorCount: 3,
  },
  {
    id: 3,
    name: 'Camera 3',
    status: false,
    angle: 'Left View',
    fps: 0,
    resolution: '224x224',
    captureSuccess: 96.2,
    quality: 85,
    uptime: 12.3,
    framesProcessed: 32156,
    errorCount: 12,
    lastError: 'Connection timeout',
  },
];

/**
 * Simulate user interactions
 */
export const simulateTyping = async (element, text, delay = 50) => {
  for (const char of text) {
    element.value += char;
    element.dispatchEvent(new Event('input', { bubbles: true }));
    await new Promise(resolve => setTimeout(resolve, delay));
  }
};

/**
 * Create mock settings data
 */
export const createMockSettings = (overrides = {}) => ({
  system: {
    autoRestart: true,
    debugMode: false,
    logLevel: 'info',
    backupEnabled: true,
  },
  processing: {
    batchSize: 32,
    confidenceThreshold: 0.85,
    maxRetries: 3,
    timeout: 300,
  },
  camera: {
    resolution: '224x224',
    fps: 30,
    autoFocus: true,
    brightness: 50,
  },
  ...overrides,
});

export default {
  renderWithProviders,
  createMockUser,
  mockLocalStorage,
  mockApiResponse,
  mockApiError,
  waitForElement,
  createMockSystemStatus,
  createMockProcessingStats,
  createMockDashboardData,
  createMockResults,
  createMockPipelineStatus,
  createMockCameraData,
  simulateTyping,
  createMockSettings,
};
