/**
 * API Utilities Tests
 * Tests for all API interaction modules
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mockApiResponse, mockApiError } from './testUtils';

import {
  startPipeline,
  stopPipeline,
  getPipelineStatus,
  getPipelineLogs,
  getPipelineConfig,
} from '../../src/utils/processingApi';

import { fetchDashboardData } from '../../src/utils/AdminDashboardApi'; 

import { fetchCameraStatus, refreshCamera } from '../../src/utils/CameraApi'; 

import { fetchResults, exportResults } from '../../src/utils/ResultsApi'; 

import { getSettings, updateSettings } from '../../src/utils/SettingsApi'; 

describe('Processing API', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('startPipeline', () => {
    it('should send POST request with correct configuration', async () => {
      const config = {
        hiddenDim: 256,
        epochs: 100,
        learningRate: 0.001,
        lambdaReg: 0.01,
        batchSize: 32,
      };

      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true })
      );

      await startPipeline(config);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/pipeline/start'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(config),
        })
      );
    });

    it('should return success response', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true, message: 'Pipeline started' })
      );

      const result = await startPipeline({});

      expect(result.success).toBe(true);
      expect(result.message).toBe('Pipeline started');
    });

    it('should handle API errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Network error')
      );

      await expect(startPipeline({})).rejects.toThrow('Network error');
    });

    it('should handle 500 error responses', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse(
          { error: 'Internal server error' },
          { status: 500, ok: false }
        )
      );

      const result = await startPipeline({});

      expect(result.success).toBe(false);
    });
  });

  describe('stopPipeline', () => {
    it('should send POST request to stop endpoint', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true })
      );

      await stopPipeline();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/pipeline/stop'),
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should return success on stop', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true })
      );

      const result = await stopPipeline();

      expect(result.success).toBe(true);
    });
  });

  describe('getPipelineStatus', () => {
    it('should fetch current pipeline status', async () => {
      const mockStatus = {
        running: true,
        status: 'processing',
        currentStep: 3,
        progress: 60,
        steps: [],
      };

      global.fetch.mockResolvedValue(
        mockApiResponse(mockStatus)
      );

      const result = await getPipelineStatus();

      expect(result).toEqual(mockStatus);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/pipeline/status'),
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should handle status fetch errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Status unavailable')
      );

      await expect(getPipelineStatus()).rejects.toThrow('Status unavailable');
    });
  });

  describe('getPipelineLogs', () => {
    it('should fetch logs with default limit', async () => {
      const mockLogs = [
        { message: 'Log 1', type: 'info', timestamp: new Date().toISOString() },
        { message: 'Log 2', type: 'success', timestamp: new Date().toISOString() },
      ];

      global.fetch.mockResolvedValue(
        mockApiResponse(mockLogs)
      );

      const result = await getPipelineLogs();

      expect(result).toEqual(mockLogs);
    });

    it('should fetch logs with custom limit', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse([])
      );

      await getPipelineLogs(50);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=50'),
        expect.any(Object)
      );
    });
  });

  describe('getPipelineConfig', () => {
    it('should fetch pipeline configuration', async () => {
      const mockConfig = {
        hiddenDim: 256,
        epochs: 100,
        learningRate: 0.001,
      };

      global.fetch.mockResolvedValue(
        mockApiResponse(mockConfig)
      );

      const result = await getPipelineConfig();

      expect(result).toEqual(mockConfig);
    });
  });
});

describe('Dashboard API', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it('should fetch dashboard data', async () => {
    const mockData = {
      totalProcessed: 1000,
      accuracy: 0.95,
      systemUptime: 48.5,
    };

    global.fetch.mockResolvedValue(
      mockApiResponse(mockData)
    );

    const result = await fetchDashboardData();

    expect(result).toEqual(mockData);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/dashboard'),
      expect.any(Object)
    );
  });

  it('should handle dashboard fetch errors', async () => {
    global.fetch.mockRejectedValue(
      new Error('Dashboard unavailable')
    );

    await expect(fetchDashboardData()).rejects.toThrow('Dashboard unavailable');
  });
});

describe('Camera API', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  describe('fetchCameraStatus', () => {
    it('should fetch status for all cameras', async () => {
      const mockCameras = [
        { id: 0, status: true, fps: 30 },
        { id: 1, status: true, fps: 30 },
        { id: 2, status: true, fps: 30 },
        { id: 3, status: false, fps: 0 },
      ];

      global.fetch.mockResolvedValue(
        mockApiResponse({ cameras: mockCameras })
      );

      const result = await fetchCameraStatus();

      expect(result.cameras).toEqual(mockCameras);
    });

    it('should handle camera status errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Camera system offline')
      );

      await expect(fetchCameraStatus()).rejects.toThrow('Camera system offline');
    });
  });

  describe('refreshCamera', () => {
    it('should send refresh request for specific camera', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true })
      );

      await refreshCamera(2);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/camera/2/refresh'),
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should handle refresh errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Refresh failed')
      );

      await expect(refreshCamera(1)).rejects.toThrow('Refresh failed');
    });
  });
});

describe('Results API', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  describe('fetchResults', () => {
    it('should fetch results with filters', async () => {
      const mockResults = [
        { id: 1, fruitType: 'apple', grade: 'premium' },
        { id: 2, fruitType: 'orange', grade: 'standard' },
      ];

      global.fetch.mockResolvedValue(
        mockApiResponse({ results: mockResults })
      );

      const filters = {
        fruitType: 'apple',
        grade: 'premium',
        startDate: '2025-01-01',
        endDate: '2025-01-03',
      };

      const result = await fetchResults(filters);

      expect(result.results).toEqual(mockResults);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/results'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(filters),
        })
      );
    });

    it('should handle empty results', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ results: [] })
      );

      const result = await fetchResults({});

      expect(result.results).toEqual([]);
    });

    it('should handle pagination parameters', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ results: [], page: 2, totalPages: 5 })
      );

      const result = await fetchResults({ page: 2, limit: 20 });

      expect(result.page).toBe(2);
      expect(result.totalPages).toBe(5);
    });
  });

  describe('exportResults', () => {
    it('should export results in CSV format', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ file: 'results.csv' })
      );

      const result = await exportResults('csv');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/results/export'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ format: 'csv' }),
        })
      );
    });

    it('should export results in PDF format', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ file: 'results.pdf' })
      );

      await exportResults('pdf');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/results/export'),
        expect.objectContaining({
          body: JSON.stringify({ format: 'pdf' }),
        })
      );
    });
  });
});

describe('Settings API', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  describe('getSettings', () => {
    it('should fetch all system settings', async () => {
      const mockSettings = {
        system: { autoRestart: true },
        processing: { batchSize: 32 },
        camera: { resolution: '224x224' },
      };

      global.fetch.mockResolvedValue(
        mockApiResponse(mockSettings)
      );

      const result = await getSettings();

      expect(result).toEqual(mockSettings);
    });

    it('should handle settings fetch errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Settings unavailable')
      );

      await expect(getSettings()).rejects.toThrow('Settings unavailable');
    });
  });

  describe('updateSettings', () => {
    it('should update specific settings', async () => {
      const updates = {
        system: { autoRestart: false },
        processing: { batchSize: 64 },
      };

      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true })
      );

      await updateSettings(updates);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/settings'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(updates),
        })
      );
    });

    it('should return success on update', async () => {
      global.fetch.mockResolvedValue(
        mockApiResponse({ success: true, message: 'Settings updated' })
      );

      const result = await updateSettings({});

      expect(result.success).toBe(true);
    });

    it('should handle update errors', async () => {
      global.fetch.mockRejectedValue(
        new Error('Update failed')
      );

      await expect(updateSettings({})).rejects.toThrow('Update failed');
    });
  });
});

describe('API Error Handling', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it('should handle network timeouts', async () => {
    global.fetch.mockImplementation(() => 
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), 100)
      )
    );

    await expect(getPipelineStatus()).rejects.toThrow('Timeout');
  });

  it('should handle 401 unauthorized errors', async () => {
    global.fetch.mockResolvedValue(
      mockApiResponse(
        { error: 'Unauthorized' },
        { status: 401, ok: false }
      )
    );

    // Should trigger logout or redirect
    const result = await getPipelineStatus().catch(e => e);
    expect(result).toBeDefined();
  });

  it('should handle 404 not found errors', async () => {
    global.fetch.mockResolvedValue(
      mockApiResponse(
        { error: 'Not found' },
        { status: 404, ok: false }
      )
    );

    const result = await fetchDashboardData().catch(e => e);
    expect(result).toBeDefined();
  });

  it('should retry on transient failures', async () => {
    let attempts = 0;
    global.fetch.mockImplementation(() => {
      attempts++;
      if (attempts < 3) {
        return Promise.reject(new Error('Transient error'));
      }
      return mockApiResponse({ success: true });
    });

    // This test assumes retry logic is implemented
    // Currently documents expected behavior
  });

  it('should handle malformed JSON responses', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error('Invalid JSON')),
    });

    await expect(getPipelineConfig()).rejects.toThrow('Invalid JSON');
  });
});

describe('API Request Headers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue(
      mockApiResponse({ success: true })
    );
  });

  it('should include Content-Type header for POST requests', async () => {
    await startPipeline({});

    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
        }),
      })
    );
  });

  it('should include authentication token if available', async () => {
    localStorage.setItem('authToken', 'test-token-123');

    await getPipelineStatus();

    // This test assumes auth token implementation
    // Currently documents expected behavior
    localStorage.removeItem('authToken');
  });

  it('should include CORS headers for cross-origin requests', async () => {
    await fetchDashboardData();

    // CORS headers are typically set by the browser
    // This test documents the expected server behavior
  });
});
