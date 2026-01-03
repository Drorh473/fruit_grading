/**
 * Processing Component Tests
 * Tests for ML pipeline processing page
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {  screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Processing from '../../src/pages/Processing';
import * as processingApi from '../../src/utils/processingApi';
import { 
  createMockPipelineStatus, 
  mockApiResponse,
  mockApiError 
} from '../utils/testUtils';

import { renderWithProviders } from '../utils/testUtils';

// Mock the API module
vi.mock('../../src/utils/processingApi');

describe('Processing Component', () => {
  const mockSetProcessingStats = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Default API mocks
    processingApi.getPipelineConfig.mockResolvedValue({
      hiddenDim: 256,
      epochs: 100,
      learningRate: 0.001,
      lambdaReg: 0.01,
      batchSize: 32,
    });

    processingApi.getPipelineStatus.mockResolvedValue(
      createMockPipelineStatus()
    );

    processingApi.getPipelineLogs.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Initial renderWithProvidersing', () => {
    it('should renderWithProviders processing page title and subtitle', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('Processing Pipeline')).toBeInTheDocument();
        expect(screen.getByText('Run the complete ML pipeline from data to model')).toBeInTheDocument();
      });
    });

    it('should show loading spinner initially', () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      expect(screen.getByRole('status', { hidden: true })).toBeInTheDocument();
    });

    it('should load initial configuration on mount', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(processingApi.getPipelineConfig).toHaveBeenCalledTimes(1);
      });
    });

    it('should check pipeline status on mount', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(processingApi.getPipelineStatus).toHaveBeenCalled();
      });
    });
  });

  describe('Pipeline Controls', () => {
    it('should renderWithProviders Start Pipeline button when idle', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /start pipeline/i })).toBeInTheDocument();
      });
    });

    it('should renderWithProviders Stop button when processing', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, status: 'running' })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
      });
    });

    it('should start pipeline when Start button clicked', async () => {
      const user = userEvent.setup();
      processingApi.startPipeline.mockResolvedValue({ success: true });

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /start pipeline/i })).toBeInTheDocument();
      });

      const startButton = screen.getByRole('button', { name: /start pipeline/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(processingApi.startPipeline).toHaveBeenCalledWith(
          expect.objectContaining({
            skipTests: true,
            hiddenDim: expect.any(Number),
            epochs: expect.any(Number),
          })
        );
      });
    });

    it('should stop pipeline when Stop button clicked', async () => {
      const user = userEvent.setup();
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, status: 'running' })
      );
      processingApi.stopPipeline.mockResolvedValue({ success: true });

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
      });

      const stopButton = screen.getByRole('button', { name: /stop/i });
      await user.click(stopButton);

      await waitFor(() => {
        expect(processingApi.stopPipeline).toHaveBeenCalled();
      });
    });

    it('should show error message when start fails', async () => {
      const user = userEvent.setup();
      processingApi.startPipeline.mockResolvedValue({ 
        success: false, 
        message: 'Database connection failed' 
      });

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const startButton = screen.getByRole('button', { name: /start pipeline/i });
        expect(startButton).toBeInTheDocument();
      });

      const startButton = screen.getByRole('button', { name: /start pipeline/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(screen.getByText(/database connection failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Progress Display', () => {
    it('should display progress percentage', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ progress: 45 })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('45%')).toBeInTheDocument();
      });
    });

    it('should update progress bar width based on percentage', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ progress: 60 })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const progressFill = document.querySelector('.progress-fill');
        expect(progressFill).toHaveStyle({ width: '60%' });
      });
    });

    it('should show 100% only when status is completed', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          status: 'completed', 
          progress: 100,
          steps: [
            { id: 1, name: 'Database Setup', status: 'completed' },
            { id: 2, name: 'Data Preprocessing', status: 'completed' },
            { id: 3, name: 'Feature Extraction', status: 'completed' },
            { id: 4, name: 'Model Training', status: 'completed' },
            { id: 5, name: 'Evaluation', status: 'completed' },
          ]
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('100%')).toBeInTheDocument();
      });
    });

    it('should display current step information', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          running: true,
          currentStep: 3,
          status: 'processing' 
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText(/processing step 3 of 5/i)).toBeInTheDocument();
      });
    });
  });

  describe('Pipeline Steps', () => {
    it('should renderWithProviders all 5 pipeline steps', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('Database Setup')).toBeInTheDocument();
        expect(screen.getByText('Data Preprocessing')).toBeInTheDocument();
        expect(screen.getByText('Feature Extraction')).toBeInTheDocument();
        expect(screen.getByText('Model Training')).toBeInTheDocument();
        expect(screen.getByText('Evaluation')).toBeInTheDocument();
      });
    });

    it('should show pending status for unstarted steps', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const steps = screen.getAllByText('pending');
        expect(steps).toHaveLength(5);
      });
    });

    it('should highlight processing step', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          steps: [
            { id: 1, name: 'Database Setup', status: 'completed' },
            { id: 2, name: 'Data Preprocessing', status: 'processing' },
            { id: 3, name: 'Feature Extraction', status: 'pending' },
            { id: 4, name: 'Model Training', status: 'pending' },
            { id: 5, name: 'Evaluation', status: 'pending' },
          ]
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const processingStep = screen.getByText('Data Preprocessing').closest('.step-item');
        expect(processingStep).toHaveClass('step-processing');
      });
    });

    it('should mark completed steps with success styling', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          steps: [
            { id: 1, name: 'Database Setup', status: 'completed' },
            { id: 2, name: 'Data Preprocessing', status: 'pending' },
            { id: 3, name: 'Feature Extraction', status: 'pending' },
            { id: 4, name: 'Model Training', status: 'pending' },
            { id: 5, name: 'Evaluation', status: 'pending' },
          ]
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const completedStep = screen.getByText('Database Setup').closest('.step-item');
        expect(completedStep).toHaveClass('step-completed');
      });
    });

    it('should mark failed steps with error styling', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          steps: [
            { id: 1, name: 'Database Setup', status: 'failed' },
            { id: 2, name: 'Data Preprocessing', status: 'pending' },
            { id: 3, name: 'Feature Extraction', status: 'pending' },
            { id: 4, name: 'Model Training', status: 'pending' },
            { id: 5, name: 'Evaluation', status: 'pending' },
          ]
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const failedStep = screen.getByText('Database Setup').closest('.step-item');
        expect(failedStep).toHaveClass('step-failed');
      });
    });
  });

  describe('Configuration Management', () => {
    it('should display current configuration values', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('256')).toBeInTheDocument(); // hiddenDim
        expect(screen.getByText('100')).toBeInTheDocument(); // epochs
      });
    });

    it('should allow changing hidden dimension', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('256')).toBeInTheDocument();
      });

      // Click dropdown to open
      const hiddenDimButton = screen.getByText('256').closest('button');
      await user.click(hiddenDimButton);

      // Select different value
      const option512 = screen.getByText('512');
      await user.click(option512);

      // Verify selection
      await waitFor(() => {
        expect(screen.getByText('512')).toBeInTheDocument();
      });
    });

    it('should disable configuration changes while processing', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, status: 'running' })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const configButton = screen.getAllByRole('button')[2]; // First config dropdown
        expect(configButton).toBeDisabled();
      });
    });
  });

  describe('Logs Display', () => {
    it('should display processing logs', async () => {
      const mockLogs = [
        { message: 'Pipeline started', type: 'info', timestamp: new Date().toISOString() },
        { message: 'Database connected', type: 'success', timestamp: new Date().toISOString() },
      ];

      processingApi.getPipelineLogs.mockResolvedValue(mockLogs);
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('Pipeline started')).toBeInTheDocument();
        expect(screen.getByText('Database connected')).toBeInTheDocument();
      });
    });

    it('should show empty state when no logs', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('No logs yet.')).toBeInTheDocument();
      });
    });

    it('should allow clearing logs', async () => {
      const user = userEvent.setup();
      const mockLogs = [
        { message: 'Test log', type: 'info', timestamp: new Date().toISOString() },
      ];

      processingApi.getPipelineLogs.mockResolvedValue(mockLogs);

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText('Test log')).toBeInTheDocument();
      });

      const clearButton = screen.getByRole('button', { name: /clear logs/i });
      await user.click(clearButton);

      await waitFor(() => {
        expect(screen.queryByText('Test log')).not.toBeInTheDocument();
      });
    });

    it('should disable clear button when no logs', async () => {
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const clearButton = screen.getByRole('button', { name: /clear logs/i });
        expect(clearButton).toBeDisabled();
      });
    });
  });

  describe('Status Polling', () => {
    it('should poll status every 2 seconds while processing', async () => {
      vi.useFakeTimers();

      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, status: 'running' })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(processingApi.getPipelineStatus).toHaveBeenCalled();
      });

      const initialCallCount = processingApi.getPipelineStatus.mock.calls.length;

      // Advance time by 2 seconds
      vi.advanceTimersByTime(2000);

      await waitFor(() => {
        expect(processingApi.getPipelineStatus).toHaveBeenCalledTimes(initialCallCount + 1);
      });

      vi.useRealTimers();
    });

    it('should stop polling when processing completes', async () => {
      vi.useFakeTimers();

      let callCount = 0;
      processingApi.getPipelineStatus.mockImplementation(() => {
        callCount++;
        if (callCount > 2) {
          return Promise.resolve(createMockPipelineStatus({ 
            running: false, 
            status: 'completed' 
          }));
        }
        return Promise.resolve(createMockPipelineStatus({ 
          running: true, 
          status: 'running' 
        }));
      });

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      // Wait for initial load
      await waitFor(() => {
        expect(processingApi.getPipelineStatus).toHaveBeenCalled();
      });

      // Advance through several polling intervals
      vi.advanceTimersByTime(10000);

      // Should eventually stop polling after completion
      const finalCallCount = processingApi.getPipelineStatus.mock.calls.length;
      
      vi.advanceTimersByTime(5000);
      
      // No new calls after completion
      expect(processingApi.getPipelineStatus).toHaveBeenCalledTimes(finalCallCount);

      vi.useRealTimers();
    });
  });

  describe('Completion Handling', () => {
    it('should update processing stats on successful completion', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          running: false,
          status: 'completed',
          totalProcessed: 200,
          accuracy: 0.95,
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      // Trigger status update
      await waitFor(() => {
        expect(mockSetProcessingStats).toHaveBeenCalledWith(
          expect.objectContaining({
            totalProcessed: 200,
            accuracy: 0.95,
          })
        );
      });
    });

    it('should show error message on pipeline failure', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ 
          running: false,
          status: 'failed',
        })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText(/pipeline failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle config load failure gracefully', async () => {
      processingApi.getPipelineConfig.mockRejectedValue(
        new Error('Network error')
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByText(/failed to load configuration/i)).toBeInTheDocument();
      });
    });

    it('should handle status check failure gracefully', async () => {
      processingApi.getPipelineStatus.mockRejectedValue(
        new Error('Status check failed')
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      // Should still renderWithProviders the page
      await waitFor(() => {
        expect(screen.getByText('Processing Pipeline')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('should reload data when refresh button clicked', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
      });

      const initialCalls = processingApi.getPipelineConfig.mock.calls.length;

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      await waitFor(() => {
        expect(processingApi.getPipelineConfig).toHaveBeenCalledTimes(initialCalls + 1);
      });
    });

    it('should disable refresh while processing', async () => {
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true })
      );

      renderWithProviders(<Processing setProcessingStats={mockSetProcessingStats} />);

      await waitFor(() => {
        const refreshButton = screen.getByRole('button', { name: /refresh/i });
        expect(refreshButton).toBeDisabled();
      });
    });
  });
});
