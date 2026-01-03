/**
 * Integration Tests
 * End-to-end user workflow testing
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../../src/App';
import { renderWithProviders } from '../utils/testUtils';
import { 
  createMockDashboardData,
  createMockPipelineStatus,
} from '../utils/testUtils';

// Mock all API modules
vi.mock('../../src/utils/processingApi');
vi.mock('../../src/utils/AdminDashboardApi');
vi.mock('../../src/utils/CameraApi');
vi.mock('../../src/utils/ResultsApi');
vi.mock('../../src/utils/SettingsApi');

describe('User Workflows - Integration Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  describe('Admin User Workflow', () => {
    it('should complete full admin login and navigation flow', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Step 1: Start at login page
      await waitFor(() => {
        expect(screen.getByText('Fruit Grading System')).toBeInTheDocument();
      });

      // Step 2: Select admin role
      const adminButton = screen.getByText('Admin').closest('button');
      await user.click(adminButton);

      // Step 3: Enter credentials
      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123');

      // Step 4: Submit login
      const loginButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(loginButton);

      // Step 5: Verify redirect to dashboard
      await waitFor(() => {
        expect(window.location.pathname).toBe('/dashboard');
      });

      // Step 6: Verify admin dashboard loads
      await waitFor(() => {
        expect(screen.getByText('System Dashboard')).toBeInTheDocument();
      });
    });

    it('should navigate between all admin pages', async () => {
      const user = userEvent.setup();
      
      // Login first
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      // Navigate to Camera Monitor
      const cameraLink = screen.getByText('Cameras');
      await user.click(cameraLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/cameras');
      });

      // Navigate to Processing
      const processingLink = screen.getByText('Processing');
      await user.click(processingLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/processing');
      });

      // Navigate to Results
      const resultsLink = screen.getByText('Results');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/results');
      });

      // Navigate to Settings
      const settingsLink = screen.getByText('Settings');
      await user.click(settingsLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/settings');
      });
    });

    it('should complete processing pipeline workflow', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const processingApi = await import('../../src/utils/processingApi');
      processingApi.getPipelineConfig.mockResolvedValue({
        hiddenDim: 256,
        epochs: 100,
        learningRate: 0.001,
      });
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus()
      );
      processingApi.startPipeline.mockResolvedValue({ success: true });

      renderWithProviders(<App />);

      // Navigate to processing page
      window.history.pushState({}, 'Test', '/processing');

      await waitFor(() => {
        expect(screen.getByText('Processing Pipeline')).toBeInTheDocument();
      });

      // Start pipeline
      const startButton = screen.getByRole('button', { name: /start pipeline/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(processingApi.startPipeline).toHaveBeenCalled();
      });

      // Verify pipeline started
      expect(processingApi.startPipeline).toHaveBeenCalledWith(
        expect.objectContaining({
          hiddenDim: expect.any(Number),
          epochs: expect.any(Number),
        })
      );
    });

    it('should logout and return to login page', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      // Click logout
      const logoutButton = screen.getByText('Logout');
      await user.click(logoutButton);

      // Verify redirect to login
      await waitFor(() => {
        expect(screen.getByText('Sign in to continue')).toBeInTheDocument();
      });

      // Verify localStorage cleared
      expect(localStorage.getItem('fruitGradingUser')).toBeNull();
    });
  });

  describe('Regular User Workflow', () => {
    it('should complete full user login and navigation flow', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Login as regular user
      await waitFor(() => {
        expect(screen.getByText('Fruit Grading System')).toBeInTheDocument();
      });

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'user');
      await user.type(passwordInput, 'user123');

      const loginButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(loginButton);

      // Verify redirect to user dashboard
      await waitFor(() => {
        expect(window.location.pathname).toBe('/user-dashboard');
      });
    });

    it('should restrict access to admin-only pages', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'user',
        role: 'user',
        loginTime: new Date().toISOString(),
      }));

      renderWithProviders(<App />);

      // Try to navigate to admin dashboard
      window.history.pushState({}, 'Test', '/dashboard');

      await waitFor(() => {
        // Should redirect or show access denied
        expect(window.location.pathname).not.toBe('/dashboard');
      });
    });

    it('should allow access to results page', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'user',
        role: 'user',
        loginTime: new Date().toISOString(),
      }));

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(screen.getByText('Results')).toBeInTheDocument();
      });

      const resultsLink = screen.getByText('Results');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/results');
      });
    });
  });

  describe('Data Flow Integration', () => {
    it('should update dashboard stats after processing completion', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const dashboardApi = await import('../../src/utils/AdminDashboardApi');
      const processingApi = await import('../../src/utils/processingApi');

      // Initial dashboard data
      dashboardApi.fetchDashboardData.mockResolvedValue(
        createMockDashboardData({ totalProcessed: 100, accuracy: 0.90 })
      );

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(screen.getByText('100')).toBeInTheDocument(); // Total processed
      });

      // Simulate processing completion
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({
          status: 'completed',
          totalProcessed: 150,
          accuracy: 0.95,
        })
      );

      // Dashboard should refresh and show updated stats
      await waitFor(() => {
        expect(screen.getByText('150')).toBeInTheDocument();
      });
    });

    it('should reflect settings changes across components', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const settingsApi = await import('../../src/utils/SettingsApi');
      settingsApi.getSettings.mockResolvedValue({
        processing: { batchSize: 32 },
      });
      settingsApi.updateSettings.mockResolvedValue({ success: true });

      renderWithProviders(<App />);

      // Navigate to settings
      window.history.pushState({}, 'Test', '/settings');

      await waitFor(() => {
        expect(screen.getByText('Settings')).toBeInTheDocument();
      });

      // Update batch size
      const batchSizeInput = screen.getByLabelText(/batch size/i);
      await user.clear(batchSizeInput);
      await user.type(batchSizeInput, '64');

      const saveButton = screen.getByRole('button', { name: /save/i });
      await user.click(saveButton);

      await waitFor(() => {
        expect(settingsApi.updateSettings).toHaveBeenCalledWith(
          expect.objectContaining({
            processing: expect.objectContaining({ batchSize: 64 }),
          })
        );
      });

      // Navigate to processing page
      window.history.pushState({}, 'Test', '/processing');

      // Verify new batch size is reflected
      await waitFor(() => {
        expect(screen.getByText('64')).toBeInTheDocument();
      });
    });
  });

  describe('Error Recovery Flows', () => {
    it('should handle API failure and allow retry', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const dashboardApi = await import('../../src/utils/AdminDashboardApi');
      
      // First call fails
      dashboardApi.fetchDashboardData.mockRejectedValueOnce(
        new Error('Network error')
      );

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(screen.getByText(/error/i)).toBeInTheDocument();
      });

      // Mock successful retry
      dashboardApi.fetchDashboardData.mockResolvedValue(
        createMockDashboardData()
      );

      const retryButton = screen.getByRole('button', { name: /retry/i });
      await user.click(retryButton);

      await waitFor(() => {
        expect(screen.queryByText(/error/i)).not.toBeInTheDocument();
      });
    });

    it('should handle session expiration and redirect to login', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const dashboardApi = await import('../../src/utils/AdminDashboardApi');
      
      // Mock 401 unauthorized
      dashboardApi.fetchDashboardData.mockRejectedValue({
        status: 401,
        message: 'Unauthorized',
      });

      renderWithProviders(<App />);

      await waitFor(() => {
        // Should redirect to login
        expect(window.location.pathname).toBe('/login');
      });

      // localStorage should be cleared
      expect(localStorage.getItem('fruitGradingUser')).toBeNull();
    });

    it('should maintain state during network interruption', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const processingApi = await import('../../src/utils/processingApi');
      
      // Pipeline running
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, progress: 50 })
      );

      renderWithProviders(<App />);

      window.history.pushState({}, 'Test', '/processing');

      await waitFor(() => {
        expect(screen.getByText('50%')).toBeInTheDocument();
      });

      // Network interruption
      processingApi.getPipelineStatus.mockRejectedValueOnce(
        new Error('Network timeout')
      );

      // Wait for retry interval
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Network restored
      processingApi.getPipelineStatus.mockResolvedValue(
        createMockPipelineStatus({ running: true, progress: 70 })
      );

      // Should eventually recover and show updated progress
      await waitFor(() => {
        expect(screen.getByText('70%')).toBeInTheDocument();
      }, { timeout: 5000 });
    });
  });

  describe('Performance and Optimization', () => {
    it('should not make redundant API calls', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const dashboardApi = await import('../../src/utils/AdminDashboardApi');
      dashboardApi.fetchDashboardData.mockResolvedValue(
        createMockDashboardData()
      );

      renderWithProviders(<App />);

      await waitFor(() => {
        expect(dashboardApi.fetchDashboardData).toHaveBeenCalledTimes(1);
      });

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Should not have made additional calls
      expect(dashboardApi.fetchDashboardData).toHaveBeenCalledTimes(1);
    });

    it('should debounce search inputs', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const resultsApi = await import('../../src/utils/ResultsApi');
      resultsApi.fetchResults.mockResolvedValue({ results: [] });

      renderWithProviders(<App />);

      window.history.pushState({}, 'Test', '/results');

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search/i);
      
      // Type quickly
      await user.type(searchInput, 'apple');

      // Should only make one API call after debounce
      await waitFor(() => {
        expect(resultsApi.fetchResults).toHaveBeenCalledTimes(1);
      }, { timeout: 1000 });
    });

    it('should implement virtual scrolling for large result sets', async () => {
      localStorage.setItem('fruitGradingUser', JSON.stringify({
        username: 'admin',
        role: 'admin',
        loginTime: new Date().toISOString(),
      }));

      const resultsApi = await import('../../src/utils/ResultsApi');
      
      // Mock 1000 results
      const largeResultSet = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        fruitType: 'apple',
        grade: 'premium',
      }));

      resultsApi.fetchResults.mockResolvedValue({ results: largeResultSet });

      renderWithProviders(<App />);

      window.history.pushState({}, 'Test', '/results');

      // Only visible rows should be renderWithProvidersed
      await waitFor(() => {
        const renderWithProvidersedRows = screen.getAllByRole('row');
        expect(renderWithProvidersedRows.length).toBeLessThan(1000);
      });
    });
  });
});
