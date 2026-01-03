/**
 * Frontend Test Suite Setup
 * Configures testing environment for React components
 */

import { cleanup } from '@testing-library/react';
import { afterEach, beforeAll, afterAll , vi} from 'vitest';

// Cleanup after each test
afterEach(() => {
  cleanup();
  localStorage.clear();
  sessionStorage.clear();
});

// Mock window.matchMedia (required for responsive components)
beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => {},
    }),
  });

  // Mock IntersectionObserver
  global.IntersectionObserver = class IntersectionObserver {
    constructor() {}
    disconnect() {}
    observe() {}
    takeRecords() {
      return [];
    }
    unobserve() {}
  };

  // Mock console methods to reduce noise in tests
  global.console = {
    ...console,
    error: vi.fn(),
    warn: vi.fn(),
  };
});

afterAll(() => {
  vi.restoreAllMocks();
});

