/**
 * Vitest Configuration
 * Testing framework setup for React frontend
 */

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    // Test environment
    environment: 'jsdom',
    
    // Setup files - CHANGED: removed frontend-tests/
    setupFiles: ['./setup/testSetup.js'],
    
    // Global test configuration
    globals: true,
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'Tests/',                    
        '*.config.js',
        '../src/main.jsx',           
        '../src/index.jsx',          
      ],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
    
    // Test file patterns
    include: [
      './**/*.test.{js,jsx}',
      './**/*.spec.{js,jsx}',
    ],
    
    // Timeout configuration
    testTimeout: 10000,
    hookTimeout: 10000,
    
    // Reporter configuration
    reporters: ['verbose', 'json', 'html'],
    
    // Mock reset configuration
    clearMocks: true,
    restoreMocks: true,
    resetMocks: true,
    
    // Parallel execution
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: false,
        maxThreads: 4,
        minThreads: 1,
      },
    },
    
    // Watch mode configuration
    watch: false,
    
    // Snapshot configuration
    snapshotFormat: {
      printBasicPrototype: false,
    },
  },
  
  // Resolve aliases to match Vite config 
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../src'),
      '@components': path.resolve(__dirname, '../src/components'),
      '@pages': path.resolve(__dirname, '../src/pages'),
      '@utils': path.resolve(__dirname, '../src/utils'),
      '@context': path.resolve(__dirname, '../src/context'),
    },
  },
});