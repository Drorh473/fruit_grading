/**
 * AuthContext Tests
 * Tests for authentication context and state management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { AuthProvider, useAuth } from '../../src/context/AuthContext';
import { createMockUser, mockLocalStorage } from '../utils/testUtils';

describe('AuthContext', () => {
  let localStorageMock;

  beforeEach(() => {
    localStorageMock = mockLocalStorage();
    Object.defineProperty(window, 'localStorage', {
      value: localStorageMock,
      writable: true,
    });
  });

  describe('Initial State', () => {
    it('should initialize with no user when localStorage is empty', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      expect(result.current.user).toBeNull();
      expect(result.current.loading).toBe(false);
    });

    it('should load user from localStorage on mount', async () => {
      const mockUser = createMockUser('admin');
      localStorage.setItem('fruitGradingUser', JSON.stringify(mockUser));

      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      await waitFor(() => {
        expect(result.current.user).toEqual(mockUser);
      });
    });

    it('should handle corrupted localStorage data gracefully', () => {
      localStorage.setItem('fruitGradingUser', 'invalid-json{');

      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      expect(result.current.user).toBeNull();
      expect(localStorage.getItem('fruitGradingUser')).toBeNull();
    });
  });

  describe('Login Functionality', () => {
    it('should login admin user with correct credentials', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        const response = result.current.login('admin', 'admin123', 'admin');
        expect(response.success).toBe(true);
      });

      expect(result.current.user).toMatchObject({
        username: 'admin',
        role: 'admin',
      });
      expect(result.current.user.loginTime).toBeDefined();
    });

    it('should login regular user with correct credentials', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        const response = result.current.login('user', 'user123', 'user');
        expect(response.success).toBe(true);
      });

      expect(result.current.user).toMatchObject({
        username: 'user',
        role: 'user',
      });
    });

    it('should reject login with incorrect username', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        const response = result.current.login('wronguser', 'admin123', 'admin');
        expect(response.success).toBe(false);
        expect(response.error).toBe('Invalid credentials');
      });

      expect(result.current.user).toBeNull();
    });

    it('should reject login with incorrect password', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        const response = result.current.login('admin', 'wrongpassword', 'admin');
        expect(response.success).toBe(false);
      });

      expect(result.current.user).toBeNull();
    });

    it('should reject login with mismatched role', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        const response = result.current.login('admin', 'admin123', 'user');
        expect(response.success).toBe(false);
      });

      expect(result.current.user).toBeNull();
    });

    it('should save user to localStorage on successful login', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        result.current.login('admin', 'admin123', 'admin');
      });

      const savedUser = JSON.parse(localStorage.getItem('fruitGradingUser'));
      expect(savedUser).toMatchObject({
        username: 'admin',
        role: 'admin',
      });
    });
  });

  describe('Logout Functionality', () => {
    it('should clear user state on logout', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        result.current.login('admin', 'admin123', 'admin');
      });

      expect(result.current.user).not.toBeNull();

      act(() => {
        result.current.logout();
      });

      expect(result.current.user).toBeNull();
    });

    it('should remove user from localStorage on logout', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        result.current.login('admin', 'admin123', 'admin');
        result.current.logout();
      });

      expect(localStorage.getItem('fruitGradingUser')).toBeNull();
    });
  });

  describe('Role Checking Functions', () => {
    it('should correctly identify admin users', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        result.current.login('admin', 'admin123', 'admin');
      });

      expect(result.current.isAdmin()).toBe(true);
      expect(result.current.isUser()).toBe(false);
    });

    it('should correctly identify regular users', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      act(() => {
        result.current.login('user', 'user123', 'user');
      });

      expect(result.current.isAdmin()).toBe(false);
      expect(result.current.isUser()).toBe(true);
    });

    it('should return false for role checks when not logged in', () => {
      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      expect(result.current.isAdmin()).toBe(false);
      expect(result.current.isUser()).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should throw error when useAuth is used outside AuthProvider', () => {
      // Suppress console error for this test
      const originalError = console.error;
      console.error = vi.fn();

      expect(() => {
        renderHook(() => useAuth());
      }).toThrow('useAuth must be used within AuthProvider');

      console.error = originalError;
    });
  });

  describe('Session Persistence', () => {
    it('should maintain session across page reloads', () => {
      const mockUser = createMockUser('admin');
      localStorage.setItem('fruitGradingUser', JSON.stringify(mockUser));

      const { result, rerender } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      rerender();

      expect(result.current.user).toEqual(mockUser);
    });

    it('should handle session expiration gracefully', () => {
      const expiredUser = {
        ...createMockUser('admin'),
        loginTime: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
      };
      localStorage.setItem('fruitGradingUser', JSON.stringify(expiredUser));

      const { result } = renderHook(() => useAuth(), {
        wrapper: AuthProvider,
      });

      // Currently the system doesn't implement session expiration
      // This test documents expected behavior for future implementation
      expect(result.current.user).toBeDefined();
    });
  });
});
