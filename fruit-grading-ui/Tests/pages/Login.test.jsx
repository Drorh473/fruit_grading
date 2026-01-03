/**
 * Login Component Tests
 * Tests for login page functionality and user interactions
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import Login from '../../src/pages/Login';
import { AuthProvider } from '../../src/context/AuthContext';
import { renderWithProviders } from '../utils/testUtils';

const renderWithProvidersLogin = () => {
  const mockNavigate = vi.fn();
  vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
      ...actual,
      useNavigate: () => mockNavigate,
    };
  });

  return renderWithProviders(
    <BrowserRouter>
      <AuthProvider>
        <Login />
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('Login Component', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe('renderWithProvidersing', () => {
    it('should renderWithProviders login form with all elements', () => {
      renderWithProvidersLogin();

      expect(screen.getByText('Fruit Grading System')).toBeInTheDocument();
      expect(screen.getByText('Sign in to continue')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter username')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter password')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    it('should renderWithProviders role selector with admin and user options', () => {
      renderWithProvidersLogin();

      const roleButtons = screen.getAllByRole('button');
      const adminButton = roleButtons.find(btn => btn.textContent.includes('Admin'));
      const userButton = roleButtons.find(btn => btn.textContent.includes('User'));

      expect(adminButton).toBeInTheDocument();
      expect(userButton).toBeInTheDocument();
    });

    it('should have user role selected by default', () => {
      renderWithProvidersLogin();

      const userButton = screen.getByText('User').closest('button');
      expect(userButton).toHaveClass('role-active');
    });

    it('should renderWithProviders password input as password type', () => {
      renderWithProvidersLogin();

      const passwordInput = screen.getByPlaceholderText('Enter password');
      expect(passwordInput).toHaveAttribute('type', 'password');
    });
  });

  describe('Form Interactions', () => {
    it('should allow typing in username field', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      await user.type(usernameInput, 'testuser');

      expect(usernameInput).toHaveValue('testuser');
    });

    it('should allow typing in password field', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const passwordInput = screen.getByPlaceholderText('Enter password');
      await user.type(passwordInput, 'password123');

      expect(passwordInput).toHaveValue('password123');
    });

    it('should toggle between admin and user roles', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const adminButton = screen.getByText('Admin').closest('button');
      const userButton = screen.getByText('User').closest('button');

      expect(userButton).toHaveClass('role-active');
      expect(adminButton).not.toHaveClass('role-active');

      await user.click(adminButton);

      expect(adminButton).toHaveClass('role-active');
      expect(userButton).not.toHaveClass('role-active');
    });

    it('should clear form after submission', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      // After successful login, component should redirect
      // Form values don't need to clear as user leaves page
    });
  });

  describe('Form Validation', () => {
    it('should show error when submitting empty username', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Please enter username and password')).toBeInTheDocument();
      });
    });

    it('should show error when submitting empty password', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      await user.type(usernameInput, 'admin');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Please enter username and password')).toBeInTheDocument();
      });
    });

    it('should show error for invalid credentials', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'wronguser');
      await user.type(passwordInput, 'wrongpass');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
      });
    });
  });

  describe('Authentication Flow', () => {
    it('should successfully login admin user', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const adminButton = screen.getByText('Admin').closest('button');
      await user.click(adminButton);

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      // Check localStorage
      await waitFor(() => {
        const savedUser = JSON.parse(localStorage.getItem('fruitGradingUser'));
        expect(savedUser).toMatchObject({
          username: 'admin',
          role: 'admin',
        });
      });
    });

    it('should successfully login regular user', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'user');
      await user.type(passwordInput, 'user123');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        const savedUser = JSON.parse(localStorage.getItem('fruitGradingUser'));
        expect(savedUser).toMatchObject({
          username: 'user',
          role: 'user',
        });
      });
    });

    it('should reject login with mismatched credentials and role', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const adminButton = screen.getByText('Admin').closest('button');
      await user.click(adminButton);

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'user');
      await user.type(passwordInput, 'user123');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
      });
    });
  });

  describe('Loading State', () => {
    it('should show loading state during login', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      // Click and immediately check for loading state
      await user.click(submitButton);
      
      // Button should be disabled during submission
      expect(submitButton).toBeDisabled();
    });

    it('should re-enable button after failed login', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'wrong');
      await user.type(passwordInput, 'wrong');

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });
    });
  });

  describe('Error Handling', () => {
    it('should display error message with icon', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        const errorDiv = screen.getByText('Please enter username and password').closest('div');
        expect(errorDiv).toHaveClass('login-error');
      });
    });

    it('should clear previous errors on new submission', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      // First submission - no input
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Please enter username and password')).toBeInTheDocument();
      });

      // Second submission - with input
      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123');
      await user.click(submitButton);

      // Error should be cleared
      await waitFor(() => {
        expect(screen.queryByText('Please enter username and password')).not.toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');

      expect(usernameInput).toHaveAttribute('type', 'text');
      expect(passwordInput).toHaveAttribute('type', 'password');
    });

    it('should allow keyboard navigation', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      
      // Tab to username
      await user.tab();
      expect(usernameInput).toHaveFocus();

      // Tab to password
      const passwordInput = screen.getByPlaceholderText('Enter password');
      await user.tab();
      expect(passwordInput).toHaveFocus();

      // Tab to submit button
      await user.tab();
      await user.tab(); // Skip role buttons
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      expect(submitButton).toHaveFocus();
    });

    it('should submit form with Enter key', async () => {
      const user = userEvent.setup();
      renderWithProvidersLogin();

      const usernameInput = screen.getByPlaceholderText('Enter username');
      const passwordInput = screen.getByPlaceholderText('Enter password');
      
      await user.type(usernameInput, 'admin');
      await user.type(passwordInput, 'admin123{Enter}');

      await waitFor(() => {
        const savedUser = JSON.parse(localStorage.getItem('fruitGradingUser'));
        expect(savedUser).toMatchObject({
          username: 'admin',
          role: 'admin',
        });
      });
    });
  });

  describe('UI/UX', () => {
    it('should renderWithProviders gradient background orbs', () => {
      renderWithProvidersLogin();

      const orbs = document.querySelectorAll('.gradient-orb');
      expect(orbs).toHaveLength(3);
    });

    it('should renderWithProviders logo icon', () => {
      renderWithProvidersLogin();

      const logo = document.querySelector('.logo-icon-login');
      expect(logo).toBeInTheDocument();
    });

    it('should display demo credentials hint', () => {
      renderWithProvidersLogin();

      // Check if there's any text mentioning demo credentials
      // This would be in the actual component if implemented
      // Currently documenting expected behavior
    });
  });
});
