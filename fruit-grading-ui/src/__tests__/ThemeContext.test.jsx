import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider, useTheme } from '../context/ThemeContext';

// Helper component that exposes context values for assertions
const ThemeConsumer = () => {
  const { theme, toggleTheme } = useTheme();
  return (
    <>
      <span data-testid="theme">{theme}</span>
      <button onClick={toggleTheme}>toggle</button>
    </>
  );
};

const renderWithProvider = () =>
  render(
    <ThemeProvider>
      <ThemeConsumer />
    </ThemeProvider>
  );

describe('ThemeContext', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('defaults to dark theme when localStorage is empty', () => {
    renderWithProvider();
    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
  });

  it('reads initial theme from localStorage', () => {
    localStorage.setItem('theme', 'light');
    renderWithProvider();
    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('sets data-theme attribute on <html> on mount', () => {
    renderWithProvider();
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
  });

  it('sets data-theme="light" when localStorage has light', () => {
    localStorage.setItem('theme', 'light');
    renderWithProvider();
    expect(document.documentElement).toHaveAttribute('data-theme', 'light');
  });

  it('toggleTheme switches dark → light', async () => {
    const user = userEvent.setup();
    renderWithProvider();

    await user.click(screen.getByRole('button', { name: 'toggle' }));

    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('toggleTheme switches light → dark', async () => {
    localStorage.setItem('theme', 'light');
    const user = userEvent.setup();
    renderWithProvider();

    await user.click(screen.getByRole('button', { name: 'toggle' }));

    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
  });

  it('persists new theme to localStorage after toggle', async () => {
    const user = userEvent.setup();
    renderWithProvider();

    await user.click(screen.getByRole('button', { name: 'toggle' }));

    expect(localStorage.getItem('theme')).toBe('light');
  });

  it('updates data-theme attribute on <html> after toggle', async () => {
    const user = userEvent.setup();
    renderWithProvider();

    await user.click(screen.getByRole('button', { name: 'toggle' }));

    expect(document.documentElement).toHaveAttribute('data-theme', 'light');
  });
});
