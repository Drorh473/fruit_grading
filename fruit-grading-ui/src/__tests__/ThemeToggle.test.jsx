import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ThemeToggle from '../components/ThemeToggle';
import { ThemeProvider } from '../context/ThemeContext';

const renderToggle = (props = {}) =>
  render(
    <ThemeProvider>
      <ThemeToggle {...props} />
    </ThemeProvider>
  );

describe('ThemeToggle', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  // ── Rendering ─────────────────────────────────────────────

  it('renders the toggle switch button', () => {
    renderToggle();
    expect(screen.getByRole('switch')).toBeInTheDocument();
  });

  it('has an accessible label', () => {
    renderToggle();
    expect(screen.getByRole('switch')).toHaveAccessibleName(
      /toggle light\/dark mode/i
    );
  });

  it('renders without fixed class by default', () => {
    const { container } = renderToggle();
    const wrapper = container.querySelector('.theme-toggle-wrapper');
    expect(wrapper).not.toHaveClass('fixed');
  });

  it('adds fixed class when fixed prop is true', () => {
    const { container } = renderToggle({ fixed: true });
    const wrapper = container.querySelector('.theme-toggle-wrapper');
    expect(wrapper).toHaveClass('fixed');
  });

  // ── Dark mode state ────────────────────────────────────────

  it('switch is aria-checked=false in dark mode (light is OFF)', () => {
    // dark is the default; the switch represents "is light mode on?"
    renderToggle();
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'false');
  });

  it('toggle track does NOT have light class in dark mode', () => {
    const { container } = renderToggle();
    expect(container.querySelector('.toggle-track')).not.toHaveClass('light');
  });

  it('moon icon is active (accent color) in dark mode', () => {
    const { container } = renderToggle();
    const icons = container.querySelectorAll('.toggle-icon');
    // First icon is moon
    expect(icons[0]).toHaveClass('active');
    expect(icons[1]).not.toHaveClass('active');
  });

  // ── Light mode state ───────────────────────────────────────

  it('switch is aria-checked=true in light mode', () => {
    localStorage.setItem('theme', 'light');
    renderToggle();
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true');
  });

  it('toggle track has light class in light mode', () => {
    localStorage.setItem('theme', 'light');
    const { container } = renderToggle();
    expect(container.querySelector('.toggle-track')).toHaveClass('light');
  });

  it('sun icon is active in light mode', () => {
    localStorage.setItem('theme', 'light');
    const { container } = renderToggle();
    const icons = container.querySelectorAll('.toggle-icon');
    // Second icon is sun
    expect(icons[1]).toHaveClass('active');
    expect(icons[0]).not.toHaveClass('active');
  });

  // ── Interaction ────────────────────────────────────────────

  it('clicking the switch toggles from dark to light', async () => {
    const user = userEvent.setup();
    renderToggle();

    await user.click(screen.getByRole('switch'));

    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true');
    expect(localStorage.getItem('theme')).toBe('light');
  });

  it('clicking the switch toggles from light to dark', async () => {
    localStorage.setItem('theme', 'light');
    const user = userEvent.setup();
    renderToggle();

    await user.click(screen.getByRole('switch'));

    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'false');
    expect(localStorage.getItem('theme')).toBe('dark');
  });
});
