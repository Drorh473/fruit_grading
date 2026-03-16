import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import PageHeader from '../components/PageHeader';
import { ThemeProvider } from '../context/ThemeContext';

// PageHeader renders ThemeToggle which requires ThemeProvider
const renderHeader = (props = {}) =>
  render(
    <ThemeProvider>
      <PageHeader {...props} />
    </ThemeProvider>
  );

describe('PageHeader', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  // ── Content rendering ──────────────────────────────────────

  it('renders the page title', () => {
    renderHeader({ title: 'System Dashboard' });
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      'System Dashboard'
    );
  });

  it('renders the subtitle when provided', () => {
    renderHeader({ title: 'Test', subtitle: 'A helpful description' });
    expect(screen.getByText('A helpful description')).toBeInTheDocument();
  });

  it('does not render a subtitle element when omitted', () => {
    renderHeader({ title: 'Test' });
    expect(screen.queryByText(/description/i)).not.toBeInTheDocument();
  });

  it('renders extra content (e.g. timestamp) below the subtitle', () => {
    renderHeader({
      title: 'Camera Monitor',
      extra: <p data-testid="timestamp">Last updated: 12:00</p>,
    });
    expect(screen.getByTestId('timestamp')).toBeInTheDocument();
  });

  // ── Children (action buttons) ──────────────────────────────

  it('renders children action buttons', () => {
    renderHeader({
      title: 'Dashboard',
      children: <button>Refresh</button>,
    });
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument();
  });

  it('renders multiple children', () => {
    renderHeader({
      title: 'Processing',
      children: (
        <>
          <button>Refresh</button>
          <button>Start Pipeline</button>
        </>
      ),
    });
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Start Pipeline' })
    ).toBeInTheDocument();
  });

  it('works with no children (Settings/AddFruit pattern)', () => {
    // Should render without crashing
    renderHeader({ title: 'Settings', subtitle: 'Configure the system' });
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  // ── ThemeToggle always present ─────────────────────────────

  it('always renders the theme toggle switch', () => {
    renderHeader({ title: 'Any Page' });
    expect(screen.getByRole('switch')).toBeInTheDocument();
  });

  it('renders toggle as the last element in the actions group', () => {
    const { container } = renderHeader({
      title: 'Dashboard',
      children: <button>Refresh</button>,
    });
    const actions = container.querySelector('.page-header-actions');
    const children = Array.from(actions.children);
    const lastChild = children[children.length - 1];
    // The toggle wrapper is the last child
    expect(lastChild).toHaveClass('theme-toggle-wrapper');
  });

  it('toggle is to the right of the refresh button', () => {
    const { container } = renderHeader({
      title: 'Dashboard',
      children: <button>Refresh</button>,
    });
    const actions = container.querySelector('.page-header-actions');
    const children = Array.from(actions.children);
    const refreshIndex = children.findIndex(
      (el) => el.textContent === 'Refresh'
    );
    const toggleIndex = children.findIndex((el) =>
      el.classList.contains('theme-toggle-wrapper')
    );
    expect(toggleIndex).toBeGreaterThan(refreshIndex);
  });

  // ── Layout structure ───────────────────────────────────────

  it('has page-header class on root element', () => {
    const { container } = renderHeader({ title: 'Test' });
    expect(container.firstChild).toHaveClass('page-header');
  });

  it('has page-header-text and page-header-actions sections', () => {
    const { container } = renderHeader({ title: 'Test' });
    expect(container.querySelector('.page-header-text')).toBeInTheDocument();
    expect(container.querySelector('.page-header-actions')).toBeInTheDocument();
  });
});
