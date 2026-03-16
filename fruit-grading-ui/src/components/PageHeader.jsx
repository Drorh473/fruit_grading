import React from "react";
import ThemeToggle from "./ThemeToggle";
import "./PageHeader.css";

/**
 * Shared page header used by every page.
 * Always renders ThemeToggle as the rightmost action.
 *
 * @param {string}      title     - Page title (h1)
 * @param {string}      subtitle  - Descriptive subtitle
 * @param {ReactNode}   extra     - Extra content below subtitle (e.g. last-update timestamp)
 * @param {ReactNode}   children  - Action buttons rendered left of the toggle
 */
const PageHeader = ({ title, subtitle, extra, children }) => {
  return (
    <div className="page-header">
      <div className="page-header-text">
        <h1>{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
        {extra}
      </div>

      <div className="page-header-actions">
        {children}
        <ThemeToggle />
      </div>
    </div>
  );
};

export default PageHeader;
