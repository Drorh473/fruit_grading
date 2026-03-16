import React from "react";
import { FiSun, FiMoon } from "react-icons/fi";
import { useTheme } from "../context/ThemeContext";
import "./ThemeToggle.css";

const ThemeToggle = ({ fixed = false }) => {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div className={`theme-toggle-wrapper${fixed ? " fixed" : ""}`}>
      <FiMoon className={`toggle-icon ${isDark ? "active" : ""}`} />
      <button
        role="switch"
        aria-checked={!isDark}
        aria-label="Toggle light/dark mode"
        className={`toggle-track ${isDark ? "" : "light"}`}
        onClick={toggleTheme}
      >
        <span className="toggle-thumb" />
      </button>
      <FiSun className={`toggle-icon ${!isDark ? "active" : ""}`} />
    </div>
  );
};

export default ThemeToggle;
