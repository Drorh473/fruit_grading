import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import {
  FiHome,
  FiCamera,
  FiCpu,
  FiBarChart2,
  FiSettings,
  FiPlusCircle,
  FiLogOut,
  FiMenu,
} from "react-icons/fi";
import { useAuth } from "../context/AuthContext";
import { NAVIGATION_BY_ROLE } from "../utils/roleConfig";
import "./Sidebar.css";

const Sidebar = () => {
  const { user, logout, isAdmin } = useAuth();
  const [isHovered, setIsHovered] = useState(false);

  // **FIX: Ensure sidebar-open class is removed on mount and unmount**
  useEffect(() => {
    // Remove class immediately when component mounts
    document.body.classList.remove("sidebar-open");

    // Cleanup: remove class when component unmounts
    return () => {
      document.body.classList.remove("sidebar-open");
    };
  }, []);

  // Get icon component by name
  const getIcon = (iconName) => {
    const icons = {
      FiHome: <FiHome />,
      FiCamera: <FiCamera />,
      FiCpu: <FiCpu />,
      FiBarChart2: <FiBarChart2 />,
      FiPlusCircle: <FiPlusCircle />,
      FiSettings: <FiSettings />,
    };
    return icons[iconName] || <FiHome />;
  };

  // Get navigation items based on user role
  const navItems = user ? NAVIGATION_BY_ROLE[user.role] || [] : [];

  const handleLogout = () => {
    if (window.confirm("Are you sure you want to logout?")) {
      logout();
    }
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
    document.body.classList.add("sidebar-open");
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    document.body.classList.remove("sidebar-open");
  };

  return (
    <>
      {/* Hamburger Button */}
      <div className="hamburger-container" onMouseEnter={handleMouseEnter}>
        <button className="hamburger-button">
          <FiMenu />
        </button>
      </div>

      {/* Sidebar */}
      <aside
        className={`sidebar ${isHovered ? "visible" : ""}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">
              <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                <circle
                  cx="18"
                  cy="18"
                  r="16"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <circle
                  cx="18"
                  cy="18"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <circle cx="18" cy="18" r="4" fill="currentColor" />
                <path
                  d="M18 2 L18 8 M18 28 L18 34 M2 18 L8 18 M28 18 L34 18"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <path
                  d="M6 6 L11 11 M25 25 L30 30 M30 6 L25 11 M11 25 L6 30"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </div>
            <div className="logo-text">
              <h1>Fruit Grading</h1>
              <p>ML System</p>
            </div>
          </div>
        </div>

        {user && (
          <div className="user-info">
            <div className="user-avatar">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div className="user-details">
              <span className="user-name">{user.username}</span>
              <span className="user-role">
                {isAdmin() ? "Administrator" : "Operator"}
              </span>
            </div>
          </div>
        )}

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
            >
              <span className="nav-icon">{getIcon(item.icon)}</span>
              <span className="nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="logout-button" onClick={handleLogout}>
            <FiLogOut />
            <span>Logout</span>
          </button>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
