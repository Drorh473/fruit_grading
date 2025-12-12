import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  FiHome, 
  FiCamera, 
  FiCpu, 
  FiBarChart2, 
  FiSettings,
  FiPlusCircle
} from 'react-icons/fi';
import './Sidebar.css';

const Sidebar = () => {
  const navItems = [
    { path: '/dashboard', icon: <FiHome />, label: 'Dashboard' },
    { path: '/cameras', icon: <FiCamera />, label: 'Camera Monitor' },
    { path: '/processing', icon: <FiCpu />, label: 'Processing' },
    { path: '/results', icon: <FiBarChart2 />, label: 'Results' },
    { path: '/add-fruit', icon: <FiPlusCircle />, label: 'Add Fruit' },
    { path: '/settings', icon: <FiSettings />, label: 'Settings' }
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-icon">🍎</div>
          <div className="logo-text">
            <h1>Fruit Grading</h1>
            <p>ML System</p>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="version-info">
          <p>Version 1.0.0</p>
          <p className="build-date">Build: Dec 2025</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
