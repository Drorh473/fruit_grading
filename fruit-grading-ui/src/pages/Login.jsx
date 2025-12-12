import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FiUser, FiLock, FiAlertCircle } from 'react-icons/fi';
import './Login.css';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Validate inputs
    if (!username || !password) {
      setError('Please enter username and password');
      setLoading(false);
      return;
    }

    // Attempt login
    const result = login(username, password, role);
    
    if (result.success) {
      // Redirect based on role
      if (role === 'admin') {
        navigate('/dashboard');
      } else {
        navigate('/user-dashboard');
      }
    } else {
      setError(result.error || 'Login failed');
    }
    
    setLoading(false);
  };

  return (
    <div className="login-container">
      <div className="login-background">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <div className="login-card">
        <div className="login-header">
          <div className="logo-container-login">
            <div className="logo-icon-login">
              <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2"/>
                <path d="M20 8 L20 32 M8 20 L32 20" stroke="currentColor" strokeWidth="2"/>
                <circle cx="20" cy="20" r="6" fill="currentColor"/>
              </svg>
            </div>
          </div>
          <h1>Fruit Grading System</h1>
          <p>Sign in to continue</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error">
              <FiAlertCircle />
              <span>{error}</span>
            </div>
          )}

          <div className="form-group">
            <label>Username</label>
            <div className="input-with-icon">
              <FiUser className="input-icon" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="login-input"
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Password</label>
            <div className="input-with-icon">
              <FiLock className="input-icon" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                className="login-input"
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Role</label>
            <div className="role-selector">
              <label className={`role-option ${role === 'user' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="role"
                  value="user"
                  checked={role === 'user'}
                  onChange={(e) => setRole(e.target.value)}
                  disabled={loading}
                />
                <div className="role-content">
                  <span className="role-title">Operator</span>
                  <span className="role-desc">View results only</span>
                </div>
              </label>
              
              <label className={`role-option ${role === 'admin' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="role"
                  value="admin"
                  checked={role === 'admin'}
                  onChange={(e) => setRole(e.target.value)}
                  disabled={loading}
                />
                <div className="role-content">
                  <span className="role-title">Administrator</span>
                  <span className="role-desc">Full system access</span>
                </div>
              </label>
            </div>
          </div>

          <button 
            type="submit" 
            className="login-button"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner-small"></div>
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="login-footer">
          <div className="demo-credentials">
            <p className="demo-title">Demo Credentials:</p>
            <div className="demo-grid">
              <div className="demo-item">
                <strong>Operator:</strong>
                <span>user / user123</span>
              </div>
              <div className="demo-item">
                <strong>Admin:</strong>
                <span>admin / admin123</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
