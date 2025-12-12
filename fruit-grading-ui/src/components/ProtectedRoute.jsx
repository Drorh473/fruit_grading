import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { hasAccess } from '../utils/roleConfig';

const ProtectedRoute = ({ children, path }) => {
  const { user, loading } = useAuth();

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: 'var(--bg-dark)',
        color: 'var(--text-primary)'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 1rem' }}></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  // Not authenticated - redirect to login
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // Authenticated but no access to this route - redirect to appropriate dashboard
  if (path && !hasAccess(user.role, path)) {
    const redirectPath = user.role === 'admin' ? '/dashboard' : '/user-dashboard';
    return <Navigate to={redirectPath} replace />;
  }

  // Authenticated and has access - render the protected content
  return children;
};

export default ProtectedRoute;
