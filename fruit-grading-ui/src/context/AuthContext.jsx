import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for saved session on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('fruitGradingUser');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        // Check if session is expired (1 hour)
        const loginTime = new Date(userData.loginTime);
        const now = new Date();
        const hoursSinceLogin = (now - loginTime) / (1000 * 60 * 60);
        
        if (hoursSinceLogin > 1) {
          // Session expired, clear storage
          localStorage.removeItem('fruitGradingUser');
        } else {
          setUser(userData);
        }
      } catch (error) {
        localStorage.removeItem('fruitGradingUser');
      }
    }
    setLoading(false);
  }, []);

  const login = (username, password, role) => {
    // In production, this would call your backend API
    // For now, we'll simulate authentication
    
    // Demo credentials:
    // Admin: username="admin", password="admin123"
    // User: username="user", password="user123"
    
    if (
      (username === 'admin' && password === 'admin123' && role === 'admin') ||
      (username === 'user' && password === 'user123' && role === 'user')
    ) {
      const userData = {
        username,
        role,
        loginTime: new Date().toISOString()
      };
      
      setUser(userData);
      localStorage.setItem('fruitGradingUser', JSON.stringify(userData));
      return { success: true };
    }
    
    return { success: false, error: 'Invalid credentials' };
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('fruitGradingUser');
  };

  const isAdmin = () => {
    return user?.role === 'admin';
  };

  const isUser = () => {
    return user?.role === 'user';
  };

  const value = {
    user,
    login,
    logout,
    isAdmin,
    isUser,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
