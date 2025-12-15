import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Sidebar from "./components/Sidebar";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import UserDashboard from "./pages/UserDashboard";
import CameraMonitor from "./pages/CameraMonitor";
import Processing from "./pages/Processing";
import Results from "./pages/Results";
import Settings from "./pages/Settings";
import AddFruit from "./pages/AddFruit";
import "./App.css";

// Main app content with authentication
function AppContent() {
  const { user } = useAuth();

  const [systemStatus, setSystemStatus] = useState({
    database: "connected",
    model: "loaded",
    cameras: [true, true, true, true],
  });

  const [processingStats, setProcessingStats] = useState({
    totalProcessed: 0,
    accuracy: 0,
    lastUpdate: null,
  });

  // Show login page if not authenticated
  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  // Show app with sidebar for authenticated users
  return (
    <div className="app">
      <Sidebar />
      <main className="main-content">
        <Routes>
          {/* Default redirect based on role */}
          <Route
            path="/"
            element={
              <Navigate
                to={user.role === "admin" ? "/dashboard" : "/user-dashboard"}
                replace
              />
            }
          />

          {/* Login route (redirects if already logged in) */}
          <Route
            path="/login"
            element={
              <Navigate
                to={user.role === "admin" ? "/dashboard" : "/user-dashboard"}
                replace
              />
            }
          />

          {/* Admin-only routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute path="/dashboard">
                <Dashboard
                  systemStatus={systemStatus}
                  processingStats={processingStats}
                />
              </ProtectedRoute>
            }
          />

          <Route
            path="/cameras"
            element={
              <ProtectedRoute path="/cameras">
                <CameraMonitor
                  systemStatus={systemStatus}
                  setSystemStatus={setSystemStatus}
                />
              </ProtectedRoute>
            }
          />

          <Route
            path="/processing"
            element={
              <ProtectedRoute path="/processing">
                <Processing setProcessingStats={setProcessingStats} />
              </ProtectedRoute>
            }
          />

          <Route
            path="/add-fruit"
            element={
              <ProtectedRoute path="/add-fruit">
                <AddFruit />
              </ProtectedRoute>
            }
          />

          <Route
            path="/settings"
            element={
              <ProtectedRoute path="/settings">
                <Settings
                  systemStatus={systemStatus}
                  setSystemStatus={setSystemStatus}
                />
              </ProtectedRoute>
            }
          />

          {/* User dashboard (user-only route) */}
          <Route
            path="/user-dashboard"
            element={
              <ProtectedRoute path="/user-dashboard">
                <UserDashboard systemStatus={systemStatus} />
              </ProtectedRoute>
            }
          />

          {/* Results page (accessible by both roles) */}
          <Route
            path="/results"
            element={
              <ProtectedRoute path="/results">
                <Results />
              </ProtectedRoute>
            }
          />

          {/* Catch all - redirect to appropriate dashboard */}
          <Route
            path="*"
            element={
              <Navigate
                to={user.role === "admin" ? "/dashboard" : "/user-dashboard"}
                replace
              />
            }
          />
        </Routes>
      </main>
    </div>
  );
}

// Main App component with AuthProvider
function App() {
  return (
    <Router>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </Router>
  );
}

export default App;
