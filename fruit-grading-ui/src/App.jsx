import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import CameraMonitor from './pages/CameraMonitor';
import Processing from './pages/Processing';
import Results from './pages/Results';
import Settings from './pages/Settings';
import AddFruit from './pages/AddFruit';
import './App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState({
    database: 'connected',
    model: 'loaded',
    cameras: [true, true, true, true]
  });

  const [processingStats, setProcessingStats] = useState({
    totalProcessed: 0,
    accuracy: 0,
    lastUpdate: null
  });

  return (
    <Router>
      <div className="app">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route 
              path="/dashboard" 
              element={
                <Dashboard 
                  systemStatus={systemStatus} 
                  processingStats={processingStats} 
                />
              } 
            />
            <Route 
              path="/cameras" 
              element={<CameraMonitor systemStatus={systemStatus} />} 
            />
            <Route 
              path="/processing" 
              element={
                <Processing 
                  setProcessingStats={setProcessingStats} 
                />
              } 
            />
            <Route path="/results" element={<Results />} />
            <Route path="/add-fruit" element={<AddFruit />} />
            <Route 
              path="/settings" 
              element={
                <Settings 
                  systemStatus={systemStatus} 
                  setSystemStatus={setSystemStatus} 
                />
              } 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
