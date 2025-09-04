import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { FairnessProvider } from './context/FairnessContext';
import Layout from './components/Layout/Layout';
import UploadPage from './components/Pages/UploadPage';
import AnalysisPage from './components/Pages/AnalysisPage';
import MitigationPage from './components/Pages/MitigationPage';
import MonitoringPage from './components/Pages/MonitoringPage';
import { Toaster } from 'react-hot-toast';
import './App.css';

function App() {
  return (
    <FairnessProvider>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/upload" replace />} />
              <Route path="upload" element={<UploadPage />} />
              <Route path="analysis" element={<AnalysisPage />} />
              <Route path="mitigation" element={<MitigationPage />} />
              <Route path="monitoring" element={<MonitoringPage />} />
            </Route>
          </Routes>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </FairnessProvider>
  );
}

export default App;