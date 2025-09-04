import React, { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import AIExplanationPanel from './AIExplanationPanel';
import { useFairness } from '../../context/FairnessContext';
import clsx from 'clsx';

const Layout: React.FC = () => {
  const { state } = useFairness();
  const location = useLocation();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [aiPanelOpen, setAiPanelOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('darkMode') === 'true' ||
        (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return false;
  });

  // Apply dark mode to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', isDarkMode.toString());
  }, [isDarkMode]);

  // Handle responsive sidebar and AI panel
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      } else {
        setSidebarCollapsed(false);
      }
      
      // Auto-close AI panel on smaller screens
      if (window.innerWidth < 1280 && aiPanelOpen) {
        setAiPanelOpen(false);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [aiPanelOpen]);

  const getCurrentPageTitle = () => {
    switch (location.pathname) {
      case '/':
      case '/upload':
        return 'Upload Data & Models';
      case '/analysis':
        return 'Bias Analysis';
      case '/mitigation':
        return 'Bias Mitigation';
      case '/monitoring':
        return 'Monitoring Dashboard';
      default:
        return 'Fairness Assessment Platform';
    }
  };

  const getMainContentClass = () => {
    return clsx(
      'flex-1 flex flex-col min-h-0 transition-all duration-300',
      // Account for sidebar width
      sidebarCollapsed ? 'ml-16' : 'ml-80',
      // Account for AI panel width dynamically
      aiPanelOpen ? 'mr-96 xl:mr-96 lg:mr-12' : 'mr-12'
    );
  };

  return (
    <div className="layout-container h-screen flex bg-gray-50 dark:bg-gray-900">
      {/* Fixed Header */}
      <Header 
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        isDarkMode={isDarkMode}
        setIsDarkMode={setIsDarkMode}
      />

      {/* Sidebar */}
      <Sidebar 
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div className={getMainContentClass()}>
        {/* Page Content - now starts from top since header is fixed */}
        <main className="flex-1 overflow-hidden bg-gray-50 dark:bg-gray-900 pt-16">
          <div className="h-full relative">
            {/* Content Container */}
            <div className="h-full overflow-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
              <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-full">
                <Outlet />
              </div>
            </div>

            {/* Loading Overlay */}
            {(state.uploading || state.analyzing || state.mitigating) && (
              <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-30">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-sm w-full mx-4">
                  <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600"></div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        {state.uploading && 'Uploading files...'}
                        {state.analyzing && 'Analyzing for bias...'}
                        {state.mitigating && 'Applying mitigation...'}
                      </h3>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        This may take a few moments
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Progress Bar */}
            {(state.uploading || state.analyzing || state.mitigating) && (
              <div className="absolute top-0 left-0 right-0 z-40">
                <div className="h-1 bg-gray-200 dark:bg-gray-700">
                  <div className="h-1 bg-gradient-to-r from-purple-600 to-blue-600 transition-all duration-300 animate-pulse"></div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Business Report Panel */}
      <AIExplanationPanel 
        onToggleOpen={setAiPanelOpen}
        isOpen={aiPanelOpen}
      />

      {/* Mobile Sidebar Overlay */}
      {!sidebarCollapsed && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      {/* Mobile Business Report Overlay */}
      {aiPanelOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 xl:hidden"
          onClick={() => setAiPanelOpen(false)}
        />
      )}
    </div>
  );
};

export default Layout;
