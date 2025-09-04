import React from 'react';
import { Shield, Menu, Sun, Moon, RotateCcw } from 'lucide-react';
import clsx from 'clsx';
import { useFairness } from '../../context/FairnessContext';
import toast from 'react-hot-toast';

interface HeaderProps {
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  isDarkMode: boolean;
  setIsDarkMode: (darkMode: boolean) => void;
}

const Header: React.FC<HeaderProps> = ({ 
  sidebarCollapsed, 
  setSidebarCollapsed, 
  isDarkMode, 
  setIsDarkMode 
}) => {
  const { state, resetState } = useFairness();
  
  // Check if there's any data to reset
  const hasDataToReset = Object.keys(state.uploadedFiles).length > 0 || 
                        state.analysisResults || 
                        state.mitigationResults || 
                        state.aiRecommendations;

  const handleStartOver = () => {
    if (window.confirm('Are you sure you want to start over? This will clear all uploaded files and results.')) {
      resetState();
      toast.success('Session cleared. Starting fresh!');
    }
  };
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 h-16 shadow-sm">
      <div className="flex items-center justify-between h-full">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="lg:hidden p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <Menu className="w-6 h-6" />
          </button>
          
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-lg">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                Fairness AI
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Machine Learning Bias Detection & Mitigation
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Start Over Button - only show if there's data to reset */}
          {hasDataToReset && (
            <button
              onClick={handleStartOver}
              className={clsx(
                "px-3 py-2 text-sm font-medium rounded-lg transition-colors",
                "text-gray-700 dark:text-gray-300",
                "bg-gray-100 dark:bg-gray-700",
                "hover:bg-gray-200 dark:hover:bg-gray-600",
                "border border-gray-300 dark:border-gray-600",
                "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              )}
              title="Clear all data and start over"
            >
              <div className="flex items-center space-x-2">
                <RotateCcw className="w-4 h-4" />
                <span className="hidden sm:inline">Start Over</span>
              </div>
            </button>
          )}
          
          {/* Dark Mode Toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={clsx(
              "p-2 rounded-lg transition-colors",
              "text-gray-600 dark:text-gray-300",
              "hover:bg-gray-100 dark:hover:bg-gray-700"
            )}
            title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDarkMode ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;