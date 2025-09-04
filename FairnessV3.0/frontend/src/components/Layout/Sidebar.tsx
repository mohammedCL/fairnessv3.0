import React from 'react';
import { NavLink } from 'react-router-dom';
import { Upload, BarChart3, Shield, Eye, CheckCircle, Circle, ArrowRight, Menu } from 'lucide-react';
import { useFairness } from '../../context/FairnessContext';
import clsx from 'clsx';

const navigationItems = [
  {
    path: '/upload',
    icon: Upload,
    label: 'Upload Model',
    description: 'Upload model and datasets',
    step: 1,
  },
  {
    path: '/analysis',
    icon: BarChart3,
    label: 'Bias Analysis',
    description: 'Detect and analyze bias',
    step: 2,
  },
  {
    path: '/mitigation',
    icon: Shield,
    label: 'Mitigation',
    description: 'Apply fairness strategies',
    step: 3,
  },
  {
    path: '/monitoring',
    icon: Eye,
    label: 'Monitoring',
    description: 'Ongoing fairness tracking',
    step: 4,
  },
];

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggleCollapse }) => {
  const { state, canProceedToAnalysis, canProceedToMitigation } = useFairness();

  const isStepAccessible = (step: number): boolean => {
    switch (step) {
      case 1:
        return true;
      case 2:
        return canProceedToAnalysis;
      case 3:
        return canProceedToMitigation;
      case 4:
        return !!state.mitigationResults;
      default:
        return false;
    }
  };

  const isStepCompleted = (step: number): boolean => {
    const workflowStep = state.workflowSteps.find(s => s.id === step);
    return workflowStep?.completed || false;
  };

  const isStepCurrent = (step: number): boolean => {
    return state.currentStep === step;
  };

  return (
    <aside className={clsx(
      'fixed left-0 top-0 bottom-0 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-all duration-300 z-30',
      collapsed ? 'w-16' : 'w-80'
    )}>
      {/* Header Spacer */}
      <div className="h-16 border-b border-gray-200 dark:border-gray-700"></div>
      
      <div className="flex flex-col h-full">
        {/* Collapse Toggle */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={onToggleCollapse}
            className="w-full flex items-center justify-center p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <Menu className="h-5 w-5" />
          </button>
        </div>

        {/* Collapsed State - Show Only Icons */}
        {collapsed && (
          <div className="flex-1 p-2">
            <div className="space-y-2">
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                const isAccessible = isStepAccessible(item.step);
                const isCurrent = isStepCurrent(item.step);
                const isCompleted = isStepCompleted(item.step);

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) => clsx(
                      'flex items-center justify-center p-3 rounded-lg transition-colors group relative',
                      isActive || isCurrent
                        ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                        : isAccessible
                        ? 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                        : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                    )}
                    onClick={(e) => !isAccessible && e.preventDefault()}
                  >
                    <IconComponent className="h-5 w-5" />
                    {isCompleted && (
                      <div className="absolute -top-1 -right-1 h-3 w-3 bg-green-500 rounded-full"></div>
                    )}
                    {/* Tooltip */}
                    <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
                      {item.label}
                    </div>
                  </NavLink>
                );
              })}
            </div>
          </div>
        )}

        {/* Expanded State - Full Content */}
        {!collapsed && (
          <>
            {/* Navigation Header */}
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Workflow Steps
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                Follow the guided process
              </p>
            </div>

            {/* Progress Tracker */}
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="space-y-4">
                {state.workflowSteps.map((step, index) => (
                  <div key={step.id} className="flex items-center space-x-3">
                    {/* Step Indicator */}
                    <div className="flex-shrink-0">
                      {isStepCompleted(step.id) ? (
                        <CheckCircle className="h-6 w-6 text-green-500" />
                      ) : isStepCurrent(step.id) ? (
                        <div className="h-6 w-6 rounded-full border-2 border-blue-500 bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                          <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                        </div>
                      ) : (
                        <Circle className="h-6 w-6 text-gray-300 dark:text-gray-600" />
                      )}
                    </div>

                    {/* Step Content */}
                    <div className="flex-1 min-w-0">
                      <p className={clsx(
                        'text-sm font-medium',
                        isStepCompleted(step.id) ? 'text-green-700 dark:text-green-400' :
                        isStepCurrent(step.id) ? 'text-blue-700 dark:text-blue-400' :
                        'text-gray-500 dark:text-gray-400'
                      )}>
                        {step.name}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {step.description}
                      </p>
                    </div>

                    {/* Next Arrow */}
                    {index < state.workflowSteps.length - 1 && (
                      <ArrowRight className="h-4 w-4 text-gray-300 dark:text-gray-600" />
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Navigation Menu */}
            <div className="flex-1 p-6">
              <nav className="space-y-2">
                {navigationItems.map((item) => {
                  const IconComponent = item.icon;
                  const isAccessible = isStepAccessible(item.step);
                  const isCurrent = isStepCurrent(item.step);

                  return (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      className={({ isActive }) => clsx(
                        'flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                        isActive || isCurrent
                          ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                          : isAccessible
                          ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                          : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                      )}
                      onClick={(e) => !isAccessible && e.preventDefault()}
                    >
                      <IconComponent className="mr-3 h-5 w-5" />
                      {item.label}
                    </NavLink>
                  );
                })}
              </nav>
            </div>

            {/* Quick Stats */}
            <div className="p-6 border-t border-gray-200 dark:border-gray-700">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                Quick Stats
              </h3>
              
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                  <p className="text-xs text-gray-600 dark:text-gray-400">Files</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {Object.keys(state.uploadedFiles).length}
                  </p>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                  <p className="text-xs text-gray-600 dark:text-gray-400">Progress</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {Math.round((state.workflowSteps.filter(s => s.completed).length / state.workflowSteps.length) * 100)}%
                  </p>
                </div>
              </div>

              {state.analysisResults && (
                <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-3 mt-3">
                  <p className="text-xs text-blue-600 dark:text-blue-400">Fairness Score</p>
                  <p className="text-xl font-bold text-blue-700 dark:text-blue-300">
                    {state.analysisResults.fairness_score.overall_score.toFixed(1)}/100
                  </p>
                  <p className="text-xs text-blue-600 dark:text-blue-400 capitalize">
                    {state.analysisResults.fairness_score.fairness_level}
                  </p>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
