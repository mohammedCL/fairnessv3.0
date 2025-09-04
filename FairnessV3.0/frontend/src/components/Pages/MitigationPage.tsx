import React, { useState } from 'react';
import { Shield, TrendingUp, Settings, CheckCircle, AlertCircle, ArrowRight, Zap, Target, BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useFairness } from '../../context/FairnessContext';
import { mitigationAPI } from '../../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';

interface MitigationStrategy {
  id: string;
  name: string;
  category: 'preprocessing' | 'inprocessing' | 'postprocessing';
  description: string;
  pros: string[];
  cons: string[];
  complexity: 'low' | 'medium' | 'high';
  effectiveness: number; // 1-10 scale
  icon: React.ReactNode;
}

const mitigationStrategies: MitigationStrategy[] = [
  {
    id: 'resampling',
    name: 'Data Resampling',
    category: 'preprocessing',
    description: 'Balance the dataset by oversampling underrepresented groups or undersampling overrepresented groups.',
    pros: ['Easy to implement', 'Works with any algorithm', 'Preserves original model'],
    cons: ['May lose information', 'Can introduce noise', 'Doesn\'t address root causes'],
    complexity: 'low',
    effectiveness: 7,
    icon: <BarChart3 className="h-5 w-5" />
  },
  {
    id: 'fair_representation',
    name: 'Fair Representation Learning',
    category: 'preprocessing',
    description: 'Transform features to remove bias while preserving predictive information.',
    pros: ['Preserves utility', 'Model-agnostic', 'Addresses multiple bias types'],
    cons: ['Complex implementation', 'May reduce accuracy', 'Requires domain expertise'],
    complexity: 'high',
    effectiveness: 8,
    icon: <Target className="h-5 w-5" />
  },
  {
    id: 'adversarial_debiasing',
    name: 'Adversarial Debiasing',
    category: 'inprocessing',
    description: 'Train model with adversarial component that prevents discrimination.',
    pros: ['State-of-the-art results', 'End-to-end training', 'Flexible constraints'],
    cons: ['Requires neural networks', 'Training instability', 'Complex hyperparameters'],
    complexity: 'high',
    effectiveness: 9,
    icon: <Zap className="h-5 w-5" />
  },
  {
    id: 'fairness_constraints',
    name: 'Fairness Constraints',
    category: 'inprocessing',
    description: 'Add fairness constraints directly to the optimization objective.',
    pros: ['Principled approach', 'Guaranteed fairness', 'Interpretable'],
    cons: ['Limited to specific algorithms', 'May reduce accuracy', 'Complex constraint formulation'],
    complexity: 'medium',
    effectiveness: 8,
    icon: <Settings className="h-5 w-5" />
  },
  {
    id: 'threshold_optimization',
    name: 'Threshold Optimization',
    category: 'postprocessing',
    description: 'Optimize decision thresholds separately for different groups.',
    pros: ['Model-agnostic', 'Easy to implement', 'Immediate results'],
    cons: ['Limited scope', 'May not address root bias', 'Requires group labels'],
    complexity: 'low',
    effectiveness: 6,
    icon: <TrendingUp className="h-5 w-5" />
  },
  {
    id: 'calibration',
    name: 'Calibration',
    category: 'postprocessing',
    description: 'Ensure prediction probabilities are well-calibrated across groups.',
    pros: ['Improves reliability', 'Model-agnostic', 'Addresses prediction quality'],
    cons: ['Limited bias reduction', 'Requires probabilistic outputs', 'Complex evaluation'],
    complexity: 'medium',
    effectiveness: 7,
    icon: <Shield className="h-5 w-5" />
  }
];

const MitigationPage: React.FC = () => {
  const { state, dispatch } = useFairness();
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [isApplying, setIsApplying] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const handleStrategyToggle = (strategyId: string) => {
    setSelectedStrategies(prev => 
      prev.includes(strategyId) 
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId]
    );
  };

  const applyMitigation = async () => {
    if (selectedStrategies.length === 0) {
      toast.error('Please select at least one mitigation strategy');
      return;
    }

    if (!state.analysisResults) {
      toast.error('Analysis results required for mitigation');
      return;
    }

    try {
      setIsApplying(true);
      dispatch({ type: 'SET_MITIGATING', payload: true });

      // Get the category of the first selected strategy
      const selectedStrategy = mitigationStrategies.find(s => s.id === selectedStrategies[0]);
      const strategyCategory = selectedStrategy?.category as 'preprocessing' | 'inprocessing' | 'postprocessing';

      if (!strategyCategory) {
        throw new Error('Invalid strategy selected');
      }

      // Start mitigation
      const mitigationJob = await mitigationAPI.startMitigation(
        state.analysisResults.job_id,
        strategyCategory
      );

      // Poll for results
      const pollResults = async () => {
        try {
          const job = await mitigationAPI.getMitigationJob(mitigationJob.job_id);
          
          if (job.status === 'completed') {
            const results = await mitigationAPI.getMitigationResults(mitigationJob.job_id);
            dispatch({ type: 'SET_MITIGATION_RESULTS', payload: results });
            dispatch({ type: 'COMPLETE_STEP', payload: 3 });
            toast.success('Bias mitigation completed successfully!');
            setIsApplying(false);
            dispatch({ type: 'SET_MITIGATING', payload: false });
          } else if (job.status === 'failed') {
            throw new Error('Mitigation failed');
          } else {
            setTimeout(pollResults, 3000);
          }
        } catch (error) {
          console.error('Error polling mitigation results:', error);
          setTimeout(pollResults, 5000);
        }
      };

      setTimeout(pollResults, 2000);

    } catch (error: any) {
      console.error('Mitigation failed:', error);
      toast.error(error.message || 'Failed to apply mitigation');
      setIsApplying(false);
      dispatch({ type: 'SET_MITIGATING', payload: false });
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-400';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'preprocessing': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
      case 'inprocessing': return 'text-purple-600 bg-purple-100 dark:bg-purple-900/20 dark:text-purple-400';
      case 'postprocessing': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-400';
    }
  };

  const filteredStrategies = selectedCategory === 'all' 
    ? mitigationStrategies 
    : mitigationStrategies.filter(s => s.category === selectedCategory);

  const renderStrategyCard = (strategy: MitigationStrategy) => {
    const isSelected = selectedStrategies.includes(strategy.id);
    
    return (
      <div
        key={strategy.id}
        className={clsx(
          'border rounded-lg p-6 cursor-pointer transition-all',
          isSelected
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
        )}
        onClick={() => handleStrategyToggle(strategy.id)}
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={clsx(
              'p-2 rounded-lg',
              getCategoryColor(strategy.category)
            )}>
              {strategy.icon}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white">
                {strategy.name}
              </h3>
              <div className="flex items-center space-x-2 mt-1">
                <span className={clsx(
                  'px-2 py-1 rounded-full text-xs font-medium capitalize',
                  getCategoryColor(strategy.category)
                )}>
                  {strategy.category.replace('processing', '-processing')}
                </span>
                <span className={clsx(
                  'px-2 py-1 rounded-full text-xs font-medium capitalize',
                  getComplexityColor(strategy.complexity)
                )}>
                  {strategy.complexity} complexity
                </span>
              </div>
            </div>
          </div>
          {isSelected && <CheckCircle className="h-6 w-6 text-blue-600" />}
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          {strategy.description}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <h4 className="text-sm font-medium text-green-700 dark:text-green-400 mb-2">
              Pros
            </h4>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              {strategy.pros.map((pro, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-green-500 mr-1">•</span>
                  {pro}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium text-red-700 dark:text-red-400 mb-2">
              Cons
            </h4>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              {strategy.cons.map((con, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-red-500 mr-1">•</span>
                  {con}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Effectiveness
          </span>
          <div className="flex items-center space-x-2">
            <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${strategy.effectiveness * 10}%` }}
              ></div>
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {strategy.effectiveness}/10
            </span>
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonResults = () => {
    if (!state.mitigationResults) return null;

    const beforeMetrics = state.analysisResults?.bias_metrics || [];
    const afterMetrics = state.mitigationResults.after_metrics;

    const comparisonData = beforeMetrics.map(before => {
      const after = afterMetrics.find((m: any) => m.metric_name === before.metric_name);
      return {
        name: before.metric_name.replace(/_/g, ' '),
        before: before.value,
        after: after?.value || before.value,
        improvement: after ? ((before.value - after.value) / before.value * 100) : 0
      };
    });

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Before vs. After Comparison
        </h3>

        <div className="h-64 mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="name" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--tooltip-bg)', 
                  border: '1px solid var(--tooltip-border)',
                  borderRadius: '6px'
                }}
              />
              <Bar dataKey="before" fill="#EF4444" name="Before Mitigation" />
              <Bar dataKey="after" fill="#10B981" name="After Mitigation" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {comparisonData.slice(0, 3).map((item, index) => (
            <div key={index} className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                {item.name}
              </h4>
              <div className="flex items-center justify-center space-x-2 mb-2">
                <span className="text-red-600 font-semibold">
                  {item.before.toFixed(3)}
                </span>
                <ArrowRight className="h-4 w-4 text-gray-400" />
                <span className="text-green-600 font-semibold">
                  {item.after.toFixed(3)}
                </span>
              </div>
              <span className={clsx(
                'inline-block px-2 py-1 rounded-full text-xs font-medium',
                item.improvement > 0 
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                  : 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400'
              )}>
                {item.improvement > 0 ? '+' : ''}{item.improvement.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (isApplying) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center py-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 dark:bg-purple-900/20 rounded-full mb-4">
            <Shield className="h-8 w-8 text-purple-600 dark:text-purple-400 animate-pulse" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Applying Bias Mitigation
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            Implementing selected strategies to reduce bias in your model...
          </p>
          <div className="max-w-md mx-auto">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div className="bg-purple-600 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Processing mitigation strategies...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!state.analysisResults) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center py-12">
          <AlertCircle className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Analysis Required
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Complete the bias analysis first to access mitigation strategies.
          </p>
          <button
            onClick={() => window.location.href = '/analysis'}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
          >
            Go to Analysis
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Bias Mitigation
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          Select and apply mitigation strategies to reduce bias in your model.
        </p>
      </div>

      {/* Strategy Filter */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Mitigation Strategies
          </h3>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">Filter by category:</span>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
            >
              <option value="all">All Categories</option>
              <option value="preprocessing">Pre-processing</option>
              <option value="inprocessing">In-processing</option>
              <option value="postprocessing">Post-processing</option>
            </select>
          </div>
        </div>

        {selectedStrategies.length > 0 && (
          <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
              Selected Strategies ({selectedStrategies.length})
            </h4>
            <div className="flex flex-wrap gap-2">
              {selectedStrategies.map(id => {
                const strategy = mitigationStrategies.find(s => s.id === id);
                return strategy ? (
                  <span key={id} className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 rounded-full text-sm">
                    {strategy.name}
                  </span>
                ) : null;
              })}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredStrategies.map(renderStrategyCard)}
        </div>

        <div className="mt-6 flex justify-between items-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Select strategies that best fit your use case and requirements.
          </p>
          <button
            onClick={applyMitigation}
            disabled={selectedStrategies.length === 0}
            className={clsx(
              'inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md transition-colors',
              selectedStrategies.length > 0
                ? 'text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500'
                : 'text-gray-400 bg-gray-200 dark:bg-gray-700 cursor-not-allowed'
            )}
          >
            Apply Mitigation
            <Shield className="ml-2 h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Results Comparison */}
      {renderComparisonResults()}

      {/* Next Step */}
      {state.mitigationResults && (
        <div className="flex justify-end">
          <button
            onClick={() => {
              dispatch({ type: 'SET_CURRENT_STEP', payload: 4 });
              window.location.href = '/monitoring';
            }}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Proceed to Monitoring
            <ArrowRight className="ml-2 h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default MitigationPage;
