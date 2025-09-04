import React, { useState, useEffect } from 'react';
import { FileText, Brain, Loader, Lightbulb, AlertCircle, CheckCircle, TrendingUp, Shield, Target, Eye, BarChart3 } from 'lucide-react';
import { useFairness } from '../../context/FairnessContext';
import { aiAPI } from '../../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';

interface AIExplanationPanelProps {
  onToggleOpen: (isOpen: boolean) => void;
  isOpen: boolean;
}

const AIExplanationPanel: React.FC<AIExplanationPanelProps> = ({ onToggleOpen, isOpen: isOpenProp }) => {
  const { state, setAIRecommendations } = useFairness();
  const [isOpen, setIsOpen] = useState(isOpenProp);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiServiceStatus, setAiServiceStatus] = useState<any>(null);

  // Sync internal state with parent prop
  React.useEffect(() => {
    setIsOpen(isOpenProp);
  }, [isOpenProp]);

  const togglePanel = () => {
    const newIsOpen = !isOpen;
    setIsOpen(newIsOpen);
    onToggleOpen(newIsOpen);
  };

  // Check AI service status on mount
  useEffect(() => {
    console.log('AIExplanationPanel mounted, checking AI status...');
    const checkAIStatus = async () => {
      try {
        console.log('Calling AI service status API...');
        const status = await aiAPI.getAIServiceStatus();
        console.log('AI service status received:', status);
        setAiServiceStatus(status);
      } catch (error) {
        console.error('Failed to check AI service status:', error);
      }
    };

    checkAIStatus();
  }, []);

  // Auto-generate recommendations when analysis is complete
  useEffect(() => {
    if (state.analysisResults && !state.aiRecommendations && !isGenerating) {
      generateRecommendations();
    }
  }, [state.analysisResults]);

  const generateRecommendations = async () => {
    if (!state.analysisResults) return;

    console.log('Starting to generate recommendations for job:', state.analysisResults.job_id);
    setIsGenerating(true);
    try {
      const recommendations = await aiAPI.generateRecommendations(
        state.analysisResults.job_id,
        state.mitigationResults?.job_id
      );
      
      console.log('Received recommendations:', recommendations);
      
      // Store the recommendations in the state
      setAIRecommendations(recommendations);
      
      toast.success('AI explanation generated successfully!');
      setIsOpen(true);
      
    } catch (error) {
      console.error('Failed to generate recommendations:', error);
      toast.error('Failed to generate AI explanation');
    } finally {
      setIsGenerating(false);
    }
  };

  const formatBusinessReport = () => {
    console.log('formatBusinessReport called with:', {
      aiRecommendations: state.aiRecommendations,
      analysisResults: state.analysisResults
    });
    
    if (!state.aiRecommendations || !state.analysisResults) return null;

    const totalMetrics = state.analysisResults.bias_metrics?.length || 0;
    const biasedMetrics = state.analysisResults.bias_metrics?.filter(m => m.is_biased).length || 0;
    const fairnessScore = state.analysisResults.fairness_score?.overall_score || 0;

    return (
      <div className="space-y-6">
        {/* Executive Summary */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
          <div className="flex items-center mb-3">
            <BarChart3 className="h-5 w-5 text-blue-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Executive Summary</h3>
          </div>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{Math.round(fairnessScore * 100)}%</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">Fairness Score</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{biasedMetrics}/{totalMetrics}</div>
              <div className="text-xs text-gray-600 dark:text-gray-400">Biased Metrics</div>
            </div>
          </div>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            {state.analysisResults.fairness_score?.assessment || 'Model assessment completed with detailed bias analysis.'}
          </p>
        </div>

        {/* Bias Reduction Strategies */}
        <div className="border-l-4 border-red-500 pl-4">
          <div className="flex items-center mb-3">
            <Target className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Bias Reduction Strategies</h3>
          </div>
          <div className="space-y-2">
            {state.aiRecommendations.bias_reduction_strategies?.map((strategy: string, index: number) => (
              <div key={index} className="flex items-start">
                <div className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{strategy}</p>
              </div>
            )) || <p className="text-sm text-gray-500">No specific strategies identified.</p>}
          </div>
        </div>

        {/* Model Retraining Approaches */}
        <div className="border-l-4 border-green-500 pl-4">
          <div className="flex items-center mb-3">
            <TrendingUp className="h-5 w-5 text-green-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Model Improvement Recommendations</h3>
          </div>
          <div className="space-y-2">
            {state.aiRecommendations.model_retraining_approaches?.map((approach: string, index: number) => (
              <div key={index} className="flex items-start">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{approach}</p>
              </div>
            )) || <p className="text-sm text-gray-500">No specific recommendations available.</p>}
          </div>
        </div>

        {/* Data Quality Improvements */}
        <div className="border-l-4 border-yellow-500 pl-4">
          <div className="flex items-center mb-3">
            <BarChart3 className="h-5 w-5 text-yellow-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Data Quality Enhancements</h3>
          </div>
          <div className="space-y-2">
            {state.aiRecommendations.data_collection_improvements?.map((improvement: string, index: number) => (
              <div key={index} className="flex items-start">
                <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{improvement}</p>
              </div>
            )) || <p className="text-sm text-gray-500">No specific improvements identified.</p>}
          </div>
        </div>

        {/* Monitoring Practices */}
        <div className="border-l-4 border-purple-500 pl-4">
          <div className="flex items-center mb-3">
            <Eye className="h-5 w-5 text-purple-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Ongoing Monitoring</h3>
          </div>
          <div className="space-y-2">
            {state.aiRecommendations.monitoring_practices?.map((practice: string, index: number) => (
              <div key={index} className="flex items-start">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{practice}</p>
              </div>
            )) || <p className="text-sm text-gray-500">No specific practices recommended.</p>}
          </div>
        </div>

        {/* Compliance Considerations */}
        <div className="border-l-4 border-indigo-500 pl-4">
          <div className="flex items-center mb-3">
            <Shield className="h-5 w-5 text-indigo-600 mr-2" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Compliance & Legal Considerations</h3>
          </div>
          <div className="space-y-2">
            {state.aiRecommendations.compliance_considerations?.map((consideration: string, index: number) => (
              <div key={index} className="flex items-start">
                <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{consideration}</p>
              </div>
            )) || <p className="text-sm text-gray-500">No specific considerations noted.</p>}
          </div>
        </div>

        {/* Confidence Score */}
        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Report Confidence</span>
            <span className="text-sm font-bold text-gray-900 dark:text-white">
              {Math.round((state.aiRecommendations.confidence_score || 0) * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
            <div 
              className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full" 
              style={{ width: `${(state.aiRecommendations.confidence_score || 0) * 100}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Generated on {new Date(state.aiRecommendations.generated_at || Date.now()).toLocaleDateString()}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className={clsx(
      'fixed right-0 top-0 bottom-0 bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700 flex flex-col transition-all duration-300 z-40',
      isOpen ? 'w-96 xl:w-96 lg:w-80 md:w-72 sm:w-64' : 'w-12'
    )}>
      {/* Header Spacer */}
      <div className="h-16 border-b border-gray-200 dark:border-gray-700"></div>
      
      {/* Toggle Button */}
      <button
        onClick={togglePanel}
        className={clsx(
          'flex items-center justify-center p-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 transition-colors',
          !isOpen && 'writing-mode-vertical'
        )}
        aria-label="Toggle AI Explanation"
      >
        <FileText className="h-5 w-5" />
        {!isOpen && (
          <span className="ml-2 text-sm font-medium transform rotate-90 whitespace-nowrap">
            Explain with AI
          </span>
        )}
        {isOpen && (
          <span className="ml-2 text-sm font-medium">Explain with AI</span>
        )}
      </button>

      {/* Panel Content */}
      {isOpen && (
        <>
          {/* Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <FileText className="h-5 w-5 text-blue-600" />
              <h3 className="font-semibold text-gray-900 dark:text-white">
                AI Explanation Report
              </h3>
            </div>
            
            {aiServiceStatus && (
              <div className="mt-2 flex items-center space-x-2">
                {aiServiceStatus.using_mock_responses ? (
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                ) : (
                  <CheckCircle className="h-4 w-4 text-green-500" />
                )}
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {aiServiceStatus.using_mock_responses ? 'Demo Mode' : 'AI Analysis'}
                </span>
              </div>
            )}
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto">
            {!state.analysisResults ? (
              /* No Analysis Yet */
              <div className="flex-1 flex items-center justify-center p-6">
                <div className="text-center">
                  <Lightbulb className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                    AI Explanation
                  </h4>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    Complete your bias analysis to generate AI-powered explanations and insights
                  </p>
                </div>
              </div>
            ) : isGenerating ? (
              /* Generating */
              <div className="flex-1 flex items-center justify-center p-6">
                <div className="text-center">
                  <Loader className="h-8 w-8 text-blue-600 mx-auto mb-4 animate-spin" />
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                    Generating AI Explanation...
                  </h4>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    Analyzing results and preparing AI-powered insights
                  </p>
                </div>
              </div>
            ) : (
              /* AI Explanation Content */
              <div className="p-4">
                {formatBusinessReport()}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default AIExplanationPanel;
