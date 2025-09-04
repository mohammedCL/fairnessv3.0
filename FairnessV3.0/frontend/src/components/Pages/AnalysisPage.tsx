import React, { useState, useEffect, useCallback } from 'react';
import { useFairness } from '../../context/FairnessContext';
import { analysisAPI } from '../../services/api';
import toast from 'react-hot-toast';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { CheckCircle, AlertTriangle, Info, Brain, Database, Target, TrendingUp } from 'lucide-react';
import clsx from 'clsx';
import './AnalysisPage.css';
// Advanced Bias Analysis Components
import AdvancedMetricsOverview from '../AdvancedBiasAnalysis/AdvancedMetricsOverview';
import DirectBiasAnalysis from '../AdvancedBiasAnalysis/DirectBiasAnalysis';
import ProxyFeatureDetection from '../AdvancedBiasAnalysis/ProxyFeatureDetection';
import StatisticalTestResults from '../AdvancedBiasAnalysis/StatisticalTestResults';
import ModelBasedAnalysis from '../AdvancedBiasAnalysis/ModelBasedAnalysis';
import MultiAttributeAnalysis from '../AdvancedBiasAnalysis/MultiAttributeAnalysis';
import '../AdvancedBiasAnalysis/AdvancedBiasAnalysis.css';

interface AnalysisPageProps {
  // Add props as needed
}

const AnalysisPage: React.FC<AnalysisPageProps> = () => {
  const { state, completeAnalysis, setAnalysisResults } = useFairness();
  const [error, setError] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);

  useEffect(() => {
    // If we have a current analysis job, start polling for status
    if (state.currentAnalysisJobId && state.analyzing) {
      console.log('Starting to poll for analysis job:', state.currentAnalysisJobId);
      pollAnalysisStatus();
    }
  }, [state.currentAnalysisJobId, state.analyzing]);

  const pollAnalysisStatus = useCallback(async () => {
    if (!state.currentAnalysisJobId) return;

    try {
      const jobInfo = await analysisAPI.getAnalysisJob(state.currentAnalysisJobId);
      console.log('Analysis job status:', jobInfo);
      
      setJobStatus(jobInfo.status);
      setProgress(jobInfo.progress || 0);

      if (jobInfo.status === 'completed') {
        // Analysis is complete, fetch results
        console.log('Analysis completed, fetching results...');
        const results = await analysisAPI.getAnalysisResults(state.currentAnalysisJobId);
        console.log('Analysis results:', results);
        
        setAnalysisResults(results);
        completeAnalysis(results);
        toast.success('Analysis completed successfully!');
        
      } else if (jobInfo.status === 'failed') {
        // Analysis failed
        console.error('Analysis failed:', jobInfo.error_message);
        setError(jobInfo.error_message || 'Analysis failed');
        completeAnalysis();
        toast.error('Analysis failed: ' + (jobInfo.error_message || 'Unknown error'));
        
      } else if (jobInfo.status === 'running') {
        // Still running, continue polling
        setTimeout(pollAnalysisStatus, 2000); // Poll every 2 seconds
      }
    } catch (err: any) {
      console.warn('Temporary error polling analysis status, will retry:', err);
      // Don't immediately fail - the backend might just be busy
      // Continue polling with a longer delay
      setTimeout(pollAnalysisStatus, 5000); // Poll every 5 seconds on error
    }
  }, [state.currentAnalysisJobId, completeAnalysis, setAnalysisResults]);

  useEffect(() => {
    // If we have a current analysis job, start polling for status
    if (state.currentAnalysisJobId && state.analyzing) {
      console.log('Starting to poll for analysis job:', state.currentAnalysisJobId);
      pollAnalysisStatus();
    }
  }, [state.currentAnalysisJobId, state.analyzing, pollAnalysisStatus]);

  return (
    <div className="analysis-page">
      <div className="analysis-header">
        <h1>Fairness Analysis</h1>
        <p>Analyze your model for bias and fairness metrics</p>
      </div>

      <div className="analysis-content">
        {state.analyzing && (
          <div className="analysis-status">
            <div className="status-card">
              <div className="status-header">
                <h3>Analysis in Progress</h3>
                <span className="status-badge running">{jobStatus || 'running'}</span>
              </div>
              
              <div className="progress-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <span className="progress-text">{progress}% Complete</span>
              </div>
              
              <p className="status-message">
                This may take a few moments as we analyze your model for bias and fairness metrics...
              </p>
            </div>
          </div>
        )}

        {state.analysisResults && !state.analyzing && (
          <div className="analysis-results">
            <div className="results-header">
              <h2>Analysis Complete</h2>
              <p>Your model has been analyzed for bias and fairness. Review the results below.</p>
            </div>

            {/* Model Information */}
            <div className="result-section">
              <div className="section-header">
                <Brain className="section-icon" />
                <h3>Model Information</h3>
              </div>
              <div className="info-grid">
                <div className="info-card">
                  <label>Model Type</label>
                  <span className="value">{state.analysisResults.model_info?.model_type || 'Unknown'}</span>
                </div>
                <div className="info-card">
                  <label>Task Type</label>
                  <span className="value">{state.analysisResults.model_info?.task_type || 'Unknown'}</span>
                </div>
                <div className="info-card">
                  <label>Target Column</label>
                  <span className="value">{state.analysisResults.model_info?.target_column || 'Unknown'}</span>
                </div>
                <div className="info-card">
                  <label>Number of Features</label>
                  <span className="value">{state.analysisResults.model_info?.n_features || 0}</span>
                </div>
                {state.analysisResults.model_info?.n_classes && (
                  <div className="info-card">
                    <label>Number of Classes</label>
                    <span className="value">{state.analysisResults.model_info.n_classes}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Sensitive Features Detection */}
            <div className="result-section">
              <div className="section-header">
                <Target className="section-icon" />
                <h3>Sensitive Features Detection</h3>
                <Info className="info-icon" />
              </div>
              
              {state.analysisResults.sensitive_features && state.analysisResults.sensitive_features.length > 0 ? (
                <div className="sensitive-features-grid">
                  {state.analysisResults.sensitive_features.map((feature, index) => (
                    <div key={index} className="feature-card">
                      <div className="feature-header">
                        <h4>{feature.feature_name}</h4>
                        <span className={clsx('significance-badge', {
                          'highly-significant': feature.significance_level === 'highly_significant',
                          'very-significant': feature.significance_level === 'very_significant',
                          'significant': feature.significance_level === 'significant',
                          'not-significant': feature.significance_level === 'not_significant'
                        })}>
                          {feature.significance_level?.replace(/_/g, ' ')}
                        </span>
                      </div>
                      
                      <div className="feature-stats">
                        <div className="stat">
                          <label>Statistical Test</label>
                          <span className="value">{feature.test_type?.replace(/_/g, ' ') || 'Unknown'}</span>
                        </div>
                        <div className="stat">
                          <label>P-value</label>
                          <span className="value">{feature.p_value?.toFixed(4) || 'N/A'}</span>
                        </div>
                        <div className="stat">
                          <label>Correlation Score</label>
                          <span className="value">{feature.correlation_score?.toFixed(4) || 'N/A'}</span>
                        </div>
                      </div>
                      
                      <p className="feature-description">{feature.description}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data">
                  <Info className="no-data-icon" />
                  <p>No sensitive features detected in the dataset.</p>
                </div>
              )}
            </div>

            {/* Bias Metrics */}
            <div className="result-section">
              <div className="section-header">
                <TrendingUp className="section-icon" />
                <h3>Bias Metrics ({state.analysisResults.bias_metrics?.length || 0} metrics calculated)</h3>
              </div>
              
              {state.analysisResults.bias_metrics && state.analysisResults.bias_metrics.length > 0 ? (
                <div className="bias-metrics-grid">
                  {state.analysisResults.bias_metrics.map((metric, index) => (
                    <div key={index} className={clsx('metric-card', {
                      'biased': metric.is_biased,
                      'fair': !metric.is_biased
                    })}>
                      <div className="metric-header">
                        <h4>{metric.metric_name}</h4>
                        {metric.is_biased ? (
                          <AlertTriangle className="metric-status alert" />
                        ) : (
                          <CheckCircle className="metric-status good" />
                        )}
                      </div>
                      
                      <div className="metric-value">
                        <span className="value">{metric.value?.toFixed(4) || 'N/A'}</span>
                        <span className="threshold">Threshold: {metric.threshold?.toFixed(4) || 'N/A'}</span>
                      </div>
                      
                      <div className="metric-severity">
                        <span className={clsx('severity-badge', metric.severity)}>
                          {metric.severity} severity
                        </span>
                      </div>
                      
                      <p className="metric-description">{metric.description}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data">
                  <Info className="no-data-icon" />
                  <p>No bias metrics calculated.</p>
                </div>
              )}
            </div>

            {/* Advanced Bias Analysis Section */}
            {state.analysisResults.bias_metrics && state.analysisResults.bias_metrics.length > 0 && (
              <div className="advanced-bias-analysis-section">
                <div className="section-header">
                  <Brain className="section-icon" />
                  <h2>Advanced Bias Analysis</h2>
                  <p className="section-description">
                    Comprehensive statistical analysis with detailed metrics, p-values, and interpretations
                  </p>
                </div>

                {/* Advanced Metrics Overview */}
                <AdvancedMetricsOverview metrics={state.analysisResults.bias_metrics} />

                {/* Direct Bias Analysis */}
                <DirectBiasAnalysis metrics={state.analysisResults.bias_metrics} />

                {/* Proxy Feature Detection */}
                <ProxyFeatureDetection metrics={state.analysisResults.bias_metrics} />

                {/* Statistical Test Results */}
                <StatisticalTestResults metrics={state.analysisResults.bias_metrics} />

                {/* Model-Based Analysis */}
                <ModelBasedAnalysis metrics={state.analysisResults.bias_metrics} />

                {/* Multi-Attribute Analysis */}
                <MultiAttributeAnalysis metrics={state.analysisResults.bias_metrics} />
              </div>
            )}

            {/* Fairness Score */}
            {state.analysisResults.fairness_score && (
              <div className="result-section">
                <div className="section-header">
                  <CheckCircle className="section-icon" />
                  <h3>Overall Fairness Assessment</h3>
                </div>
                
                <div className="fairness-summary">
                  <div className="fairness-score-card">
                    <div className="score-circle">
                      <span className="score-value">
                        {Math.round(state.analysisResults.fairness_score.overall_score * 100)}
                      </span>
                      <span className="score-label">Fairness Score</span>
                    </div>
                    
                    <div className="fairness-level">
                      <span className={clsx('level-badge', state.analysisResults.fairness_score.fairness_level)}>
                        {state.analysisResults.fairness_score.fairness_level?.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                  
                  <div className="fairness-details">
                    <h4>Assessment Details</h4>
                    <p>{state.analysisResults.fairness_score.assessment}</p>
                    
                    {state.analysisResults.fairness_score.recommendations && (
                      <div className="recommendations">
                        <h4>Recommendations</h4>
                        <ul>
                          {state.analysisResults.fairness_score.recommendations.map((rec, index) => (
                            <li key={index}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Visualizations */}
            {state.analysisResults.visualizations && (
              <div className="result-section">
                <div className="section-header">
                  <BarChart className="section-icon" />
                  <h3>Bias Visualizations</h3>
                </div>
                
                <div className="visualizations-grid">
                  {/* Target Distribution Charts */}
                  {state.analysisResults.visualizations.target_distribution && 
                   Array.isArray(state.analysisResults.visualizations.target_distribution) && 
                   state.analysisResults.visualizations.target_distribution.length > 0 ? (
                    <div className="chart-container">
                      <h4>Target Distribution by Sensitive Features</h4>
                      {state.analysisResults.visualizations.target_distribution.map((featureData: any, index: number) => (
                        <div key={index} style={{ marginBottom: '20px' }}>
                          <h5>{featureData.feature_name}</h5>
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={featureData.data}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="group" />
                              <YAxis />
                              <Tooltip />
                              {featureData.keys.map((key: string, keyIndex: number) => (
                                <Bar 
                                  key={key} 
                                  dataKey={key} 
                                  fill={keyIndex === 0 ? "#3B82F6" : "#10B981"} 
                                />
                              ))}
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="chart-container">
                      <h4>Target Distribution by Sensitive Features</h4>
                      <div className="chart-placeholder">
                        No target distribution data available
                      </div>
                    </div>
                  )}

                  {/* Bias Metrics Chart */}
                  {state.analysisResults.visualizations.bias_metrics && 
                   Array.isArray(state.analysisResults.visualizations.bias_metrics) && 
                   state.analysisResults.visualizations.bias_metrics.length > 0 ? (
                    <div className="chart-container">
                      <h4>Bias Metrics Overview</h4>
                      <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={state.analysisResults.visualizations.bias_metrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="metric" 
                            angle={-45} 
                            textAnchor="end" 
                            height={100}
                            fontSize={12}
                          />
                          <YAxis />
                          <Tooltip 
                            formatter={(value, name) => [
                              `${Number(value).toFixed(4)}`, 
                              name === 'value' ? 'Metric Value' : 'Threshold'
                            ]}
                          />
                          <Bar dataKey="value" fill="#DC2626" name="Current Value" />
                          <Bar dataKey="threshold" fill="#9CA3AF" name="Threshold" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="chart-container">
                      <h4>Bias Metrics Overview</h4>
                      <div className="chart-placeholder">
                        No bias metrics visualization data available
                      </div>
                    </div>
                  )}

                  {/* Fairness Breakdown */}
                  {state.analysisResults.visualizations.fairness_breakdown && 
                   Array.isArray(state.analysisResults.visualizations.fairness_breakdown) && 
                   state.analysisResults.visualizations.fairness_breakdown.length > 0 ? (
                    <div className="chart-container">
                      <h4>Fairness Score Breakdown</h4>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={state.analysisResults.visualizations.fairness_breakdown}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="category" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="score" fill="#059669" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="chart-container">
                      <h4>Fairness Score Breakdown</h4>
                      <div className="chart-placeholder">
                        No fairness breakdown data available
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Analysis Summary */}
            {state.analysisResults.analysis_summary && (
              <div className="result-section">
                <div className="section-header">
                  <Database className="section-icon" />
                  <h3>Analysis Summary</h3>
                </div>
                
                <div className="summary-content">
                  <div className="summary-text">
                    {typeof state.analysisResults.analysis_summary === 'string' ? (
                      <p>{state.analysisResults.analysis_summary}</p>
                    ) : (
                      <div>
                        {Object.entries(state.analysisResults.analysis_summary).map(([key, value]) => (
                          <div key={key} className="summary-item">
                            <strong>{key.replace(/_/g, ' ')}:</strong> {String(value)}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-message">
            <h3>Analysis Error</h3>
            <p>{error}</p>
          </div>
        )}

        {!state.analyzing && !state.analysisResults && !error && (
          <div className="no-analysis">
            <p>No analysis in progress. Please start from the Upload page.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisPage;