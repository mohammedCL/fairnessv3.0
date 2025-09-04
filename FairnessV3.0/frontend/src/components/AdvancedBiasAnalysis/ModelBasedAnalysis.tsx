import React from 'react';
import { Brain, Target, BarChart3, AlertTriangle, CheckCircle } from 'lucide-react';
import { BiasMetric } from '../../types';
import clsx from 'clsx';

interface ModelBasedAnalysisProps {
  metrics: BiasMetric[];
}

const ModelBasedAnalysis: React.FC<ModelBasedAnalysisProps> = ({ metrics }) => {
  // Filter for model-based metrics
  const modelMetrics = metrics.filter(metric => 
    metric.metric_name.includes('Model-Based') || 
    metric.metric_name.includes('Standard')
  );

  if (modelMetrics.length === 0) {
    return null;
  }

  // Group by fairness metric type
  const fairnessTypes = {
    'Statistical Parity': modelMetrics.filter(m => m.metric_name.includes('Statistical Parity')),
    'Disparate Impact': modelMetrics.filter(m => m.metric_name.includes('Disparate Impact')),
    'Equal Opportunity': modelMetrics.filter(m => m.metric_name.includes('Equal Opportunity')),
    'Equalized Odds': modelMetrics.filter(m => m.metric_name.includes('Equalized Odds')),
    'Calibration': modelMetrics.filter(m => m.metric_name.includes('Calibration')),
    'Individual Fairness': modelMetrics.filter(m => m.metric_name.includes('Entropy'))
  };

  const getFairnessDescription = (type: string) => {
    switch (type) {
      case 'Statistical Parity':
        return 'Equal positive prediction rates across all groups';
      case 'Disparate Impact':
        return 'Ratio of positive prediction rates should be close to 1.0';
      case 'Equal Opportunity':
        return 'Equal true positive rates across groups (important for beneficial outcomes)';
      case 'Equalized Odds':
        return 'Equal true positive and false positive rates across groups';
      case 'Calibration':
        return 'Prediction probabilities should reflect actual outcomes equally across groups';
      case 'Individual Fairness':
        return 'Similar individuals should receive similar treatment';
      default:
        return 'Fairness metric assessment';
    }
  };

  const getFairnessThresholds = (type: string) => {
    switch (type) {
      case 'Statistical Parity':
        return { good: 0.05, acceptable: 0.1, poor: 0.2 };
      case 'Disparate Impact':
        return { good: 0.9, acceptable: 0.8, poor: 0.6 };
      case 'Equal Opportunity':
        return { good: 0.05, acceptable: 0.1, poor: 0.2 };
      case 'Equalized Odds':
        return { good: 0.05, acceptable: 0.1, poor: 0.2 };
      case 'Calibration':
        return { good: 0.05, acceptable: 0.1, poor: 0.2 };
      case 'Individual Fairness':
        return { good: 0.3, acceptable: 0.5, poor: 1.0 };
      default:
        return { good: 0.05, acceptable: 0.1, poor: 0.2 };
    }
  };

  const getFairnessLevel = (value: number, type: string) => {
    const thresholds = getFairnessThresholds(type);
    
    if (type === 'Disparate Impact') {
      if (value >= thresholds.good) return { level: 'good', color: 'success' };
      if (value >= thresholds.acceptable) return { level: 'acceptable', color: 'warning' };
      return { level: 'poor', color: 'danger' };
    } else {
      if (value <= thresholds.good) return { level: 'good', color: 'success' };
      if (value <= thresholds.acceptable) return { level: 'acceptable', color: 'warning' };
      return { level: 'poor', color: 'danger' };
    }
  };

  return (
    <div className="result-section">
      <div className="section-header">
        <Brain className="section-icon" />
        <h3>Model-Based Fairness Analysis</h3>
        <p className="section-description">
          Comprehensive fairness metrics computed using model predictions
        </p>
      </div>

      {Object.entries(fairnessTypes).map(([fairnessType, typeMetrics]) => {
        if (typeMetrics.length === 0) return null;

        return (
          <div key={fairnessType} className="fairness-type-section">
            <div className="fairness-type-header">
              <Target className="fairness-icon" />
              <div className="fairness-info">
                <h4>{fairnessType}</h4>
                <p>{getFairnessDescription(fairnessType)}</p>
              </div>
              <div className="fairness-status">
                {typeMetrics.some(m => m.is_biased) ? (
                  <span className="status-badge biased">BIAS DETECTED</span>
                ) : (
                  <span className="status-badge fair">FAIR</span>
                )}
              </div>
            </div>

            <div className="fairness-metrics-grid">
              {typeMetrics.map((metric, index) => {
                const fairnessLevel = getFairnessLevel(metric.value, fairnessType);

                return (
                  <div key={index} className={clsx('fairness-metric-card', {
                    'biased': metric.is_biased,
                    'fair': !metric.is_biased,
                    [`level-${fairnessLevel.color}`]: true
                  })}>
                    <div className="metric-header">
                      <h5>{metric.metric_name.replace('Model-Based ', '').replace('Standard ', '')}</h5>
                      {metric.is_biased ? (
                        <AlertTriangle className="metric-status alert" />
                      ) : (
                        <CheckCircle className="metric-status good" />
                      )}
                    </div>

                    <div className="metric-value-display">
                      <div className="primary-value">
                        <span className="value">{metric.value.toFixed(4)}</span>
                        <span className="value-label">
                          {fairnessType === 'Disparate Impact' ? 'Ratio' : 'Difference'}
                        </span>
                      </div>
                      
                      <div className="threshold-comparison">
                        <span className="threshold-label">Threshold: {metric.threshold}</span>
                        <div className={clsx('threshold-status', fairnessLevel.color)}>
                          {fairnessLevel.level.toUpperCase()}
                        </div>
                      </div>
                    </div>

                    {fairnessType === 'Disparate Impact' && (
                      <div className="disparate-impact-visual">
                        <div className="impact-ratio-bar">
                          <div className="ratio-scale">
                            <span>0.5</span>
                            <span>0.8</span>
                            <span>1.0</span>
                            <span>1.25</span>
                          </div>
                          <div className="ratio-indicator" style={{ 
                            left: `${Math.min(Math.max((metric.value - 0.5) / 0.75 * 100, 0), 100)}%` 
                          }}>
                            <div className={clsx('ratio-marker', fairnessLevel.color)} />
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="fairness-interpretation">
                      <p>
                        {metric.is_biased ? (
                          <>
                            <strong>Fairness violation detected.</strong> The metric value of{' '}
                            <strong>{metric.value.toFixed(4)}</strong> exceeds the acceptable threshold,
                            indicating potential bias in model predictions.
                          </>
                        ) : (
                          <>
                            <strong>Fairness criteria met.</strong> The metric value of{' '}
                            <strong>{metric.value.toFixed(4)}</strong> is within acceptable bounds.
                          </>
                        )}
                      </p>
                    </div>

                    <div className="severity-indicator">
                      <span className={clsx('severity-badge', metric.severity)}>
                        {metric.severity.toUpperCase()} SEVERITY
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <div className="model-fairness-summary">
        <div className="summary-grid">
          <div className="summary-card">
            <h4>Total Fairness Metrics</h4>
            <span className="summary-value">{modelMetrics.length}</span>
          </div>
          <div className="summary-card">
            <h4>Failing Metrics</h4>
            <span className="summary-value biased">
              {modelMetrics.filter(m => m.is_biased).length}
            </span>
          </div>
          <div className="summary-card">
            <h4>Passing Metrics</h4>
            <span className="summary-value fair">
              {modelMetrics.filter(m => !m.is_biased).length}
            </span>
          </div>
          <div className="summary-card">
            <h4>Overall Fairness Rate</h4>
            <span className="summary-value">
              {((modelMetrics.filter(m => !m.is_biased).length / modelMetrics.length) * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="fairness-recommendation">
          {modelMetrics.some(m => m.is_biased) ? (
            <div className="recommendation-content warning">
              <AlertTriangle className="recommendation-icon" />
              <div>
                <h4>Fairness Issues Detected</h4>
                <p>
                  Your model shows bias across {modelMetrics.filter(m => m.is_biased).length} fairness metric(s).
                  Consider applying bias mitigation techniques or model retraining.
                </p>
              </div>
            </div>
          ) : (
            <div className="recommendation-content success">
              <CheckCircle className="recommendation-icon" />
              <div>
                <h4>Model Passes Fairness Checks</h4>
                <p>
                  All fairness metrics are within acceptable ranges. Continue monitoring
                  model performance and consider regular fairness audits.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelBasedAnalysis;
