import React from 'react';
import { AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';
import { BiasMetric } from '../../types';
import clsx from 'clsx';

interface DirectBiasAnalysisProps {
  metrics: BiasMetric[];
}

const DirectBiasAnalysis: React.FC<DirectBiasAnalysisProps> = ({ metrics }) => {
  // Filter for direct bias metrics
  const directBiasMetrics = metrics.filter(metric => 
    metric.metric_name.includes('Direct Bias') || 
    metric.metric_name.includes('Chi-square')
  );

  if (directBiasMetrics.length === 0) {
    return null;
  }

  return (
    <div className="result-section">
      <div className="section-header">
        <TrendingUp className="section-icon" />
        <h3>Direct Bias Analysis</h3>
        <p className="section-description">
          Statistical testing for direct relationships between sensitive attributes and target variables
        </p>
      </div>

      <div className="advanced-metrics-grid">
        {directBiasMetrics.map((metric, index) => (
          <div key={index} className={clsx('advanced-metric-card', {
            'biased': metric.is_biased,
            'fair': !metric.is_biased,
            [`severity-${metric.severity}`]: metric.severity
          })}>
            <div className="metric-header">
              <div className="metric-title">
                <h4>{metric.metric_name}</h4>
                {metric.is_biased ? (
                  <AlertTriangle className="metric-status alert" />
                ) : (
                  <CheckCircle className="metric-status good" />
                )}
              </div>
              <div className={clsx('bias-indicator', metric.is_biased ? 'biased' : 'fair')}>
                {metric.is_biased ? 'BIAS DETECTED' : 'NO BIAS'}
              </div>
            </div>

            <div className="metric-details">
              <div className="metric-value-section">
                <div className="metric-value">
                  <label>P-Value</label>
                  <span className="value">
                    {metric.value < 0.001 ? '< 0.001' : metric.value.toExponential(3)}
                  </span>
                </div>
                <div className="metric-threshold">
                  <label>Significance Threshold</label>
                  <span className="threshold">{metric.threshold}</span>
                </div>
              </div>

              <div className="statistical-significance">
                <div className={clsx('significance-badge', {
                  'highly-significant': metric.value < 0.001,
                  'significant': metric.value >= 0.001 && metric.value < 0.01,
                  'moderately-significant': metric.value >= 0.01 && metric.value < 0.05,
                  'not-significant': metric.value >= 0.05
                })}>
                  {metric.value < 0.001 ? 'Highly Significant (p < 0.001)' :
                   metric.value < 0.01 ? 'Significant (p < 0.01)' :
                   metric.value < 0.05 ? 'Moderately Significant (p < 0.05)' :
                   'Not Significant (p ≥ 0.05)'}
                </div>
              </div>

              <div className="severity-section">
                <span className={clsx('severity-badge', metric.severity)}>
                  {metric.severity.toUpperCase()} SEVERITY
                </span>
              </div>
            </div>

            <div className="metric-description">
              <p>{metric.description}</p>
            </div>

            <div className="interpretation">
              <h5>Interpretation:</h5>
              <p>
                {metric.is_biased ? (
                  <>
                    The chi-square test indicates a <strong>statistically significant</strong> relationship 
                    between this sensitive attribute and the target variable. This suggests direct bias 
                    where the model may be making decisions based on protected characteristics.
                  </>
                ) : (
                  <>
                    No statistically significant direct relationship found between this sensitive 
                    attribute and the target variable. This is a positive finding indicating no 
                    direct bias detected.
                  </>
                )}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="analysis-summary">
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Total Attributes Tested</span>
            <span className="stat-value">{directBiasMetrics.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Biased Attributes</span>
            <span className="stat-value biased">
              {directBiasMetrics.filter(m => m.is_biased).length}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Fair Attributes</span>
            <span className="stat-value fair">
              {directBiasMetrics.filter(m => !m.is_biased).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DirectBiasAnalysis;
