import React from 'react';
import { Shield, AlertTriangle, CheckCircle, Eye } from 'lucide-react';
import { BiasMetric } from '../../types';
import clsx from 'clsx';

interface ProxyFeatureDetectionProps {
  metrics: BiasMetric[];
}

const ProxyFeatureDetection: React.FC<ProxyFeatureDetectionProps> = ({ metrics }) => {
  // Filter for proxy feature detection metrics
  const proxyMetrics = metrics.filter(metric => 
    metric.metric_name.includes('Proxy Feature') || 
    metric.metric_name.includes('proxy')
  );

  if (proxyMetrics.length === 0) {
    return null;
  }

  return (
    <div className="result-section">
      <div className="section-header">
        <Eye className="section-icon" />
        <h3>Proxy Feature Detection</h3>
        <p className="section-description">
          Analysis of features that might indirectly encode sensitive attribute information
        </p>
      </div>

      <div className="advanced-metrics-grid">
        {proxyMetrics.map((metric, index) => (
          <div key={index} className={clsx('advanced-metric-card proxy-card', {
            'biased': metric.is_biased,
            'fair': !metric.is_biased,
            [`severity-${metric.severity}`]: metric.severity
          })}>
            <div className="metric-header">
              <div className="metric-title">
                <Shield className="proxy-icon" />
                <h4>{metric.metric_name}</h4>
                {metric.is_biased ? (
                  <AlertTriangle className="metric-status alert" />
                ) : (
                  <CheckCircle className="metric-status good" />
                )}
              </div>
              <div className={clsx('proxy-indicator', metric.is_biased ? 'proxy-detected' : 'no-proxy')}>
                {metric.is_biased ? 'PROXY DETECTED' : 'NO PROXY'}
              </div>
            </div>

            <div className="metric-details">
              <div className="prediction-accuracy">
                <label>Prediction Accuracy</label>
                <div className="accuracy-display">
                  <span className="value">{(metric.value * 100).toFixed(2)}%</span>
                  <div className="accuracy-bar">
                    <div 
                      className="accuracy-fill" 
                      style={{ width: `${metric.value * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="proxy-threshold">
                <label>Baseline Threshold</label>
                <span className="threshold">{(metric.threshold * 100).toFixed(2)}%</span>
              </div>

              <div className="proxy-risk">
                <div className={clsx('risk-badge', {
                  'high-risk': metric.value > 0.8,
                  'medium-risk': metric.value > 0.65 && metric.value <= 0.8,
                  'low-risk': metric.value <= 0.65
                })}>
                  {metric.value > 0.8 ? 'HIGH PROXY RISK' :
                   metric.value > 0.65 ? 'MEDIUM PROXY RISK' :
                   'LOW PROXY RISK'}
                </div>
              </div>
            </div>

            <div className="metric-description">
              <p>{metric.description}</p>
            </div>

            <div className="proxy-explanation">
              <h5>What this means:</h5>
              <p>
                {metric.is_biased ? (
                  <>
                    This feature can predict the sensitive attribute with <strong>{(metric.value * 100).toFixed(1)}%</strong> accuracy, 
                    which is significantly above the baseline. This indicates the feature may serve as a 
                    <strong> proxy</strong> for the sensitive attribute, potentially enabling indirect discrimination.
                  </>
                ) : (
                  <>
                    This feature shows <strong>{(metric.value * 100).toFixed(1)}%</strong> prediction accuracy, 
                    which is within acceptable bounds. No significant proxy relationship detected with sensitive attributes.
                  </>
                )}
              </p>
            </div>

            {metric.is_biased && (
              <div className="mitigation-hint">
                <h5>Recommended Actions:</h5>
                <ul>
                  <li>Consider removing or transforming this feature</li>
                  <li>Apply feature anonymization techniques</li>
                  <li>Use fairness-aware feature selection</li>
                  <li>Monitor this feature in production</li>
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="proxy-summary">
        <div className="summary-alert">
          {proxyMetrics.some(m => m.is_biased) ? (
            <div className="alert-content warning">
              <AlertTriangle className="alert-icon" />
              <div>
                <h4>Proxy Features Detected</h4>
                <p>
                  {proxyMetrics.filter(m => m.is_biased).length} proxy feature(s) found that could enable 
                  indirect discrimination. Review and consider mitigation strategies.
                </p>
              </div>
            </div>
          ) : (
            <div className="alert-content success">
              <CheckCircle className="alert-icon" />
              <div>
                <h4>No Proxy Features Detected</h4>
                <p>
                  All analyzed features show acceptable prediction accuracy levels for sensitive attributes. 
                  No indirect bias pathways detected.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProxyFeatureDetection;
