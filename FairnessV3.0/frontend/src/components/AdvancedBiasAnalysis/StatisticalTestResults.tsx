import React from 'react';
import { BarChart3, TrendingUp, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { BiasMetric } from '../../types';
import clsx from 'clsx';

interface StatisticalTestResultsProps {
  metrics: BiasMetric[];
}

const StatisticalTestResults: React.FC<StatisticalTestResultsProps> = ({ metrics }) => {
  // Filter for statistical test metrics
  const statisticalMetrics = metrics.filter(metric => 
    metric.metric_name.includes('Statistical Test') ||
    metric.metric_name.includes('ks_test') ||
    metric.metric_name.includes('mannwhitney') ||
    metric.metric_name.includes('correlation') ||
    metric.metric_name.includes('chi2_test') ||
    metric.metric_name.includes('mutual_info')
  );

  if (statisticalMetrics.length === 0) {
    return null;
  }

  // Group metrics by test type
  const testTypes = {
    'Kolmogorov-Smirnov': statisticalMetrics.filter(m => m.metric_name.includes('ks_test')),
    'Mann-Whitney U': statisticalMetrics.filter(m => m.metric_name.includes('mannwhitney')),
    'Correlation': statisticalMetrics.filter(m => m.metric_name.includes('correlation')),
    'Chi-square': statisticalMetrics.filter(m => m.metric_name.includes('chi2_test')),
    'Mutual Information': statisticalMetrics.filter(m => m.metric_name.includes('mutual_info')),
    'Other': statisticalMetrics.filter(m => 
      !m.metric_name.includes('ks_test') &&
      !m.metric_name.includes('mannwhitney') &&
      !m.metric_name.includes('correlation') &&
      !m.metric_name.includes('chi2_test') &&
      !m.metric_name.includes('mutual_info')
    )
  };

  const getTestDescription = (testType: string) => {
    switch (testType) {
      case 'Kolmogorov-Smirnov':
        return 'Tests for differences in distributions between groups for numerical features';
      case 'Mann-Whitney U':
        return 'Non-parametric test for differences between groups (does not assume normal distribution)';
      case 'Correlation':
        return 'Measures linear relationships between features and sensitive attributes';
      case 'Chi-square':
        return 'Tests independence between categorical features and sensitive attributes';
      case 'Mutual Information':
        return 'Measures non-linear dependencies between features and sensitive attributes';
      default:
        return 'Additional statistical tests for bias detection';
    }
  };

  const formatPValue = (value: number) => {
    if (value < 0.001) return '< 0.001';
    if (value < 0.01) return value.toFixed(4);
    return value.toFixed(3);
  };

  const getSignificanceLevel = (value: number, isCorrelation: boolean = false) => {
    if (isCorrelation) {
      const absValue = Math.abs(value);
      if (absValue > 0.7) return { level: 'strong', color: 'high' };
      if (absValue > 0.3) return { level: 'moderate', color: 'medium' };
      if (absValue > 0.1) return { level: 'weak', color: 'low' };
      return { level: 'negligible', color: 'none' };
    } else {
      if (value < 0.001) return { level: 'highly significant', color: 'high' };
      if (value < 0.01) return { level: 'significant', color: 'medium' };
      if (value < 0.05) return { level: 'marginally significant', color: 'low' };
      return { level: 'not significant', color: 'none' };
    }
  };

  return (
    <div className="result-section">
      <div className="section-header">
        <Activity className="section-icon" />
        <h3>Comprehensive Statistical Testing</h3>
        <p className="section-description">
          Multiple statistical tests to detect various forms of bias and dependencies
        </p>
      </div>

      {Object.entries(testTypes).map(([testType, testMetrics]) => {
        if (testMetrics.length === 0) return null;

        return (
          <div key={testType} className="statistical-test-section">
            <div className="test-type-header">
              <BarChart3 className="test-icon" />
              <div className="test-info">
                <h4>{testType} Test Results</h4>
                <p>{getTestDescription(testType)}</p>
              </div>
              <div className="test-summary">
                <span className="test-count">{testMetrics.length} tests</span>
                <span className={clsx('significant-count', {
                  'has-significant': testMetrics.some(m => m.is_biased)
                })}>
                  {testMetrics.filter(m => m.is_biased).length} significant
                </span>
              </div>
            </div>

            <div className="statistical-tests-grid">
              {testMetrics.map((metric, index) => {
                const isCorrelation = testType === 'Correlation';
                const isMutualInfo = testType === 'Mutual Information';
                const significance = getSignificanceLevel(metric.value, isCorrelation);

                return (
                  <div key={index} className={clsx('statistical-test-card', {
                    'significant': metric.is_biased,
                    'not-significant': !metric.is_biased,
                    [`significance-${significance.color}`]: true
                  })}>
                    <div className="test-header">
                      <h5>{metric.metric_name.replace('Statistical Test ', '').replace(/\([^)]*\) - /, '')}</h5>
                      {metric.is_biased ? (
                        <AlertTriangle className="test-status alert" />
                      ) : (
                        <CheckCircle className="test-status good" />
                      )}
                    </div>

                    <div className="test-values">
                      <div className="test-value">
                        <label>
                          {isCorrelation ? 'Correlation Coefficient' :
                           isMutualInfo ? 'Mutual Information Score' :
                           'P-Value'}
                        </label>
                        <span className="value">
                          {isCorrelation || isMutualInfo ? 
                            metric.value.toFixed(4) : 
                            formatPValue(metric.value)
                          }
                        </span>
                      </div>

                      <div className="test-threshold">
                        <label>Threshold</label>
                        <span className="threshold">
                          {isCorrelation || isMutualInfo ? 
                            metric.threshold.toFixed(3) : 
                            metric.threshold
                          }
                        </span>
                      </div>
                    </div>

                    <div className="significance-indicator">
                      <div className={clsx('significance-badge', significance.color)}>
                        {significance.level.toUpperCase()}
                      </div>
                    </div>

                    {isCorrelation && (
                      <div className="correlation-visualization">
                        <div className="correlation-bar">
                          <div 
                            className={clsx('correlation-fill', {
                              'positive': metric.value > 0,
                              'negative': metric.value < 0
                            })}
                            style={{ 
                              width: `${Math.abs(metric.value) * 100}%`,
                              marginLeft: metric.value < 0 ? `${50 - Math.abs(metric.value) * 50}%` : '50%'
                            }}
                          />
                        </div>
                        <div className="correlation-labels">
                          <span>-1</span>
                          <span>0</span>
                          <span>+1</span>
                        </div>
                      </div>
                    )}

                    <div className="test-interpretation">
                      <p>
                        {metric.is_biased ? (
                          <>
                            <strong>Significant relationship detected.</strong> This test indicates 
                            a statistically significant dependency between the feature and sensitive attribute.
                          </>
                        ) : (
                          <>
                            <strong>No significant relationship.</strong> This test shows no 
                            statistically significant dependency.
                          </>
                        )}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <div className="statistical-summary">
        <div className="summary-stats">
          <div className="stat-card">
            <h4>Total Tests Performed</h4>
            <span className="stat-number">{statisticalMetrics.length}</span>
          </div>
          <div className="stat-card">
            <h4>Significant Results</h4>
            <span className="stat-number biased">
              {statisticalMetrics.filter(m => m.is_biased).length}
            </span>
          </div>
          <div className="stat-card">
            <h4>Non-Significant Results</h4>
            <span className="stat-number fair">
              {statisticalMetrics.filter(m => !m.is_biased).length}
            </span>
          </div>
          <div className="stat-card">
            <h4>Significance Rate</h4>
            <span className="stat-number">
              {((statisticalMetrics.filter(m => m.is_biased).length / statisticalMetrics.length) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatisticalTestResults;
