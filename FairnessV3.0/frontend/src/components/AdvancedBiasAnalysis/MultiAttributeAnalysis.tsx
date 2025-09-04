import React from 'react';
import { Users, Network, AlertTriangle, CheckCircle, Layers } from 'lucide-react';
import { BiasMetric } from '../../types';
import clsx from 'clsx';

interface MultiAttributeAnalysisProps {
  metrics: BiasMetric[];
}

const MultiAttributeAnalysis: React.FC<MultiAttributeAnalysisProps> = ({ metrics }) => {
  // Filter for multi-attribute metrics
  const multiAttrMetrics = metrics.filter(metric => 
    metric.metric_name.includes('Multi-Attribute') || 
    metric.metric_name.includes('Cross-Attribute')
  );

  if (multiAttrMetrics.length === 0) {
    return null;
  }

  // Separate multi-attribute summary metrics from cross-attribute interaction metrics
  const summaryMetrics = multiAttrMetrics.filter(m => m.metric_name.includes('Multi-Attribute'));
  const crossAttrMetrics = multiAttrMetrics.filter(m => m.metric_name.includes('Cross-Attribute'));

  return (
    <div className="result-section">
      <div className="section-header">
        <Network className="section-icon" />
        <h3>Multi-Sensitive Attribute Analysis</h3>
        <p className="section-description">
          Analysis of bias patterns across multiple sensitive attributes and their interactions
        </p>
      </div>

      {/* Summary Metrics */}
      {summaryMetrics.length > 0 && (
        <div className="multi-attr-summary">
          <div className="summary-header">
            <Users className="summary-icon" />
            <h4>Overall Multi-Attribute Summary</h4>
          </div>
          
          <div className="summary-metrics-grid">
            {summaryMetrics.map((metric, index) => (
              <div key={index} className={clsx('summary-metric-card', {
                'has-issues': metric.is_biased,
                'no-issues': !metric.is_biased,
                [`severity-${metric.severity}`]: metric.severity
              })}>
                <div className="summary-metric-header">
                  <h5>{metric.metric_name.replace('Multi-Attribute ', '')}</h5>
                  {metric.is_biased ? (
                    <AlertTriangle className="metric-status alert" />
                  ) : (
                    <CheckCircle className="metric-status good" />
                  )}
                </div>

                <div className="summary-metric-value">
                  <span className="value">{metric.value}</span>
                  <span className="value-label">
                    {metric.metric_name.includes('Count') ? 'attributes' : 'features'}
                  </span>
                </div>

                <div className="summary-interpretation">
                  <p>{metric.description}</p>
                </div>

                {metric.is_biased && (
                  <div className="summary-alert">
                    <span className={clsx('alert-badge', metric.severity)}>
                      {metric.severity.toUpperCase()} PRIORITY
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cross-Attribute Interactions */}
      {crossAttrMetrics.length > 0 && (
        <div className="cross-attr-analysis">
          <div className="cross-attr-header">
            <Layers className="cross-attr-icon" />
            <h4>Cross-Attribute Interactions</h4>
            <p>Analysis of bias in combinations of sensitive attributes</p>
          </div>

          <div className="cross-attr-grid">
            {crossAttrMetrics.map((metric, index) => {
              // Extract attribute names from metric name
              const attributePair = metric.metric_name
                .replace('Cross-Attribute Bias - ', '')
                .split('_x_');

              return (
                <div key={index} className={clsx('cross-attr-card', {
                  'interaction-bias': metric.is_biased,
                  'no-interaction-bias': !metric.is_biased,
                  [`severity-${metric.severity}`]: metric.severity
                })}>
                  <div className="interaction-header">
                    <div className="attribute-pair">
                      <span className="attribute">{attributePair[0]}</span>
                      <Network className="interaction-icon" />
                      <span className="attribute">{attributePair[1]}</span>
                    </div>
                    {metric.is_biased ? (
                      <AlertTriangle className="interaction-status alert" />
                    ) : (
                      <CheckCircle className="interaction-status good" />
                    )}
                  </div>

                  <div className="interaction-metrics">
                    <div className="interaction-value">
                      <label>P-Value</label>
                      <span className="value">
                        {metric.value < 0.001 ? '< 0.001' : metric.value.toExponential(3)}
                      </span>
                    </div>

                    <div className="interaction-significance">
                      <div className={clsx('significance-badge', {
                        'highly-significant': metric.value < 0.001,
                        'significant': metric.value >= 0.001 && metric.value < 0.01,
                        'moderately-significant': metric.value >= 0.01 && metric.value < 0.05,
                        'not-significant': metric.value >= 0.05
                      })}>
                        {metric.value < 0.001 ? 'Highly Significant' :
                         metric.value < 0.01 ? 'Significant' :
                         metric.value < 0.05 ? 'Moderately Significant' :
                         'Not Significant'}
                      </div>
                    </div>
                  </div>

                  <div className="interaction-description">
                    <p>{metric.description}</p>
                  </div>

                  <div className="interaction-explanation">
                    <h6>Interaction Effect:</h6>
                    <p>
                      {metric.is_biased ? (
                        <>
                          A significant interaction effect was detected between{' '}
                          <strong>{attributePair[0]}</strong> and <strong>{attributePair[1]}</strong>.
                          This suggests that bias patterns change when these attributes are combined,
                          indicating intersectional bias effects.
                        </>
                      ) : (
                        <>
                          No significant interaction effect detected between these attributes.
                          The bias patterns for each attribute appear to be independent.
                        </>
                      )}
                    </p>
                  </div>

                  {metric.is_biased && (
                    <div className="interaction-implications">
                      <h6>Implications:</h6>
                      <ul>
                        <li>Individuals with both attributes may face compound discrimination</li>
                        <li>Separate mitigation strategies may be needed for this combination</li>
                        <li>Monitor this intersection closely in production</li>
                      </ul>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Overall Multi-Attribute Assessment */}
      <div className="multi-attr-assessment">
        <div className="assessment-content">
          {multiAttrMetrics.some(m => m.is_biased) ? (
            <div className="assessment-alert warning">
              <AlertTriangle className="assessment-icon" />
              <div className="assessment-text">
                <h4>Complex Bias Patterns Detected</h4>
                <p>
                  Your model shows bias across multiple sensitive attributes or their interactions.
                  This indicates complex discrimination patterns that may require sophisticated
                  mitigation strategies addressing intersectional bias.
                </p>
                <div className="assessment-stats">
                  <span>
                    Issues found in {multiAttrMetrics.filter(m => m.is_biased).length} of{' '}
                    {multiAttrMetrics.length} multi-attribute analyses
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="assessment-alert success">
              <CheckCircle className="assessment-icon" />
              <div className="assessment-text">
                <h4>No Multi-Attribute Bias Detected</h4>
                <p>
                  Your model shows fair treatment across multiple sensitive attributes and their
                  interactions. This is an excellent result indicating good intersectional fairness.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="assessment-recommendations">
          <h5>Recommendations:</h5>
          <ul>
            {multiAttrMetrics.some(m => m.is_biased) ? (
              <>
                <li>Consider intersectional fairness mitigation techniques</li>
                <li>Analyze subgroup-specific performance metrics</li>
                <li>Implement monitoring for compound discrimination</li>
                <li>Review data collection for underrepresented intersections</li>
              </>
            ) : (
              <>
                <li>Continue monitoring intersectional fairness in production</li>
                <li>Maintain current data collection and training practices</li>
                <li>Consider this model as a fairness best practice example</li>
                <li>Document successful practices for future model development</li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MultiAttributeAnalysis;
