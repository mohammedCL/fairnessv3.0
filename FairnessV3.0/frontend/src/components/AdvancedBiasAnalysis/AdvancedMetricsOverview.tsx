import React from 'react';
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { BiasMetric } from '../../types';
import { PieChart as RechartsPieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import clsx from 'clsx';

interface AdvancedMetricsOverviewProps {
  metrics: BiasMetric[];
}

const AdvancedMetricsOverview: React.FC<AdvancedMetricsOverviewProps> = ({ metrics }) => {
  // Categorize metrics by analysis type
  const metricCategories = {
    'Direct Bias': metrics.filter(m => m.metric_name.includes('Direct Bias') || m.metric_name.includes('Chi-square')),
    'Proxy Features': metrics.filter(m => m.metric_name.includes('Proxy Feature')),
    'Statistical Tests': metrics.filter(m => m.metric_name.includes('Statistical Test')),
    'Model-Based': metrics.filter(m => m.metric_name.includes('Model-Based') || m.metric_name.includes('Standard')),
    'Multi-Attribute': metrics.filter(m => m.metric_name.includes('Multi-Attribute') || m.metric_name.includes('Cross-Attribute'))
  };

  // Calculate overall statistics
  const totalMetrics = metrics.length;
  const biasedMetrics = metrics.filter(m => m.is_biased).length;
  const fairMetrics = totalMetrics - biasedMetrics;
  const biasPercentage = totalMetrics > 0 ? (biasedMetrics / totalMetrics) * 100 : 0;

  // Severity distribution
  const severityDistribution = {
    high: metrics.filter(m => m.severity === 'high').length,
    medium: metrics.filter(m => m.severity === 'medium').length,
    low: metrics.filter(m => m.severity === 'low').length
  };

  // Data for charts
  const biasDistributionData = [
    { name: 'Biased', value: biasedMetrics, color: '#ef4444' },
    { name: 'Fair', value: fairMetrics, color: '#22c55e' }
  ];

  const categoryData = Object.entries(metricCategories).map(([category, categoryMetrics]) => ({
    category,
    total: categoryMetrics.length,
    biased: categoryMetrics.filter(m => m.is_biased).length,
    fair: categoryMetrics.filter(m => !m.is_biased).length,
    biasRate: categoryMetrics.length > 0 ? (categoryMetrics.filter(m => m.is_biased).length / categoryMetrics.length) * 100 : 0
  }));

  const severityData = [
    { name: 'High', value: severityDistribution.high, color: '#dc2626' },
    { name: 'Medium', value: severityDistribution.medium, color: '#f59e0b' },
    { name: 'Low', value: severityDistribution.low, color: '#10b981' }
  ];

  const getOverallAssessment = () => {
    if (biasPercentage === 0) return { level: 'excellent', color: 'success', message: 'Excellent fairness - no bias detected' };
    if (biasPercentage <= 10) return { level: 'good', color: 'success', message: 'Good fairness - minimal bias detected' };
    if (biasPercentage <= 25) return { level: 'acceptable', color: 'warning', message: 'Acceptable fairness - some bias present' };
    if (biasPercentage <= 50) return { level: 'concerning', color: 'warning', message: 'Concerning bias levels detected' };
    return { level: 'poor', color: 'danger', message: 'Poor fairness - significant bias detected' };
  };

  const assessment = getOverallAssessment();

  if (totalMetrics === 0) {
    return (
      <div className="result-section">
        <div className="section-header">
          <Info className="section-icon" />
          <h3>Advanced Metrics Overview</h3>
        </div>
        <div className="no-data">
          <Info className="no-data-icon" />
          <p>No advanced bias metrics available to display.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="result-section">
      <div className="section-header">
        <BarChart3 className="section-icon" />
        <h3>Advanced Bias Analysis Overview</h3>
        <p className="section-description">
          Comprehensive summary of all bias detection methods and results
        </p>
      </div>

      {/* Overall Assessment Card */}
      <div className="overview-assessment">
        <div className={clsx('assessment-card', assessment.color)}>
          <div className="assessment-header">
            {assessment.color === 'success' ? (
              <CheckCircle className="assessment-icon" />
            ) : (
              <AlertTriangle className="assessment-icon" />
            )}
            <h4>Overall Fairness Assessment</h4>
          </div>
          <div className="assessment-content">
            <div className="assessment-score">
              <span className="score-value">{(100 - biasPercentage).toFixed(1)}%</span>
              <span className="score-label">Fairness Score</span>
            </div>
            <div className="assessment-message">
              <p>{assessment.message}</p>
              <div className="assessment-details">
                <span>{biasedMetrics} of {totalMetrics} metrics indicate bias</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="overview-stats-grid">
        <div className="stat-card total">
          <div className="stat-icon">
            <BarChart3 />
          </div>
          <div className="stat-content">
            <h4>Total Metrics</h4>
            <span className="stat-value">{totalMetrics}</span>
            <span className="stat-description">Comprehensive analysis</span>
          </div>
        </div>

        <div className="stat-card biased">
          <div className="stat-icon">
            <AlertTriangle />
          </div>
          <div className="stat-content">
            <h4>Biased Metrics</h4>
            <span className="stat-value">{biasedMetrics}</span>
            <span className="stat-description">{biasPercentage.toFixed(1)}% of total</span>
          </div>
        </div>

        <div className="stat-card fair">
          <div className="stat-icon">
            <CheckCircle />
          </div>
          <div className="stat-content">
            <h4>Fair Metrics</h4>
            <span className="stat-value">{fairMetrics}</span>
            <span className="stat-description">{(100 - biasPercentage).toFixed(1)}% of total</span>
          </div>
        </div>

        <div className="stat-card severity">
          <div className="stat-icon">
            <TrendingUp />
          </div>
          <div className="stat-content">
            <h4>High Severity</h4>
            <span className="stat-value">{severityDistribution.high}</span>
            <span className="stat-description">Requires immediate attention</span>
          </div>
        </div>
      </div>

      {/* Visualization Section */}
      <div className="overview-charts">
        <div className="chart-row">
          {/* Bias Distribution Pie Chart */}
          <div className="chart-container">
            <h5>Overall Bias Distribution</h5>
            <ResponsiveContainer width="100%" height={200}>
              <RechartsPieChart>
                <Pie
                  data={biasDistributionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  dataKey="value"
                >
                  {biasDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </RechartsPieChart>
            </ResponsiveContainer>
            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#ef4444' }} />
                <span>Biased ({biasedMetrics})</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#22c55e' }} />
                <span>Fair ({fairMetrics})</span>
              </div>
            </div>
          </div>

          {/* Severity Distribution */}
          <div className="chart-container">
            <h5>Severity Distribution</h5>
            <ResponsiveContainer width="100%" height={200}>
              <RechartsPieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  dataKey="value"
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </RechartsPieChart>
            </ResponsiveContainer>
            <div className="chart-legend">
              {severityData.map(item => (
                <div key={item.name} className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: item.color }} />
                  <span>{item.name} ({item.value})</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Category Analysis Bar Chart */}
        <div className="chart-container full-width">
          <h5>Bias by Analysis Category</h5>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="biased" fill="#ef4444" name="Biased" />
              <Bar dataKey="fair" fill="#22c55e" name="Fair" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="category-breakdown">
        <h4>Analysis Categories Breakdown</h4>
        <div className="category-grid">
          {Object.entries(metricCategories).map(([category, categoryMetrics]) => {
            if (categoryMetrics.length === 0) return null;
            
            const categoryBiasRate = (categoryMetrics.filter(m => m.is_biased).length / categoryMetrics.length) * 100;
            
            return (
              <div key={category} className={clsx('category-card', {
                'has-bias': categoryMetrics.some(m => m.is_biased),
                'no-bias': !categoryMetrics.some(m => m.is_biased)
              })}>
                <div className="category-header">
                  <h5>{category}</h5>
                  <span className="category-count">{categoryMetrics.length} metrics</span>
                </div>
                
                <div className="category-stats">
                  <div className="category-bias-rate">
                    <span className="bias-rate">{categoryBiasRate.toFixed(1)}%</span>
                    <span className="bias-rate-label">bias rate</span>
                  </div>
                  
                  <div className="category-breakdown-numbers">
                    <span className="biased-count">{categoryMetrics.filter(m => m.is_biased).length} biased</span>
                    <span className="fair-count">{categoryMetrics.filter(m => !m.is_biased).length} fair</span>
                  </div>
                </div>

                <div className="category-progress">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill biased" 
                      style={{ width: `${categoryBiasRate}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Key Insights */}
      <div className="overview-insights">
        <h4>Key Insights</h4>
        <div className="insights-grid">
          <div className="insight-card">
            <h5>Most Problematic Category</h5>
            <p>
              {(() => {
                const mostProblematic = categoryData.reduce((max, current) => 
                  current.biasRate > max.biasRate ? current : max
                );
                return `${mostProblematic.category} (${mostProblematic.biasRate.toFixed(1)}% bias rate)`;
              })()}
            </p>
          </div>

          <div className="insight-card">
            <h5>Best Performing Category</h5>
            <p>
              {(() => {
                const bestPerforming = categoryData.reduce((min, current) => 
                  current.biasRate < min.biasRate ? current : min
                );
                return `${bestPerforming.category} (${bestPerforming.biasRate.toFixed(1)}% bias rate)`;
              })()}
            </p>
          </div>

          <div className="insight-card">
            <h5>Immediate Attention Required</h5>
            <p>
              {severityDistribution.high > 0 
                ? `${severityDistribution.high} high-severity issues need immediate attention`
                : 'No high-severity issues detected'
              }
            </p>
          </div>

          <div className="insight-card">
            <h5>Overall Model Status</h5>
            <p>
              {biasPercentage <= 10 
                ? 'Model shows excellent fairness characteristics'
                : biasPercentage <= 25
                ? 'Model has acceptable fairness with room for improvement'
                : 'Model requires significant bias mitigation efforts'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedMetricsOverview;
