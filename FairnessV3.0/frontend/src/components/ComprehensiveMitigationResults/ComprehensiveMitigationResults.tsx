import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Progress, 
  Badge, 
  Button, 
  Tabs, 
  Statistic, 
  Row, 
  Col, 
  Alert,
  Spin,
  Typography,
  Divider
} from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  TrophyOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  RiseOutlined
} from '@ant-design/icons';
import './ComprehensiveMitigationResults.css';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface BiasMetric {
  [key: string]: number;
}

interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

interface MitigationStrategy {
  strategy: string;
  strategy_type: string;
  metrics: BiasMetric;
  fairness_score: number;
  model_performance: ModelPerformance;
  execution_time: number;
}

interface ExecutionSummary {
  total_strategies: number;
  successful_strategies: number;
  failed_strategies: number;
  success_rate: number;
  total_execution_time: number;
  strategy_breakdown: {
    preprocessing: number;
    inprocessing: number;
    postprocessing: number;
  };
}

interface ComprehensiveMitigationData {
  job_id: string;
  bias_before: BiasMetric;
  bias_after: MitigationStrategy[];
  best_strategy: string;
  improvements: BiasMetric;
  overall_fairness_improvement: number;
  execution_summary: ExecutionSummary;
  recommendations: string[];
  strategies_applied: number;
  successful_strategies: number;
}

interface ComprehensiveMitigationResultsProps {
  jobId: string;
  onBack?: () => void;
}

const ComprehensiveMitigationResults: React.FC<ComprehensiveMitigationResultsProps> = ({ 
  jobId, 
  onBack 
}) => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<ComprehensiveMitigationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchResults();
  }, [jobId]);

  const fetchResults = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/mitigation/comprehensive/results/${jobId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch results: ${response.statusText}`);
      }
      
      const resultsData = await response.json();
      setData(resultsData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  const getStrategyTypeColor = (type: string) => {
    switch (type) {
      case 'preprocessing': return 'blue';
      case 'inprocessing': return 'green';
      case 'postprocessing': return 'orange';
      default: return 'default';
    }
  };

  const getFairnessLevel = (score: number) => {
    if (score >= 80) return { level: 'Excellent', color: 'success' };
    if (score >= 60) return { level: 'Good', color: 'warning' };
    if (score >= 40) return { level: 'Fair', color: 'error' };
    return { level: 'Poor', color: 'error' };
  };

  const formatMetricName = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const renderOverviewTab = () => (
    <div className="overview-tab">
      {/* Summary Statistics */}
      <Row gutter={[16, 16]} className="summary-stats">
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="Strategies Applied"
            value={data!.strategies_applied}
            prefix={<ThunderboltOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="Success Rate"
            value={((data!.successful_strategies / data!.strategies_applied) * 100).toFixed(1)}
            suffix="%"
            prefix={<CheckCircleOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="Best Fairness Score"
            value={data!.overall_fairness_improvement.toFixed(1)}
            prefix={<TrophyOutlined />}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="Execution Time"
            value={data!.execution_summary.total_execution_time.toFixed(1)}
            suffix="s"
            prefix={<ClockCircleOutlined />}
          />
        </Col>
      </Row>

      {/* Best Strategy Highlight */}
      <Card className="best-strategy-card" bordered={false}>
        <div className="best-strategy-header">
          <TrophyOutlined className="trophy-icon" />
          <Title level={4}>Best Performing Strategy</Title>
        </div>
        <div className="best-strategy-content">
          <Text strong className="strategy-name">{data!.best_strategy}</Text>
          <div className="strategy-details">
            {data!.bias_after.find(s => s.strategy === data!.best_strategy) && (
              <>
                <Badge 
                  color={getStrategyTypeColor(data!.bias_after.find(s => s.strategy === data!.best_strategy)!.strategy_type)}
                  text={data!.bias_after.find(s => s.strategy === data!.best_strategy)!.strategy_type}
                />
                <Divider type="vertical" />
                <Text>Fairness Score: {data!.bias_after.find(s => s.strategy === data!.best_strategy)!.fairness_score.toFixed(1)}</Text>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* Execution Summary */}
      <Card title="Execution Summary" className="execution-summary">
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <div className="strategy-breakdown">
              <Title level={5}>Strategy Types Applied</Title>
              <div className="breakdown-item">
                <Badge color="blue" />
                <Text>Preprocessing: {data!.execution_summary.strategy_breakdown.preprocessing}</Text>
              </div>
              <div className="breakdown-item">
                <Badge color="green" />
                <Text>In-processing: {data!.execution_summary.strategy_breakdown.inprocessing}</Text>
              </div>
              <div className="breakdown-item">
                <Badge color="orange" />
                <Text>Post-processing: {data!.execution_summary.strategy_breakdown.postprocessing}</Text>
              </div>
            </div>
          </Col>
          <Col xs={24} md={12}>
            <div className="success-rate-visual">
              <Title level={5}>Success Rate</Title>
              <Progress
                type="circle"
                percent={data!.execution_summary.success_rate}
                format={percent => `${percent}%`}
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );

  const renderComparisonTab = () => {
    const columns = [
      {
        title: 'Strategy',
        dataIndex: 'strategy',
        key: 'strategy',
        render: (text: string, record: MitigationStrategy) => (
          <div>
            <div className="strategy-name">{text}</div>
            <Badge color={getStrategyTypeColor(record.strategy_type)} text={record.strategy_type} />
          </div>
        ),
      },
      {
        title: 'Fairness Score',
        dataIndex: 'fairness_score',
        key: 'fairness_score',
        render: (score: number) => {
          const level = getFairnessLevel(score);
          return (
            <div>
              <div className="fairness-score">{score.toFixed(1)}</div>
              <Badge status={level.color as any} text={level.level} />
            </div>
          );
        },
        sorter: (a: MitigationStrategy, b: MitigationStrategy) => a.fairness_score - b.fairness_score,
        defaultSortOrder: 'descend' as const,
      },
      {
        title: 'Accuracy',
        dataIndex: ['model_performance', 'accuracy'],
        key: 'accuracy',
        render: (accuracy: number) => (accuracy * 100).toFixed(1) + '%',
        sorter: (a: MitigationStrategy, b: MitigationStrategy) => 
          a.model_performance.accuracy - b.model_performance.accuracy,
      },
      {
        title: 'Execution Time',
        dataIndex: 'execution_time',
        key: 'execution_time',
        render: (time: number) => `${time.toFixed(2)}s`,
        sorter: (a: MitigationStrategy, b: MitigationStrategy) => a.execution_time - b.execution_time,
      },
      {
        title: 'Best Strategy',
        key: 'is_best',
        render: (_: any, record: MitigationStrategy) => 
          record.strategy === data!.best_strategy ? (
            <TrophyOutlined style={{ color: '#faad14' }} />
          ) : null,
      },
    ];

    return (
      <div className="comparison-tab">
        <Table
          columns={columns}
          dataSource={data!.bias_after}
          rowKey="strategy"
          pagination={false}
          className="strategies-table"
        />
      </div>
    );
  };

  const renderMetricsTab = () => (
    <div className="metrics-tab">
      {/* Before/After Comparison */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Bias Metrics Before Mitigation" className="metrics-card">
            {Object.entries(data!.bias_before).map(([metric, value]) => (
              <div key={metric} className="metric-item">
                <Text>{formatMetricName(metric)}</Text>
                <Text strong>{value.toFixed(4)}</Text>
              </div>
            ))}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Improvements Achieved" className="metrics-card">
            {Object.entries(data!.improvements).map(([metric, improvement]) => (
              <div key={metric} className="metric-item">
                <Text>{formatMetricName(metric)}</Text>
                <div className="improvement-value">
                  {improvement > 0 ? (
                    <Text type="success">
                      <RiseOutlined /> +{improvement.toFixed(1)}%
                    </Text>
                  ) : (
                    <Text type="danger">
                      {improvement.toFixed(1)}%
                    </Text>
                  )}
                </div>
              </div>
            ))}
          </Card>
        </Col>
      </Row>

      {/* Detailed Metrics for Best Strategy */}
      {data!.bias_after.find(s => s.strategy === data!.best_strategy) && (
        <Card title={`Detailed Metrics - ${data!.best_strategy}`} className="best-strategy-metrics">
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Title level={5}>Bias Metrics</Title>
              {Object.entries(
                data!.bias_after.find(s => s.strategy === data!.best_strategy)!.metrics
              ).map(([metric, value]) => (
                <div key={metric} className="metric-detail">
                  <Text>{formatMetricName(metric)}</Text>
                  <Text strong>{value.toFixed(4)}</Text>
                </div>
              ))}
            </Col>
            <Col xs={24} md={12}>
              <Title level={5}>Model Performance</Title>
              {Object.entries(
                data!.bias_after.find(s => s.strategy === data!.best_strategy)!.model_performance
              ).map(([metric, value]) => (
                <div key={metric} className="metric-detail">
                  <Text>{formatMetricName(metric)}</Text>
                  <Text strong>{(value * 100).toFixed(2)}%</Text>
                </div>
              ))}
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );

  const renderRecommendationsTab = () => (
    <div className="recommendations-tab">
      <Card className="recommendations-card">
        <div className="recommendations-header">
          <BulbOutlined className="bulb-icon" />
          <Title level={4}>AI-Powered Recommendations</Title>
        </div>
        
        <div className="recommendations-list">
          {data!.recommendations.map((recommendation, index) => (
            <Alert
              key={index}
              message={`Recommendation ${index + 1}`}
              description={recommendation}
              type="info"
              showIcon
              className="recommendation-item"
            />
          ))}
        </div>

        {data!.recommendations.length === 0 && (
          <Alert
            message="No specific recommendations available"
            description="The mitigation analysis completed successfully, but no additional recommendations were generated."
            type="warning"
            showIcon
          />
        )}
      </Card>
    </div>
  );

  if (loading) {
    return (
      <div className="loading-container">
        <Spin size="large" />
        <Text>Loading comprehensive mitigation results...</Text>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="Error Loading Results"
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" onClick={fetchResults}>
            Retry
          </Button>
        }
      />
    );
  }

  if (!data) {
    return (
      <Alert
        message="No Results Available"
        description="No mitigation results found for this job."
        type="warning"
        showIcon
      />
    );
  }

  return (
    <div className="comprehensive-mitigation-results">
      <div className="results-header">
        <div className="header-content">
          <Title level={2}>
            <BarChartOutlined /> Comprehensive Bias Mitigation Results
          </Title>
          <Paragraph>
            Automatic evaluation of {data.strategies_applied} bias mitigation strategies
          </Paragraph>
        </div>
        {onBack && (
          <Button onClick={onBack} className="back-button">
            Back to Analysis
          </Button>
        )}
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab} className="results-tabs">
        <TabPane tab="Overview" key="overview">
          {renderOverviewTab()}
        </TabPane>
        <TabPane tab="Strategy Comparison" key="comparison">
          {renderComparisonTab()}
        </TabPane>
        <TabPane tab="Bias Metrics" key="metrics">
          {renderMetricsTab()}
        </TabPane>
        <TabPane tab="Recommendations" key="recommendations">
          {renderRecommendationsTab()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ComprehensiveMitigationResults;
