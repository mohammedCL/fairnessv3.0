import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Steps, 
  Alert, 
  Spin, 
  Typography, 
  Row, 
  Col,
  Progress,
  Modal,
  List,
  Badge
} from 'antd';
import { 
  PlayCircleOutlined, 
  CheckCircleOutlined, 
  LoadingOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  TrophyOutlined
} from '@ant-design/icons';
import ComprehensiveMitigationResults from '../ComprehensiveMitigationResults/ComprehensiveMitigationResults';
import './BiasManagement.css';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;

interface BiasManagementProps {
  analysisJobId: string;
  onBack?: () => void;
}

interface MitigationJob {
  job_id: string;
  analysis_job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

const BiasManagement: React.FC<BiasManagementProps> = ({ analysisJobId, onBack }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [mitigationJob, setMitigationJob] = useState<MitigationJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [showStrategyModal, setShowStrategyModal] = useState(false);

  const strategies = [
    {
      key: 'reweighing',
      name: 'Data Reweighing',
      type: 'preprocessing',
      description: 'Reweight training samples to balance sensitive groups',
      ideal_for: 'Datasets with imbalanced sensitive groups'
    },
    {
      key: 'disparate_impact_remover',
      name: 'Disparate Impact Remover',
      type: 'preprocessing',
      description: 'Remove features that cause disparate impact',
      ideal_for: 'When direct discrimination through features is suspected'
    },
    {
      key: 'data_augmentation',
      name: 'Data Augmentation',
      type: 'preprocessing',
      description: 'Augment underrepresented groups with synthetic data',
      ideal_for: 'Small datasets with underrepresented groups'
    },
    {
      key: 'fairness_regularization',
      name: 'Fairness Regularization',
      type: 'inprocessing',
      description: 'Add fairness penalty to model training',
      ideal_for: 'When you can retrain the model with fairness constraints'
    },
    {
      key: 'adversarial_debiasing',
      name: 'Adversarial Debiasing',
      type: 'inprocessing',
      description: 'Use adversarial training to remove bias',
      ideal_for: 'Complex models where traditional methods fail'
    },
    {
      key: 'threshold_optimization',
      name: 'Threshold Optimization',
      type: 'postprocessing',
      description: 'Optimize decision thresholds for each group',
      ideal_for: 'When model retraining is not possible'
    },
    {
      key: 'calibration_adjustment',
      name: 'Calibration Adjustment',
      type: 'postprocessing',
      description: 'Calibrate predictions to ensure fairness',
      ideal_for: 'Probability-based models with calibration issues'
    },
    {
      key: 'equalized_odds_postprocessing',
      name: 'Equalized Odds Post-processing',
      type: 'postprocessing',
      description: 'Adjust predictions to achieve equalized odds',
      ideal_for: 'When equalized odds is the primary fairness goal'
    }
  ];

  const getStrategyTypeColor = (type: string) => {
    switch (type) {
      case 'preprocessing': return 'blue';
      case 'inprocessing': return 'green';
      case 'postprocessing': return 'orange';
      default: return 'default';
    }
  };

  const startComprehensiveMitigation = async () => {
    try {
      setLoading(true);
      setError(null);
      setCurrentStep(1);

      const response = await fetch('/api/mitigation/comprehensive/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ analysis_job_id: analysisJobId }),
      });

      if (!response.ok) {
        throw new Error(`Failed to start mitigation: ${response.statusText}`);
      }

      const job = await response.json();
      setMitigationJob(job);
      setCurrentStep(2);

      // Poll for progress updates
      pollJobProgress(job.job_id);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start mitigation');
      setCurrentStep(0);
    } finally {
      setLoading(false);
    }
  };

  const pollJobProgress = async (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/mitigation/comprehensive/job/${jobId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch job status');
        }

        const jobStatus = await response.json();
        setMitigationJob(jobStatus);

        if (jobStatus.status === 'completed') {
          clearInterval(pollInterval);
          setCurrentStep(3);
        } else if (jobStatus.status === 'failed') {
          clearInterval(pollInterval);
          setError(jobStatus.error_message || 'Mitigation failed');
          setCurrentStep(0);
        }
      } catch (err) {
        clearInterval(pollInterval);
        setError('Failed to fetch job progress');
        setCurrentStep(0);
      }
    }, 2000);

    // Clear interval after 5 minutes to prevent infinite polling
    setTimeout(() => clearInterval(pollInterval), 300000);
  };

  const viewResults = () => {
    setShowResults(true);
  };

  const steps = [
    {
      title: 'Start',
      description: 'Initialize comprehensive bias mitigation',
      icon: <PlayCircleOutlined />,
    },
    {
      title: 'Processing',
      description: 'Applying all mitigation strategies',
      icon: loading ? <LoadingOutlined /> : <ThunderboltOutlined />,
    },
    {
      title: 'Evaluation',
      description: 'Evaluating strategy effectiveness',
      icon: <BarChartOutlined />,
    },
    {
      title: 'Complete',
      description: 'Results ready for review',
      icon: <CheckCircleOutlined />,
    },
  ];

  if (showResults && mitigationJob) {
    return (
      <ComprehensiveMitigationResults 
        jobId={mitigationJob.job_id}
        onBack={() => setShowResults(false)}
      />
    );
  }

  return (
    <div className="bias-management">
      <div className="management-header">
        <div className="header-content">
          <Title level={2}>
            <ThunderboltOutlined /> Comprehensive Bias Mitigation
          </Title>
          <Paragraph>
            Automatically apply and evaluate all available bias mitigation strategies to find the best approach for your model.
          </Paragraph>
        </div>
        {onBack && (
          <Button onClick={onBack} className="back-button">
            Back to Analysis
          </Button>
        )}
      </div>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={16}>
          <Card className="main-card">
            <Steps current={currentStep} className="mitigation-steps">
              {steps.map((step, index) => (
                <Step 
                  key={index} 
                  title={step.title} 
                  description={step.description}
                  icon={step.icon}
                />
              ))}
            </Steps>

            {error && (
              <Alert
                message="Mitigation Error"
                description={error}
                type="error"
                showIcon
                className="error-alert"
                action={
                  <Button size="small" onClick={() => setError(null)}>
                    Dismiss
                  </Button>
                }
              />
            )}

            {currentStep === 0 && (
              <div className="start-section">
                <Title level={4}>Ready to Start Comprehensive Mitigation</Title>
                <Paragraph>
                  This process will automatically apply all {strategies.length} available bias mitigation strategies 
                  and evaluate their effectiveness. You'll get:
                </Paragraph>
                <ul className="benefits-list">
                  <li>Bias metrics before and after each strategy</li>
                  <li>Fairness score for each approach</li>
                  <li>Model performance comparison</li>
                  <li>Recommendation for the best strategy</li>
                  <li>Detailed improvement analysis</li>
                </ul>
                
                <div className="action-buttons">
                  <Button 
                    type="primary" 
                    size="large"
                    onClick={startComprehensiveMitigation}
                    disabled={loading}
                    icon={<PlayCircleOutlined />}
                  >
                    Start Comprehensive Mitigation
                  </Button>
                  <Button 
                    onClick={() => setShowStrategyModal(true)}
                    className="view-strategies-btn"
                  >
                    View All Strategies
                  </Button>
                </div>
              </div>
            )}

            {(currentStep === 1 || currentStep === 2) && mitigationJob && (
              <div className="progress-section">
                <Title level={4}>Processing Mitigation Strategies</Title>
                <div className="progress-info">
                  <Text>Status: {mitigationJob.status}</Text>
                  <Text>Job ID: {mitigationJob.job_id}</Text>
                </div>
                <Progress 
                  percent={mitigationJob.progress} 
                  status={mitigationJob.status === 'running' ? 'active' : 'normal'}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
                <Paragraph>
                  Applying and evaluating multiple bias mitigation strategies. This may take several minutes...
                </Paragraph>
              </div>
            )}

            {currentStep === 3 && mitigationJob && (
              <div className="completion-section">
                <div className="completion-header">
                  <TrophyOutlined className="trophy-icon" />
                  <Title level={4}>Mitigation Complete!</Title>
                </div>
                <Paragraph>
                  All mitigation strategies have been applied and evaluated. 
                  View the comprehensive results to see which strategy works best for your model.
                </Paragraph>
                <Button 
                  type="primary" 
                  size="large"
                  onClick={viewResults}
                  icon={<BarChartOutlined />}
                >
                  View Comprehensive Results
                </Button>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="Strategies Overview" className="strategies-overview">
            <div className="strategy-counts">
              <div className="count-item">
                <Badge color="blue" />
                <Text>Preprocessing: {strategies.filter(s => s.type === 'preprocessing').length}</Text>
              </div>
              <div className="count-item">
                <Badge color="green" />
                <Text>In-processing: {strategies.filter(s => s.type === 'inprocessing').length}</Text>
              </div>
              <div className="count-item">
                <Badge color="orange" />
                <Text>Post-processing: {strategies.filter(s => s.type === 'postprocessing').length}</Text>
              </div>
            </div>

            <div className="strategy-list">
              {strategies.slice(0, 4).map((strategy) => (
                <div key={strategy.key} className="strategy-item">
                  <div className="strategy-header">
                    <Text strong>{strategy.name}</Text>
                    <Badge color={getStrategyTypeColor(strategy.type)} text={strategy.type} />
                  </div>
                  <Text className="strategy-description">{strategy.description}</Text>
                </div>
              ))}
              {strategies.length > 4 && (
                <Button 
                  type="link" 
                  onClick={() => setShowStrategyModal(true)}
                  className="view-all-btn"
                >
                  View all {strategies.length} strategies
                </Button>
              )}
            </div>
          </Card>
        </Col>
      </Row>

      <Modal
        title="All Available Mitigation Strategies"
        visible={showStrategyModal}
        onCancel={() => setShowStrategyModal(false)}
        footer={null}
        width={800}
        className="strategies-modal"
      >
        <List
          dataSource={strategies}
          renderItem={(strategy) => (
            <List.Item>
              <List.Item.Meta
                title={
                  <div className="strategy-title">
                    <span>{strategy.name}</span>
                    <Badge color={getStrategyTypeColor(strategy.type)} text={strategy.type} />
                  </div>
                }
                description={
                  <div>
                    <Paragraph>{strategy.description}</Paragraph>
                    <Text type="secondary">Ideal for: {strategy.ideal_for}</Text>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Modal>
    </div>
  );
};

export default BiasManagement;
