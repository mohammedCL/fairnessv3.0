export interface UploadedFile {
  upload_id: string;
  filename: string;
  file_type: 'model' | 'train_dataset' | 'test_dataset';
  file_size: number;
  status: string;
  uploaded_at: string;
}

export interface ModelInfo {
  model_type: string;
  task_type: string;
  framework: string;
  n_features?: number;
  n_classes?: number;
  feature_names?: string[];
  target_column?: string;
}

export interface SensitiveFeature {
  feature_name: string;
  correlation_score: number;
  p_value: number;
  test_type: string;
  significance_level: string;
  description: string;
}

export interface BiasMetric {
  metric_name: string;
  value: number;
  threshold: number;
  is_biased: boolean;
  severity: 'low' | 'medium' | 'high';
  description: string;
  // Advanced bias analysis properties
  p_value?: number;
  statistical_test?: string;
  test_statistic?: number;
  degrees_of_freedom?: number;
  effect_size?: string;
  confidence_interval?: [number, number];
  correlation_coefficient?: number;
  prediction_accuracy?: number;
  risk_level?: 'low' | 'medium' | 'high';
  attribute_combination?: string;
  interaction_strength?: 'weak' | 'moderate' | 'strong';
  sensitive_attribute?: string;
  feature_type?: string;
  analysis_category?: 'Direct Bias' | 'Proxy Features' | 'Statistical Tests' | 'Model-Based' | 'Multi-Attribute';
}

export interface FairnessScore {
  overall_score: number;
  bias_score: number;
  fairness_level: 'excellent' | 'good' | 'fair' | 'poor';
  metrics_breakdown: Record<string, number>;
  recommendations: string[];
  assessment?: string;
}

export interface AnalysisJob {
  job_id: string;
  model_upload_id: string;
  train_dataset_upload_id: string;
  test_dataset_upload_id?: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface AnalysisResults {
  job_id: string;
  model_info: ModelInfo;
  sensitive_features: SensitiveFeature[];
  bias_metrics: BiasMetric[];
  fairness_score: FairnessScore;
  visualizations: Record<string, any>;
  analysis_summary: string;
}

export interface MitigationJob {
  job_id: string;
  analysis_job_id: string;
  strategy: 'preprocessing' | 'inprocessing' | 'postprocessing';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
}

export interface MitigationResults {
  job_id: string;
  strategy_applied: string;
  before_metrics: BiasMetric[];
  after_metrics: BiasMetric[];
  improvement_summary: Record<string, number>;
  fairness_improvement: number;
  mitigation_details: Record<string, any>;
}

export interface AIRecommendation {
  job_id: string;
  recommendations: string;
  bias_reduction_strategies: string[];
  model_retraining_approaches: string[];
  data_collection_improvements: string[];
  monitoring_practices: string[];
  compliance_considerations: string[];
  confidence_score: number;
  generated_at: string;
}

export interface WorkflowStep {
  id: number;
  name: string;
  description: string;
  completed: boolean;
  current: boolean;
}

export interface FairnessState {
  // Loading states
  uploading: boolean;
  analyzing: boolean;
  mitigating: boolean;
  
  // Data
  uploadedFiles: Record<string, UploadedFile>;
  analysisResults?: AnalysisResults;
  mitigationResults?: MitigationResults;
  aiRecommendations?: AIRecommendation;
  currentStep: number;
  workflowSteps: WorkflowStep[];
  
  // Current job IDs
  currentAnalysisJobId?: string;
  
  // Error handling
  error?: string;
}
