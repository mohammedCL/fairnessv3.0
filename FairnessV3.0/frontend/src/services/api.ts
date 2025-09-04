import axios from 'axios';
import { UploadedFile, AnalysisJob, AnalysisResults, MitigationJob, MitigationResults, AIRecommendation } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Upload API
export const uploadAPI = {
  uploadModel: async (file: File): Promise<UploadedFile> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload/model', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  uploadTrainDataset: async (file: File): Promise<UploadedFile> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload/train-dataset', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  uploadTestDataset: async (file: File): Promise<UploadedFile> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload/test-dataset', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  getUploadStatus: async (uploadId: string) => {
    const response = await api.get(`/api/upload/status/${uploadId}`);
    return response.data;
  },

  listUploads: async () => {
    const response = await api.get('/api/upload/list');
    return response.data.uploads;
  },

  deleteUpload: async (uploadId: string) => {
    const response = await api.delete(`/api/upload/${uploadId}`);
    return response.data;
  },
};

// Analysis API
export const analysisAPI = {
  startAnalysis: async (
    modelUploadId: string,
    trainDatasetUploadId: string,
    testDatasetUploadId?: string
  ): Promise<AnalysisJob> => {
    const params = new URLSearchParams({
      model_upload_id: modelUploadId,
      train_dataset_upload_id: trainDatasetUploadId,
    });
    
    if (testDatasetUploadId) {
      params.append('test_dataset_upload_id', testDatasetUploadId);
    }
    
    const response = await api.post(`/api/analysis/start?${params.toString()}`);
    return response.data;
  },

  getAnalysisJob: async (jobId: string): Promise<AnalysisJob> => {
    const response = await api.get(`/api/analysis/job/${jobId}`);
    return response.data;
  },

  getAnalysisResults: async (jobId: string): Promise<AnalysisResults> => {
    const response = await api.get(`/api/analysis/results/${jobId}`);
    return response.data;
  },

  getBiasMetrics: async (jobId: string) => {
    const response = await api.get(`/api/analysis/bias-metrics/${jobId}`);
    return response.data;
  },

  getFairnessScore: async (jobId: string) => {
    const response = await api.get(`/api/analysis/fairness-score/${jobId}`);
    return response.data;
  },

  getVisualizations: async (jobId: string) => {
    const response = await api.get(`/api/analysis/visualizations/${jobId}`);
    return response.data;
  },

  getAnalysisSummary: async (jobId: string) => {
    const response = await api.get(`/api/analysis/summary/${jobId}`);
    return response.data;
  },

  listAnalysisJobs: async () => {
    const response = await api.get('/api/analysis/list');
    return response.data.jobs;
  },
};

// Mitigation API
export const mitigationAPI = {
  startMitigation: async (
    analysisJobId: string,
    strategy: 'preprocessing' | 'inprocessing' | 'postprocessing',
    strategyParams?: Record<string, any>
  ): Promise<MitigationJob> => {
    const params = new URLSearchParams({
      analysis_job_id: analysisJobId,
      strategy: strategy,
    });
    
    const response = await api.post(`/api/mitigation/start?${params.toString()}`, {
      strategy_params: strategyParams || {},
    });
    return response.data;
  },

  getMitigationJob: async (jobId: string): Promise<MitigationJob> => {
    const response = await api.get(`/api/mitigation/job/${jobId}`);
    return response.data;
  },

  getMitigationResults: async (jobId: string): Promise<MitigationResults> => {
    const response = await api.get(`/api/mitigation/results/${jobId}`);
    return response.data;
  },

  getBeforeAfterComparison: async (jobId: string) => {
    const response = await api.get(`/api/mitigation/comparison/${jobId}`);
    return response.data;
  },

  getImprovementSummary: async (jobId: string) => {
    const response = await api.get(`/api/mitigation/improvement-summary/${jobId}`);
    return response.data;
  },

  getAvailableStrategies: async () => {
    const response = await api.get('/api/mitigation/strategies');
    return response.data.strategies;
  },

  listMitigationJobs: async () => {
    const response = await api.get('/api/mitigation/list');
    return response.data.jobs;
  },
};

// AI Recommendations API
export const aiAPI = {
  generateRecommendations: async (
    analysisJobId: string,
    mitigationJobId?: string,
    additionalContext?: string
  ): Promise<AIRecommendation> => {
    const requestData = {
      analysis_job_id: analysisJobId,
      mitigation_job_id: mitigationJobId,
      additional_context: additionalContext,
    };
    
    const response = await api.post('/api/ai/generate', requestData);
    return response.data;
  },

  getRecommendation: async (jobId: string): Promise<AIRecommendation> => {
    const response = await api.get(`/api/ai/recommendation/${jobId}`);
    return response.data;
  },

  chatFollowup: async (recommendationJobId: string, question: string) => {
    const requestData = {
      recommendation_job_id: recommendationJobId,
      question: question,
    };
    
    const response = await api.post('/api/ai/chat', requestData);
    return response.data;
  },

  getAIServiceStatus: async () => {
    const response = await api.get('/api/ai/status');
    return response.data;
  },

  listAIRecommendations: async () => {
    const response = await api.get('/api/ai/list');
    return response.data.recommendations;
  },
};

// Health check
export const healthAPI = {
  checkHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
