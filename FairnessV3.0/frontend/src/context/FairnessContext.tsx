import React, { createContext, useContext, useReducer, ReactNode, useEffect } from 'react';
import { FairnessState, WorkflowStep, UploadedFile, AnalysisResults, MitigationResults, AIRecommendation } from '../types';

// Initial workflow steps
const initialWorkflowSteps: WorkflowStep[] = [
  { id: 1, name: 'Upload Files', description: 'Upload model and datasets', completed: false, current: true },
  { id: 2, name: 'Analysis', description: 'Bias detection and analysis', completed: false, current: false },
  { id: 3, name: 'Mitigation', description: 'Apply bias mitigation strategies', completed: false, current: false },
  { id: 4, name: 'Monitoring', description: 'Ongoing fairness monitoring', completed: false, current: false },
];

// Initial state
const initialState: FairnessState = {
  // Loading states
  uploading: false,
  analyzing: false,
  mitigating: false,
  
  // Data
  uploadedFiles: {},
  currentStep: 1,
  workflowSteps: initialWorkflowSteps,
};

// Action types
type FairnessAction =
  | { type: 'SET_UPLOADING'; payload: boolean }
  | { type: 'SET_ANALYZING'; payload: boolean }
  | { type: 'SET_MITIGATING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | undefined }
  | { type: 'ADD_UPLOADED_FILE'; payload: UploadedFile }
  | { type: 'SET_ANALYSIS_RESULTS'; payload: AnalysisResults }
  | { type: 'SET_MITIGATION_RESULTS'; payload: MitigationResults }
  | { type: 'SET_AI_RECOMMENDATIONS'; payload: AIRecommendation }
  | { type: 'SET_CURRENT_STEP'; payload: number }
  | { type: 'COMPLETE_STEP'; payload: number }
  | { type: 'START_ANALYSIS'; payload: string } // job ID
  | { type: 'COMPLETE_ANALYSIS'; payload?: AnalysisResults }
  | { type: 'RESET_STATE' };

// Reducer
function fairnessReducer(state: FairnessState, action: FairnessAction): FairnessState {
  let newState: FairnessState;

  switch (action.type) {
    case 'SET_UPLOADING':
      newState = { ...state, uploading: action.payload };
      break;
    
    case 'SET_ANALYZING':
      newState = { ...state, analyzing: action.payload };
      break;
    
    case 'SET_MITIGATING':
      newState = { ...state, mitigating: action.payload };
      break;
    
    case 'SET_ERROR':
      newState = { ...state, error: action.payload };
      break;

    case 'ADD_UPLOADED_FILE':
      newState = {
        ...state,
        uploadedFiles: {
          ...state.uploadedFiles,
          [action.payload.upload_id]: action.payload,
        },
      };
      break;

    case 'SET_ANALYSIS_RESULTS':
      newState = {
        ...state,
        analysisResults: action.payload,
      };
      break;

    case 'SET_MITIGATION_RESULTS':
      newState = {
        ...state,
        mitigationResults: action.payload,
      };
      break;

    case 'SET_AI_RECOMMENDATIONS':
      newState = {
        ...state,
        aiRecommendations: action.payload,
      };
      break;

    case 'SET_CURRENT_STEP':
      newState = {
        ...state,
        currentStep: action.payload,
        workflowSteps: state.workflowSteps.map(step => ({
          ...step,
          current: step.id === action.payload,
        })),
      };
      break;

    case 'COMPLETE_STEP':
      newState = {
        ...state,
        workflowSteps: state.workflowSteps.map(step =>
          step.id === action.payload
            ? { ...step, completed: true, current: false }
            : step
        ),
      };
      break;

    case 'START_ANALYSIS':
      newState = {
        ...state,
        currentAnalysisJobId: action.payload,
        analyzing: true,
      };
      break;

    case 'COMPLETE_ANALYSIS':
      newState = {
        ...state,
        currentAnalysisJobId: undefined,
        analyzing: false,
        ...(action.payload && { analysisResults: action.payload }),
      };
      break;

    case 'RESET_STATE':
      newState = initialState;
      break;

    default:
      newState = state;
      break;
  }

  return newState;
}

// Context
interface FairnessContextType {
  state: FairnessState;
  dispatch: React.Dispatch<FairnessAction>;
  
  // Helper functions
  addUploadedFile: (file: UploadedFile) => void;
  setAnalysisResults: (results: AnalysisResults) => void;
  setMitigationResults: (results: MitigationResults) => void;
  setAIRecommendations: (recommendations: AIRecommendation) => void;
  setCurrentStep: (step: number) => void;
  completeStep: (step: number) => void;
  startAnalysis: (jobId: string) => void;
  completeAnalysis: (results?: AnalysisResults) => void;
  resetState: () => void;
  
  // Computed values
  hasModelFile: boolean;
  hasTrainDataset: boolean;
  hasTestDataset: boolean;
  canProceedToAnalysis: boolean;
  canProceedToMitigation: boolean;
}

const FairnessContext = createContext<FairnessContextType | undefined>(undefined);

// Provider component
interface FairnessProviderProps {
  children: ReactNode;
}

export function FairnessProvider({ children }: FairnessProviderProps) {
  // Initialize state from sessionStorage (cleared when browser closes)
  const getInitialState = (): FairnessState => {
    if (typeof window !== 'undefined') {
      try {
        const saved = sessionStorage.getItem('fairness-session');
        if (saved) {
          const parsed = JSON.parse(saved);
          console.log('Restored session data for step:', parsed.currentStep);
          return parsed;
        }
      } catch (error) {
        console.warn('Failed to restore session data:', error);
      }
    }
    
    return initialState;
  };

  const [state, dispatch] = useReducer(fairnessReducer, getInitialState());

  // Save to sessionStorage whenever state changes (cleared on browser close)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        sessionStorage.setItem('fairness-session', JSON.stringify(state));
      } catch (error) {
        console.warn('Failed to save session data:', error);
      }
    }
  }, [state]);

  // Clear session when user explicitly resets (goes to home)
  const resetSession = () => {
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem('fairness-session');
    }
    dispatch({ type: 'RESET_STATE' });
  };

  // Helper functions
  const addUploadedFile = (file: UploadedFile) => {
    dispatch({ type: 'ADD_UPLOADED_FILE', payload: file });
  };

  const setAnalysisResults = (results: AnalysisResults) => {
    dispatch({ type: 'SET_ANALYSIS_RESULTS', payload: results });
  };

  const setMitigationResults = (results: MitigationResults) => {
    dispatch({ type: 'SET_MITIGATION_RESULTS', payload: results });
  };

  const setAIRecommendations = (recommendations: AIRecommendation) => {
    dispatch({ type: 'SET_AI_RECOMMENDATIONS', payload: recommendations });
  };

  const setCurrentStep = (step: number) => {
    dispatch({ type: 'SET_CURRENT_STEP', payload: step });
  };

  const completeStep = (step: number) => {
    dispatch({ type: 'COMPLETE_STEP', payload: step });
  };

  const startAnalysis = (jobId: string) => {
    dispatch({ type: 'START_ANALYSIS', payload: jobId });
  };

  const completeAnalysis = (results?: AnalysisResults) => {
    dispatch({ type: 'COMPLETE_ANALYSIS', payload: results });
  };

  // Computed values
  const uploadedFilesArray = Object.values(state.uploadedFiles);
  const hasModelFile = uploadedFilesArray.some(file => file.file_type === 'model');
  const hasTrainDataset = uploadedFilesArray.some(file => file.file_type === 'train_dataset');
  const hasTestDataset = uploadedFilesArray.some(file => file.file_type === 'test_dataset');
  const canProceedToAnalysis = hasModelFile && hasTrainDataset;
  const canProceedToMitigation = !!state.analysisResults;

  const value: FairnessContextType = {
    state,
    dispatch,
    addUploadedFile,
    setAnalysisResults,
    setMitigationResults,
    setAIRecommendations,
    setCurrentStep,
    completeStep,
    startAnalysis,
    completeAnalysis,
    resetState: resetSession, // Use resetSession instead of resetState
    hasModelFile,
    hasTrainDataset,
    hasTestDataset,
    canProceedToAnalysis,
    canProceedToMitigation,
  };

  return <FairnessContext.Provider value={value}>{children}</FairnessContext.Provider>;
}

// Custom hook to use the context
export function useFairness() {
  const context = useContext(FairnessContext);
  if (context === undefined) {
    throw new Error('useFairness must be used within a FairnessProvider');
  }
  return context;
}
