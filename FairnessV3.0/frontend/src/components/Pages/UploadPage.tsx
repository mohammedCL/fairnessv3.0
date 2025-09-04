import React, { useState, useCallback, useEffect } from 'react';
import { Upload, File, CheckCircle, X, AlertCircle, ArrowRight, Database, Brain } from 'lucide-react';
import { useFairness } from '../../context/FairnessContext';
import { uploadAPI, analysisAPI } from '../../services/api';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import { useNavigate } from 'react-router-dom';

interface FileUploadState {
  isDragging: boolean;
  uploadProgress: Record<string, number>;
  validationErrors: Record<string, string>;
}

type UploadState = FileUploadState;

const UploadPage: React.FC = () => {
  console.log('UploadPage component loaded');
  
  const {
    state,
    dispatch,
    canProceedToAnalysis,
    startAnalysis
  } = useFairness();
  const navigate = useNavigate();
  const [uploadState, setUploadState] = useState<UploadState>({
    isDragging: false,
    uploadProgress: {},
    validationErrors: {}
  });

  // Debug: Log uploaded files whenever state changes
  useEffect(() => {
    console.log('Current uploaded files in state:', state.uploadedFiles);
    console.log('Number of uploaded files:', Object.keys(state.uploadedFiles).length);
  }, [state.uploadedFiles]);

  // Supported file types
  const supportedModelTypes = ['.pkl', '.joblib', '.onnx'];
  const supportedDataTypes = ['.csv', '.json', '.parquet', '.xlsx'];

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setUploadState(prev => ({ ...prev, isDragging: true }));
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setUploadState(prev => ({ ...prev, isDragging: false }));
  }, []);

  const validateFile = (file: File, type: 'model' | 'dataset'): string | null => {
    const maxSize = 100 * 1024 * 1024; // 100MB
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (file.size > maxSize) {
      return 'File size exceeds 100MB limit';
    }

    if (type === 'model' && !supportedModelTypes.includes(extension)) {
      return `Unsupported model format. Supported: ${supportedModelTypes.join(', ')}`;
    }

    if (type === 'dataset' && !supportedDataTypes.includes(extension)) {
      return `Unsupported dataset format. Supported: ${supportedDataTypes.join(', ')}`;
    }

    return null;
  };

  const uploadFile = async (file: File, fileType: 'model' | 'train_dataset' | 'test_dataset') => {
    console.log('uploadFile called with:', file.name, fileType);
    
    const fileId = `${fileType}_${Date.now()}`;
    console.log('Generated fileId:', fileId);
    
    try {
      console.log('Starting upload process...');
      // Set uploading state
      dispatch({ type: 'SET_UPLOADING', payload: true });
      
      // Initialize progress
      setUploadState(prev => ({
        ...prev,
        uploadProgress: { ...prev.uploadProgress, [fileId]: 0 }
      }));

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadState(prev => ({
          ...prev,
          uploadProgress: {
            ...prev.uploadProgress,
            [fileId]: Math.min(prev.uploadProgress[fileId] + Math.random() * 30, 90)
          }
        }));
      }, 500);

      // Upload file based on type
      let result;
      switch (fileType) {
        case 'model':
          result = await uploadAPI.uploadModel(file);
          break;
        case 'train_dataset':
          result = await uploadAPI.uploadTrainDataset(file);
          break;
        case 'test_dataset':
          result = await uploadAPI.uploadTestDataset(file);
          break;
        default:
          throw new Error('Invalid file type');
      }
      
      // Complete progress
      clearInterval(progressInterval);
      setUploadState(prev => ({
        ...prev,
        uploadProgress: { ...prev.uploadProgress, [fileId]: 100 }
      }));

      // Add to state
      console.log('Upload result:', result);
      console.log('Adding to state:', result);
      dispatch({ type: 'ADD_UPLOADED_FILE', payload: result });
      
      console.log('Current uploaded files after dispatch:', state.uploadedFiles);
      
      toast.success(`${file.name} uploaded successfully!`);
      
      // Clean up progress after delay
      setTimeout(() => {
        setUploadState(prev => {
          const newProgress = { ...prev.uploadProgress };
          delete newProgress[fileId];
          return { ...prev, uploadProgress: newProgress };
        });
      }, 2000);

    } catch (error: any) {
      console.error('Upload failed:', error);
      
      setUploadState(prev => ({
        ...prev,
        validationErrors: {
          ...prev.validationErrors,
          [fileId]: error.message || 'Upload failed'
        }
      }));
      
      toast.error(`Failed to upload ${file.name}`);
      
      // Clean up progress
      setUploadState(prev => {
        const newProgress = { ...prev.uploadProgress };
        delete newProgress[fileId];
        return { ...prev, uploadProgress: newProgress };
      });
    } finally {
      dispatch({ type: 'SET_UPLOADING', payload: false });
    }
  };

  const handleFileDrop = useCallback(async (e: React.DragEvent, fileType: 'model' | 'train_dataset' | 'test_dataset') => {
    console.log('handleFileDrop called with fileType:', fileType);
    e.preventDefault();
    setUploadState(prev => ({ ...prev, isDragging: false }));

    const files = Array.from(e.dataTransfer.files);
    const file = files[0]; // Take only the first file

    console.log('Dropped file:', file?.name);

    if (!file) {
      console.log('No file found in drop event');
      return;
    }

    const validationType = fileType === 'model' ? 'model' : 'dataset';
    const error = validateFile(file, validationType);
    
    if (error) {
      console.log('Validation error:', error);
      toast.error(error);
      return;
    }

    console.log('File validation passed, calling uploadFile...');
    await uploadFile(file, fileType);
  }, [uploadFile, validateFile]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>, fileType: 'model' | 'train_dataset' | 'test_dataset') => {
    console.log('handleFileSelect called with fileType:', fileType);
    const file = e.target.files?.[0];
    console.log('Selected file:', file?.name);
    
    if (!file) {
      console.log('No file selected');
      return;
    }

    const validationType = fileType === 'model' ? 'model' : 'dataset';
    const error = validateFile(file, validationType);
    
    if (error) {
      console.log('Validation error:', error);
      toast.error(error);
      return;
    }

    console.log('File validation passed, calling uploadFile...');
    await uploadFile(file, fileType);
    
    // Reset input
    e.target.value = '';
  }, [uploadFile, validateFile]);

  const removeFile = (uploadId: string) => {
    // This would typically call an API to delete the file
    const newFiles = { ...state.uploadedFiles };
    delete newFiles[uploadId];
    // Update state (you'd need to add this action to your reducer)
    toast.success('File removed');
  };

  const proceedToAnalysis = async () => {
    console.log('🎯 Proceed to Analysis clicked');
    
    if (!canProceedToAnalysis) {
      console.log('❌ Cannot proceed to analysis - missing files');
      toast.error('Please upload both a model and training dataset before proceeding.');
      return;
    }

    // Check if analysis is already running
    if (state.currentAnalysisJobId) {
      console.log('⚠️ Analysis already running with job ID:', state.currentAnalysisJobId);
      navigate('/analysis');
      return;
    }

    const uploadIds = Object.keys(state.uploadedFiles);
    console.log('📤 Starting analysis with upload IDs:', uploadIds);

    const files = Object.values(state.uploadedFiles);
    const modelFile = files.find(f => f.file_type === 'model');
    const trainDataFile = files.find(f => f.file_type === 'train_dataset');
    const testDataFile = files.find(f => f.file_type === 'test_dataset');

    if (!modelFile || !trainDataFile) {
      toast.error('Required files not found');
      return;
    }

    try {
      const response = await analysisAPI.startAnalysis(
        modelFile.upload_id,
        trainDataFile.upload_id,
        testDataFile?.upload_id
      );
      console.log('✅ Analysis started:', response);
      
      // Update state with job ID
      startAnalysis(response.job_id);
      
      // Update workflow steps
      dispatch({ type: 'SET_CURRENT_STEP', payload: 2 });
      dispatch({ type: 'COMPLETE_STEP', payload: 1 });

      toast.success('Analysis started successfully!');
      
      // Navigate to analysis page
      navigate('/analysis');
    } catch (error) {
      console.error('❌ Error starting analysis:', error);
      toast.error('Failed to start analysis. Please try again.');
    }
  };

  const renderFileUploadZone = (
    title: string,
    description: string,
    fileType: 'model' | 'train_dataset' | 'test_dataset',
    icon: React.ReactNode,
    acceptedFormats: string[]
  ) => {
    const existingFile = Object.values(state.uploadedFiles).find(f => f.file_type === fileType);
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-4">
          {icon}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
          </div>
        </div>

        {existingFile ? (
          // Show uploaded file
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-green-800 dark:text-green-200">
                    {existingFile.filename}
                  </p>
                  <p className="text-xs text-green-600 dark:text-green-400">
                    {(existingFile.file_size / 1024 / 1024).toFixed(1)} MB • Uploaded {new Date(existingFile.uploaded_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <button
                onClick={() => removeFile(existingFile.upload_id)}
                className="p-1 text-green-600 hover:text-red-600 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        ) : (
          // Show upload zone
          <div
            className={clsx(
              'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
              uploadState.isDragging
                ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleFileDrop(e, fileType)}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              Drop your file here
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              or click to browse files
            </p>
            
            <input
              type="file"
              accept={acceptedFormats.join(',')}
              onChange={(e) => handleFileSelect(e, fileType)}
              className="hidden"
              id={`file-input-${fileType}`}
            />
            <label
              htmlFor={`file-input-${fileType}`}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 cursor-pointer"
            >
              Select File
            </label>
            
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
              Supported formats: {acceptedFormats.join(', ')} • Max size: 100MB
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Page Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Upload Files
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Upload your machine learning model and datasets to begin the fairness assessment process.
          </p>
        </div>
      </div>

      {/* Upload Instructions */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-8">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-6 w-6 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div>
            <h3 className="text-lg font-medium text-blue-900 dark:text-blue-100 mb-2">
              Before You Begin
            </h3>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <li>• <strong>Model file</strong> is required (trained ML model in .pkl, .joblib, or .onnx format)</li>
              <li>• <strong>Training dataset</strong> is required (data used to train the model)</li>
              <li>• <strong>Test dataset</strong> is optional (if not provided, we'll split the training data)</li>
              <li>• Ensure your datasets include the target column and any sensitive attributes</li>
            </ul>
          </div>
        </div>
      </div>

      {/* File Upload Zones */}
      <div className="space-y-6 mb-8">
        {/* Model Upload */}
        {renderFileUploadZone(
          'Machine Learning Model',
          'Upload your trained model file',
          'model',
          <Brain className="h-6 w-6 text-purple-600" />,
          supportedModelTypes
        )}

        {/* Training Dataset Upload */}
        {renderFileUploadZone(
          'Training Dataset',
          'Upload the dataset used to train your model',
          'train_dataset',
          <Database className="h-6 w-6 text-green-600" />,
          supportedDataTypes
        )}

        {/* Test Dataset Upload */}
        {renderFileUploadZone(
          'Test Dataset (Optional)',
          'Upload a separate test dataset for evaluation',
          'test_dataset',
          <File className="h-6 w-6 text-blue-600" />,
          supportedDataTypes
        )}
      </div>

      {/* Upload Progress */}
      {Object.keys(uploadState.uploadProgress).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Upload Progress
          </h3>
          <div className="space-y-3">
            {Object.entries(uploadState.uploadProgress).map(([fileId, progress]) => (
              <div key={fileId}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600 dark:text-gray-400">Uploading...</span>
                  <span className="text-gray-900 dark:text-white">{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Uploaded Files Summary */}
      {Object.keys(state.uploadedFiles).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Uploaded Files ({Object.keys(state.uploadedFiles).length})
          </h3>
          <div className="space-y-3">
            {Object.values(state.uploadedFiles).map((file) => (
              <div key={file.upload_id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={clsx(
                    'p-2 rounded-lg',
                    file.file_type === 'model' ? 'bg-purple-100 dark:bg-purple-900/30' :
                    file.file_type === 'train_dataset' ? 'bg-green-100 dark:bg-green-900/30' :
                    'bg-blue-100 dark:bg-blue-900/30'
                  )}>
                    {file.file_type === 'model' ? (
                      <Brain className={clsx('h-5 w-5', 'text-purple-600 dark:text-purple-400')} />
                    ) : (
                      <Database className={clsx('h-5 w-5', 
                        file.file_type === 'train_dataset' ? 'text-green-600 dark:text-green-400' : 'text-blue-600 dark:text-blue-400'
                      )} />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">{file.filename}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {file.file_type.replace('_', ' ').toUpperCase()} • {(file.file_size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </div>
                </div>
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next Step Button */}
      <div className="flex justify-end">
        <button
          onClick={proceedToAnalysis}
          disabled={!canProceedToAnalysis || state.analyzing}
          className={clsx(
            'inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md transition-colors',
            canProceedToAnalysis && !state.analyzing
              ? 'text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
              : 'text-gray-400 bg-gray-200 dark:bg-gray-700 cursor-not-allowed'
          )}
        >
          {state.analyzing ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Starting Analysis...
            </>
          ) : (
            <>
              Proceed to Analysis
              <ArrowRight className="ml-2 h-5 w-5" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default UploadPage;
