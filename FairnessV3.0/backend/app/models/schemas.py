from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
from datetime import datetime


class FileType(str, Enum):
    MODEL = "model"
    TRAIN_DATASET = "train_dataset"
    TEST_DATASET = "test_dataset"


class ModelType(str, Enum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class MitigationStrategy(str, Enum):
    PREPROCESSING = "preprocessing"
    INPROCESSING = "inprocessing"
    POSTPROCESSING = "postprocessing"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    upload_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: FileType
    file_size: int
    status: str = "uploaded"
    message: str = "File uploaded successfully"
    uploaded_at: datetime = Field(default_factory=datetime.now)


class UploadStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    upload_id: str
    filename: str
    file_type: FileType
    status: str
    progress: int = 0
    message: str = ""


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: ModelType
    task_type: TaskType
    framework: str
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    feature_names: Optional[List[str]] = None
    target_column: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None


class SensitiveFeature(BaseModel):
    feature_name: str
    correlation_score: float
    p_value: float
    test_type: str
    significance_level: str
    description: str


class BiasMetric(BaseModel):
    metric_name: str
    value: float
    threshold: float
    is_biased: bool
    severity: str  # low, medium, high
    description: str


class FairnessScore(BaseModel):
    overall_score: float
    bias_score: float
    fairness_level: str  # excellent, good, fair, poor
    metrics_breakdown: Dict[str, float]
    recommendations: List[str]


class AnalysisJob(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_upload_id: str
    train_dataset_upload_id: str
    test_dataset_upload_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisResults(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    model_info: ModelInfo
    sensitive_features: List[SensitiveFeature]
    bias_metrics: List[BiasMetric]
    fairness_score: FairnessScore
    visualizations: Dict[str, Any]  # chart_type -> chart_data
    analysis_summary: str


class MitigationJob(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_job_id: str
    strategy: MitigationStrategy
    strategy_params: Dict[str, Any] = {}
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class MitigationResults(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    strategy_applied: MitigationStrategy
    before_metrics: List[BiasMetric]
    after_metrics: List[BiasMetric]
    improvement_summary: Dict[str, float]
    fairness_improvement: float
    mitigation_details: Dict[str, Any]


class AIRecommendationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    analysis_job_id: str
    mitigation_job_id: Optional[str] = None
    additional_context: Optional[str] = None


class AIRecommendation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recommendations: str
    bias_reduction_strategies: List[str]
    model_retraining_approaches: List[str]
    data_collection_improvements: List[str]
    monitoring_practices: List[str]
    compliance_considerations: List[str]
    confidence_score: float
    generated_at: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
